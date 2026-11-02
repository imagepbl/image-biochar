from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUTCOMES = {
    "yield": ("lnRR_Yield", "Var_Yield"),
    "n2o": ("lnRR_N2O", "Var_N2O"),
    "ch4": ("lnRR_CH4", "Var_CH4"),
}
COL = {
    "ph": "pH", "soc": "initial SOC (g/kg)", "tn": "initial TN (g/kg)",
    "dose": "Total biochar rate (t/ha)", "clay": "Clay (%)",
    "nrate": "Seasonal N rate (kg/ha)",
}
TERMS = {
    "yield": ["Intercept", "land_upland", "ph_z", "dose_z", "tn_z:dose_z",
              "soc_z:paddy", "soc_z:upland"],
    "n2o": ["Intercept", "ph_z", "clay_z", "nrate_z"],
    "ch4": ["Intercept", "is_acidic", "dose_z"],
}
PRIORS = {
    "yield": {"beta_sd": [.30, .19, .19, .19, .13, .13, .13],
              "study_sd": .28, "tau_sd": .28},
    "n2o": {"beta_sd": [.43, .27, .27, .27],
            "study_sd": .48, "tau_sd": .48},
    "ch4": {"beta_sd": [.42, .25, .25], "study_sd": .42, "tau_sd": .42},
}

IMAGE_TERMS = {
    "yield": [
        "Intercept", "C(Land_use)[T.upland]", "Q('pH_c')", "rate_log_c",
        "Q('initial TN (g/kg)_c'):rate_log_c",
        "Q('initial SOC (g/kg)_c'):C(Land_use)[paddy]",
        "Q('initial SOC (g/kg)_c'):C(Land_use)[upland]",
    ],
    "n2o": [
        "Intercept", "C(Land_use)[T.upland]", "Q('pH_c')",
        "Q('Clay (%)_c')", "Q('Seasonal N rate (kg/ha)_c')",
    ],
    "ch4": ["Intercept", "is_acidic", "rate_log"],
}

def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.15g")

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def prepare_data(workbook: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Apply the original final-fit sample filters, including vegetable -> upland."""
    if outcome not in OUTCOMES:
        raise ValueError(f"Unknown response: {outcome}")
    response, variance = OUTCOMES[outcome]
    d = workbook.copy()
    d["row_id"] = np.arange(len(d), dtype=np.int64) + 2
    required = [response, variance, "No.", "Land_use", COL["ph"]]
    if outcome == "yield":
        required += [COL["soc"], COL["tn"], COL["dose"]]
    elif outcome == "n2o":
        required += [COL["clay"], COL["nrate"]]
    else:
        required += [COL["dose"]]
    absent = sorted(set(required) - set(d.columns))
    if absent:
        raise KeyError(f"Missing source columns: {absent}")

    d["Land_use"] = d["Land_use"].astype("string").str.strip().str.lower()
    d["land_use_in_source"] = d["Land_use"].copy()
    d["Land_use"] = d["Land_use"].replace({"vegetable": "upland"})
    d["study_id"] = d["No."].astype("string").str.strip()
    d.loc[d["No."].isna(), "study_id"] = pd.NA
    for col in required:
        if col not in ("No.", "Land_use"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=required + ["study_id"]).copy()
    d = d[np.isfinite(d[response]) & np.isfinite(d[variance]) & (d[variance] > 0)].copy()
    d = d[d["Land_use"].isin(["paddy", "upland"])].copy()
    if outcome == "yield":
        d = d[d[response] < 2].copy()  # Keep the selected one-sided lnRR filter.
    elif outcome == "n2o":
        d = d[d["Land_use"].eq("upland")].copy()
    else:
        d = d[d["Land_use"].eq("paddy")].copy()  # No duration exclusion.
    d = d[d[COL["ph"]].between(0, 14)].copy()
    for col in (COL["dose"], COL["soc"], COL["tn"], COL["clay"], COL["nrate"]):
        if col in required:
            d = d[d[col] >= 0].copy()
    if COL["clay"] in required:
        d = d[d[COL["clay"]] <= 100].copy()
    d = d.sort_values("row_id").reset_index(drop=True)
    if len(d) < 25 or d["study_id"].nunique() < 3:
        raise ValueError(f"Insufficient calibration observations/studies for {outcome}")
    return d

def predictors(d: pd.DataFrame, outcome: str) -> dict[str, np.ndarray]:
    def numeric(column: str) -> np.ndarray:
        value = np.asarray(pd.to_numeric(d[column], errors="coerce"), dtype=float)
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite predictor: {column}")
        return value

    if outcome == "yield":
        return {"ph": numeric(COL["ph"]), "soc": numeric(COL["soc"]),
                "tn": numeric(COL["tn"]), "dose_log": np.log1p(numeric(COL["dose"]))}
    if outcome == "n2o":
        return {"ph": numeric(COL["ph"]), "clay": numeric(COL["clay"]),
                "nrate": numeric(COL["nrate"])}
    return {"dose_log": np.log1p(numeric(COL["dose"])), "ph": numeric(COL["ph"])}

def build_transform(d: pd.DataFrame, outcome: str) -> dict:
    value = predictors(d, outcome)
    keys = {"yield": ("ph", "soc", "tn", "dose_log"),
            "n2o": ("ph", "clay", "nrate"), "ch4": ("dose_log",)}[outcome]
    centers, scales = {}, {}
    for key in keys:
        anchor = {"ph": 6.5, "dose_log": float(np.log1p(20.0))}.get(key)
        centers[key] = float(value[key].mean()) if anchor is None else anchor
        scales[key] = float(value[key].std(ddof=0))
        if not np.isfinite(scales[key]) or scales[key] < 1e-10:
            raise ValueError(f"Degenerate moderator scale: {key}")
    return {
        "mode": "standardised_reference_v2", "centres": centers, "scales": scales,
        # IMPORTANT: the TN-by-dose interaction uses the sample mean log-dose,
        # while the main dose term is centred at IMAGE's 20 t/ha reference.
        "interaction_dose_centre": (float(value["dose_log"].mean())
                                    if "dose_log" in value else None),
        "dose_reference_t_ha": 20.0,
    }

def build_design(d: pd.DataFrame, outcome: str, tf: dict) -> tuple[np.ndarray, list[str]]:
    value = predictors(d, outcome)
    z = {k: (value[k] - tf["centres"][k]) / tf["scales"][k]
         for k in tf["centres"]}
    one = np.ones(len(d))
    if outcome == "yield":
        land = d["Land_use"].astype("string").str.strip().str.lower()
        land = land.replace({"vegetable": "upland"})
        if not land.isin(["paddy", "upland"]).all():
            raise ValueError("Yield observations must be paddy/upland")
        upland = land.eq("upland").to_numpy(dtype=float)
        paddy = land.eq("paddy").to_numpy(dtype=float)
        dose_inter = ((value["dose_log"] - tf["interaction_dose_centre"])
                      / tf["scales"]["dose_log"])
        parts = [one, upland, z["ph"], z["dose_log"], z["tn"] * dose_inter,
                 z["soc"] * paddy, z["soc"] * upland]
    elif outcome == "n2o":
        parts = [one, z["ph"], z["clay"], z["nrate"]]
    else:
        parts = [one, (value["ph"] < 6.5).astype(float), z["dose_log"]]
    design = np.column_stack(parts)
    if design.shape[1] != len(TERMS[outcome]) or not np.isfinite(design).all():
        raise ValueError(f"Invalid design matrix for {outcome}")
    return design, TERMS[outcome]

def fit_bayesian_model(d: pd.DataFrame, outcome: str, X: np.ndarray,
                       *, draws: int, tune: int, chains: int,
                       target_accept: float, seed: int):
    import pymc as pm

    yname, vname = OUTCOMES[outcome]
    y, v = d[yname].to_numpy(float), d[vname].to_numpy(float)
    study, study_idx = np.unique(d["study_id"].astype(str), return_inverse=True)
    priors = PRIORS[outcome]
    if (not np.isfinite(X).all() or not np.isfinite(y).all()
            or not np.isfinite(v).all() or (v <= 0).any()):
        raise ValueError("Non-finite design/response/variance, or nonpositive variance")
    if np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError(f"Rank-deficient design for {outcome}")
    if len(priors["beta_sd"]) != X.shape[1]:
        raise ValueError("Prior vector does not match fixed-effect design")
    with pm.Model():
        beta = pm.Normal("beta", mu=0., sigma=np.asarray(priors["beta_sd"]),
                         shape=X.shape[1])
        sigma_u = pm.HalfNormal("sigma_u", sigma=priors["study_sd"])
        u = pm.Normal("u", mu=0., sigma=sigma_u, shape=len(study))
        tau = pm.HalfNormal("tau", sigma=priors["tau_sd"])
        mu = pm.math.dot(X, beta) + u[study_idx]
        pm.Normal("y_obs", mu=mu, sigma=pm.math.sqrt(v + tau**2), observed=y)
        return pm.sample(draws=draws, tune=tune, chains=chains,
                         target_accept=target_accept, random_seed=seed,
                         return_inferencedata=True)

def posterior_beta(idata) -> np.ndarray:
    beta = np.asarray(idata.posterior["beta"].values, float)
    return beta.reshape(-1, beta.shape[-1])

def sampling_diagnostics(idata) -> dict:
    import arviz as az
    selected = ["beta", "sigma_u", "tau"]
    rhat_ds = az.rhat(idata, var_names=selected)
    ess_ds = az.ess(idata, var_names=selected, method="bulk")
    rhat = np.concatenate([np.asarray(v.values).ravel() for v in rhat_ds.data_vars.values()])
    ess = np.concatenate([np.asarray(v.values).ravel() for v in ess_ds.data_vars.values()])
    bfmi = np.asarray(az.bfmi(idata), float)
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    if not (np.isfinite(rhat).all() and np.isfinite(ess).all() and np.isfinite(bfmi).all()):
        raise ValueError("Invalid sampling diagnostics")
    max_rhat, min_ess, min_bfmi = float(rhat.max()), float(ess.min()), float(bfmi.min())
    return {
        "max_rhat": max_rhat, "min_ess_bulk": min_ess,
        "divergences": divergences, "min_bfmi": min_bfmi,
        "n_draws": int(posterior_beta(idata).shape[0]),
        "healthy": bool(max_rhat < 1.01 and min_ess > 400
                        and divergences == 0 and min_bfmi > .3),
    }

def posterior_summary(idata, names: list[str]) -> pd.DataFrame:
    entries = []
    beta = posterior_beta(idata)
    for j, name in enumerate(names):
        entries.append((name, beta[:, j]))
    for key in ("sigma_u", "tau"):
        entries.append((key, np.asarray(idata.posterior[key].values, float).ravel()))
    return pd.DataFrame([
        {"parameter": name, "mean": float(draws.mean()),
         "sd": float(draws.std(ddof=1)), "q2.5": float(np.quantile(draws, .025)),
         "q50": float(np.quantile(draws, .5)),
         "q97.5": float(np.quantile(draws, .975))}
        for name, draws in entries
    ])

def image_centres(outcome: str, tf: dict) -> dict[str, float]:
    c = tf["centres"]
    if outcome == "yield":
        return {"pH": c["ph"], "initial SOC (g/kg)": c["soc"],
                "initial TN (g/kg)": c["tn"],
                "rate_log": tf["interaction_dose_centre"]}
    if outcome == "n2o":
        return {"pH": c["ph"], "Clay (%)": c["clay"],
                "Seasonal N rate (kg/ha)": c["nrate"]}
    return {}

def image_coefficients(outcome: str, standard_beta: np.ndarray, tf: dict) -> np.ndarray:
    b = np.asarray(standard_beta, float)
    if b.ndim == 1:
        b = b[None, :]
    if b.ndim != 2 or b.shape[1] != len(TERMS[outcome]):
        raise ValueError("Posterior coefficient shape is incompatible with selected model")
    c, s = tf["centres"], tf["scales"]
    if outcome == "yield":
        shift = tf["interaction_dose_centre"] - c["dose_log"]
        return np.column_stack([
            b[:, 0] + b[:, 3] * shift / s["dose_log"], b[:, 1],
            b[:, 2] / s["ph"], b[:, 3] / s["dose_log"],
            b[:, 4] / (s["tn"] * s["dose_log"]),
            b[:, 5] / s["soc"], b[:, 6] / s["soc"],
        ])
    if outcome == "n2o":
        # The zero upland entry is only a simulator-compatibility placeholder.
        return np.column_stack([
            b[:, 0], np.zeros(len(b)), b[:, 1] / s["ph"],
            b[:, 2] / s["clay"], b[:, 3] / s["nrate"],
        ])
    return np.column_stack([
        b[:, 0] - b[:, 2] * c["dose_log"] / s["dose_log"],
        b[:, 1], b[:, 2] / s["dose_log"],
    ])

def image_design(d: pd.DataFrame, outcome: str, tf: dict) -> np.ndarray:
    value = predictors(d, outcome)
    c = image_centres(outcome, tf)
    one = np.ones(len(d))
    if outcome == "yield":
        land = d["Land_use"].astype("string").str.lower().str.strip()
        land = land.replace({"vegetable": "upland"})
        upland = land.eq("upland").to_numpy(dtype=float)
        paddy = land.eq("paddy").to_numpy(dtype=float)
        dose = value["dose_log"] - c["rate_log"]
        soc = value["soc"] - c["initial SOC (g/kg)"]
        return np.column_stack([
            one, upland, value["ph"] - c["pH"], dose,
            (value["tn"] - c["initial TN (g/kg)"]) * dose,
            soc * paddy, soc * upland,
        ])
    if outcome == "n2o":
        return np.column_stack([
            one, np.zeros(len(d)), value["ph"] - c["pH"],
            value["clay"] - c["Clay (%)"],
            value["nrate"] - c["Seasonal N rate (kg/ha)"],
        ])
    return np.column_stack([
        one, (value["ph"] < 6.5).astype(float), value["dose_log"],
    ])


def export_image_coefficients(outcome: str, d: pd.DataFrame, tf: dict,
                              X: np.ndarray, standard_draws: np.ndarray,
                              folder: Path) -> float:
    """Save production IMAGE CSVs after checking predictions match fitted space."""
    betas = image_coefficients(outcome, standard_draws, tf)
    physical_X = image_design(d, outcome, tf)
    # Subsample draws for the algebraic identity, not calibration uncertainty.
    indices = np.linspace(0, len(betas) - 1, min(64, len(betas)), dtype=int)
    max_error = float(np.max(np.abs(X @ standard_draws[indices].T
                                    - physical_X @ betas[indices].T)))
    if not np.isfinite(max_error) or max_error >= 1e-9:
        raise AssertionError(f"{outcome} export changes fitted predictions ({max_error})")
    write_csv(folder / f"{outcome}_beta.csv", pd.DataFrame({
        "predictor": IMAGE_TERMS[outcome], "beta": betas.mean(axis=0),
    }))
    if outcome in ("yield", "n2o"):
        centres = image_centres(outcome, tf)
        write_csv(folder / f"{outcome}_centering_means.csv",
                  pd.DataFrame({"variable": list(centres), "mean": list(centres.values())}))
    if outcome == "yield":
        export_yield_sensitivity(betas, physical_X, folder)
    return max_error

def export_yield_sensitivity(beta: np.ndarray, X: np.ndarray, folder: Path) -> None:
    scores = np.empty(len(beta))
    for start in range(0, len(beta), 128):
        stop = min(start + 128, len(beta))
        with np.errstate(over="raise", invalid="raise"):
            scores[start:stop] = (100 * np.expm1(X @ beta[start:stop].T)).mean(axis=0)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite posterior yield responses")
    reference = float((100 * np.expm1(X @ beta.mean(axis=0))).mean())
    selections = [{"scenario": "mean", "posterior_quantile": np.nan,
                   "posterior_draw_index": -1,
                   "calibration_mean_yield_effect_percent": reference,
                   "quantile_target_percent": np.nan}]
    for label, quantile in (("low", .10), ("high", .90)):
        target = float(np.quantile(scores, quantile))
        i = int(np.argmin(np.abs(scores - target)))
        write_csv(folder / f"yield_beta_{label}.csv", pd.DataFrame({
            "predictor": IMAGE_TERMS["yield"], "beta": beta[i],
        }))
        selections.append({
            "scenario": label, "posterior_quantile": quantile,
            "posterior_draw_index": i,
            "calibration_mean_yield_effect_percent": float(scores[i]),
            "quantile_target_percent": target,
        })
    write_csv(folder / "yield_sensitivity_selection.csv", pd.DataFrame(selections))

def fit_and_export(workbook: pd.DataFrame, data_path: Path, folder: Path,
                   outcome: str, args: argparse.Namespace) -> dict:
    d = prepare_data(workbook, outcome)
    tf = build_transform(d, outcome)
    X, names = build_design(d, outcome, tf)
    rank = int(np.linalg.matrix_rank(X))
    if rank != X.shape[1]:
        raise ValueError(f"Rank-deficient fixed-effect matrix: {outcome}")
    report = {"outcome": outcome, "observations": len(d),
              "studies": int(d["study_id"].nunique())}
    audit = {
        **report, "predictors": names, "matrix_rank": rank, "priors": PRIORS[outcome],
        "transform": tf, "paddy_observations": int(d["Land_use"].eq("paddy").sum()),
        "upland_observations": int(d["Land_use"].eq("upland").sum()),
        "vegetable_observations_merged": int(d["land_use_in_source"].eq("vegetable").sum()),
    }
    if args.dry_run:
        write_json(folder / f"{outcome}_design_audit.json", audit)
        return report

    idata = fit_bayesian_model(d, outcome, X, draws=args.draws, tune=args.tune,
                               chains=args.chains, target_accept=args.target_accept,
                               seed=args.seed)
    diagnostics = sampling_diagnostics(idata)
    report.update(diagnostics)
    fit_dir = folder / "selected_fits" / outcome
    fit_dir.mkdir(parents=True, exist_ok=True)

    idata.to_netcdf(fit_dir / "inference_data.nc")
    write_csv(fit_dir / "posterior_summary.csv", posterior_summary(idata, names))
    export_error = None
    if not args.no_image_exports:
        export_error = export_image_coefficients(
            outcome, d, tf, X, posterior_beta(idata), folder)
    diagnostics["max_image_export_lnrr_error"] = export_error
    write_json(fit_dir / "diagnostics.json", diagnostics)
    write_json(fit_dir / "model_manifest.json", {
        "outcome": outcome, "specification": names,
        "source_workbook_sha256": sha256(data_path),
        "observations": len(d), "studies": int(d["study_id"].nunique()),
        "sample_audit": audit, "image_exports_written": not args.no_image_exports,
        "mcmc": {"draws": args.draws, "tune": args.tune, "chains": args.chains,
                 "target_accept": args.target_accept, "seed": args.seed},
        "model_assumptions": {
            "vegetable": "merged into upland for yield/N2O",
            "n2o": "upland only; pH, clay, seasonal N; no protocol/duration terms",
            "ch4": "paddy only; all eligible experiment durations",
            "yield": "lnRR_Yield < 2; TN-by-dose uses sample-mean log-dose centre",
        },
    })
    return report

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path,
                        help="Yang et al. (2025) experimental workbook")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Separate calibration output directory (not a live IMAGE data folder)")
    parser.add_argument("--outcomes", nargs="+", choices=tuple(OUTCOMES),
                        default=list(OUTCOMES))
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Audit datasets and designs without Bayesian sampling")
    parser.add_argument("--no-image-exports", action="store_true",
                        help="Fit and archive posterior without generating IMAGE CSVs")
    args = parser.parse_args(argv)
    if not args.data.is_file():
        parser.error(f"Workbook does not exist: {args.data}")
    if min(args.draws, args.tune, args.chains) < 1:
        parser.error("Draws, tuning iterations and chains must be positive")
    if not 0 < args.target_accept < 1:
        parser.error("target-accept must be between zero and one")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workbook = pd.read_excel(args.data) 
    reports = []
    for outcome in args.outcomes:
        result = fit_and_export(workbook, args.data, args.output_dir, outcome, args)
        reports.append(result)
        print(f"{outcome}: {result['observations']} observations, "
              f"{result['studies']} studies"
              + (f", max R-hat {result['max_rhat']:.4f}, "
                 f"healthy={result['healthy']}" if not args.dry_run else " [design audit]"))
    write_csv(args.output_dir / "fit_summary.csv", pd.DataFrame(reports))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
