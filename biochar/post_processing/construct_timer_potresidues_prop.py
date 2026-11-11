import pym
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Any, Mapping

def _load(path, variable):
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            variables = list(ds.data_vars)
            variable = variables[0]

        return ds[variable].transpose("time", "region", "bin").load().astype(np.float64)

def _align_multiplier(multiplier, resource, require_complete=True):

    multiplier = multiplier.reindex(
        region=resource["region"],
        bin=resource["bin"],
        fill_value=1.0,
    )

    resource_years = np.rint(resource["time"].values.astype(float)).astype(int)
    multiplier_years = np.rint(multiplier["time"].values.astype(float)).astype(int)

    year_index = {year: i for i, year in enumerate(multiplier_years)}
    values = np.ones(resource.shape, dtype=np.float64)
    missing = []

    for i, year in enumerate(resource_years):
        if year in year_index:
            values[i] = multiplier.values[year_index[year]]
        elif require_complete and multiplier_years.min() <= year <= multiplier_years.max():
            missing.append(int(year))

    return xr.DataArray(values, dims=resource.dims, coords=resource.coords)

def _append_world(data):
    n_regions = data.sizes["region"]

    if n_regions == 27:
        data = data.copy()
        data.values[:, 26, :] = data.isel(region=slice(0, 26)).sum("region").values
        return data

    if n_regions != 26:
        raise ValueError(f"Expected 26 or 27 regions, found {n_regions}.")

    region = np.asarray(data["region"].values)
    world_coord = np.max(region) + 1 if np.issubdtype(region.dtype, np.number) else "World"
    world = data.sum("region").expand_dims(region=[world_coord])

    return xr.concat([data, world], dim="region")

def _strip_comments(path):
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text(
        "".join(line for i, line in enumerate(lines) if i == 0 or not line.startswith("!")),
        encoding="utf-8",
    )

def build_timer_residual_potresidues_scn(
    *,
    scenario_name,
    multiplier_path,
    timer_total_path,
    exo_residues_path,
    output_dir,
    output_filename=None,
    total_variable="TimerResiduesPotCurve",
    exo_variable="ExoResiduesPotCurve",
    multiplier_variable="TimerResidualResidueMultiplier",
    mym_variable_name="PotResidues",
    include_global_region=True,
    strip_pym_comments=True,
    require_complete_inside_multiplier_period=True,
    write_audit_netcdf=True,
):
    scenario_name = str(scenario_name).strip()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (output_filename or "PotResidues.SCN")

    total = _load(timer_total_path, total_variable)
    exo = _load(exo_residues_path, exo_variable)
    multiplier = _load(multiplier_path, multiplier_variable)

    total, exo = xr.align(total, exo, join="exact")
    total = total.clip(min=0.0)
    exo = exo.clip(min=0.0)

    tolerance = np.maximum(1.0, 1.0e-6 * np.maximum(total.values, 1.0))
    exo = xr.where(exo > total, total, exo)

    multiplier = _align_multiplier(
        multiplier,
        total,
        require_complete=require_complete_inside_multiplier_period,
    )
    minimum = float(multiplier.min())
    maximum = float(multiplier.max())

    multiplier = multiplier.clip(0.0, 1.0)
    shared_original = (total - exo).clip(min=0.0)
    shared_rationed = multiplier * shared_original
    coupled = exo + shared_rationed

    if np.any(coupled.values > total.values + tolerance):
        raise ValueError("Coupled PotResidues exceeds the original TIMER resource")
    if np.any(coupled.values < exo.values - tolerance):
        raise ValueError("Coupled PotResidues falls below the exogenous TIMER-only resource")

    data = _append_world(coupled) if include_global_region else coupled
    data = data.transpose("time", "region", "bin").astype(np.float32)
    years = np.rint(data["time"].values.astype(float)).astype(int)

    pym.write_mym(
        data=data.values,
        time=years,
        filename=str(output_path),
        variable_name=mym_variable_name,
        table_format="wide",
        comment=(
            f"! real PotResidues[NRC,NBC](t); Unit=GJ/yr; Scenario={scenario_name}; "
            "TIMER residue potential after biochar competition"
        ),
        sep=",",
    )

    if strip_pym_comments:
        _strip_comments(output_path)

    audit_path = None

    if write_audit_netcdf:
        audit_path = output_dir / f"POTRESIDUES_{scenario_name.upper()}_AUDIT.nc"
        audit = xr.Dataset(
            {
                "PotResiduesOriginal": total.astype(np.float32),
                "ExoResiduesPotCurve": exo.astype(np.float32),
                "SharedResiduesOriginal": shared_original.astype(np.float32),
                "TimerResidualResidueMultiplier": multiplier.astype(np.float32),
                "SharedResiduesRationed": shared_rationed.astype(np.float32),
                "PotResiduesCoupled": coupled.astype(np.float32),
            }
        )

        for variable in (
            "PotResiduesOriginal",
            "ExoResiduesPotCurve",
            "SharedResiduesOriginal",
            "SharedResiduesRationed",
            "PotResiduesCoupled",
        ):
            audit[variable].attrs["units"] = "GJ/yr"

        audit["TimerResidualResidueMultiplier"].attrs["units"] = "-"
        audit.attrs["scenario"] = scenario_name
        audit.attrs["coupling_equation"] = (
            "PotResiduesCoupled = ExoResiduesPotCurve + "
            "TimerResidualResidueMultiplier * "
            "(PotResiduesOriginal - ExoResiduesPotCurve)"
        )
        audit.to_netcdf(audit_path)

    total_ej = total.sum(("region", "bin")).values / 1.0e9
    exo_ej = exo.sum(("region", "bin")).values / 1.0e9
    shared_ej = shared_original.sum(("region", "bin")).values / 1.0e9
    rationed_ej = shared_rationed.sum(("region", "bin")).values / 1.0e9
    coupled_ej = coupled.sum(("region", "bin")).values / 1.0e9

    print(f"TIMER residual-resource scenario: {scenario_name}")
    for year, original, exogenous, shared, rationed, new in zip(
        years, total_ej, exo_ej, shared_ej, rationed_ej, coupled_ej
    ):
        print(
            f"{year}: total={original:.3f}, exogenous={exogenous:.3f}, "
            f"shared={shared:.3f}, rationed={rationed:.3f}, new={new:.3f} EJ/yr"
        )
    print(f"Multiplier min/max: {minimum:.4f} / {maximum:.4f}")
    print(f"SCN written to: {output_path}")
    if audit_path is not None:
        print(f"Audit NetCDF written to: {audit_path}")

    return {
        "scenario_name": scenario_name,
        "scn_path": output_path,
        "audit_path": audit_path,
        "potresidues_original": total,
        "exo_residues": exo,
        "shared_residues_original": shared_original,
        "timer_resource_multiplier": multiplier,
        "shared_residues_rationed": shared_rationed,
        "potresidues_coupled": coupled,
        "potresidues_for_mym": data,
    }

def build_multiple_timer_residual_scenarios(
    *,
    scenarios: Mapping[str, Mapping[str, Any]],
    default_timer_total_path=None,
    default_exo_residues_path=None,
    default_output_root,
    **common_kwargs,
):
    results = {}
    for scenario_name, config in scenarios.items():
        config = dict(config)
        multiplier_path = config.pop("multiplier_path")
        timer_total_path = config.pop("timer_total_path", default_timer_total_path)
        exo_residues_path = config.pop("exo_residues_path", default_exo_residues_path)
        output_dir = config.pop("output_dir", Path(default_output_root) / scenario_name)

        if timer_total_path is None or exo_residues_path is None:
            raise ValueError(f"{scenario_name}: residue input paths were not supplied.")

        results[scenario_name] = build_timer_residual_potresidues_scn(
            scenario_name=scenario_name,
            multiplier_path=multiplier_path,
            timer_total_path=timer_total_path,
            exo_residues_path=exo_residues_path,
            output_dir=output_dir,
            **config,
            **common_kwargs,
        )
    return results

build_proportional_timer_potresidues_scn = build_timer_residual_potresidues_scn
build_multiple_proportional_timer_scenarios = build_multiple_timer_residual_scenarios

if __name__ == "__main__":
    residue_dir = Path(
        r"X:\user\orhalem\image_biochar"
        r"\timer\data\scen\biochar\SSP2_M_main"
    )
    biochar_dir = Path(
        r"X:\user\orhalem\image_biochar"
        r"\cdr_output\biochar_c400_prop"
    )
    output_root = Path(
        r"X:\user\orhalem\image_biochar"
        r"\cdr_output\biochar2timer"
    )

    build_timer_residual_potresidues_scn(
        scenario_name="biochar_c400_prop",
        multiplier_path=biochar_dir / "TimerResidualResidueMultiplier.nc",
        timer_total_path=residue_dir / "TimerResiduesPotCurve.nc",
        exo_residues_path=residue_dir / "ExoResiduesPotCurve.nc",
        output_dir=output_root / "biochar_c400_prop",
    )