import prism
import numpy as np
import xarray as xr
from pathlib import Path
from timer.globals.constants import EPS
from timer.cdr.cdr_util import (
    time_step_scen,
    Region, Region26,
    RCC,
    RCCPlus,
    PyrolysisTechnology,
    BiocharFeedstock,
)

def _require_directory(model, attribute):
    directory = getattr(model, attribute, None)
    if directory is None:
        raise ValueError(
            f"Biochar setting '{attribute}' is required when proportional residue rationing is enabled"
        )
    return Path(directory)

def _resolve_netcdf_path(directory, variable):
    directory = Path(directory)
    for suffix in (".NC", ".nc"):
        path = directory / f"{variable}{suffix}"

        if path.exists():
            return path
    return directory / f"{variable}.NC"

def _read_netcdf(directory, variable, time):
    path = _resolve_netcdf_path(
        directory=directory,
        variable=variable
    )

    with xr.open_dataset(
        path,
        engine="netcdf4",
        chunks={"time": 1},
    ) as ds:
        return np.asarray(ds[variable].isel(time=time_step_scen(time)), dtype=np.float32)

def read_timer_residue_accounting(
        model,
        variable,
        time,
):
    residue_directory = _require_directory(model,"residue_directory")

    data = _read_netcdf(
        directory=residue_directory,
        variable=variable,
        time=time,
    )

    bins = (
        len(RCCPlus.coords)
        if data.size % len(RCCPlus.coords) == 0
        else len(RCC.coords)
    )

    return data.reshape(-1, bins)[:len(Region26.coords), :len(RCC.coords)]

def read_reference_biochar_residue_use(
        model,
        time,
):

    reference_directory = _require_directory(
        model,
        "residue_rationing_reference_directory",
    )

    data = _read_netcdf(
        directory=reference_directory,
        variable="BiocharResidueDMUsed",
        time=time,
    )

    return data.reshape(
        len(Region26.coords),
        len(PyrolysisTechnology.coords),
        len(BiocharFeedstock.coords),
        len(RCC.coords),
    )

def get_timer_shared_residue_utilisation(
        model,
        time,
):
    carbon_tax_directory = _require_directory(
        model,
        "carbon_tax_directory",
    )

    timeline = getattr(
        model,
        "simulation_timeline",
        None,
    )

    timer_total_utilisation = np.asarray(
        prism.TimeVariable[Region, RCCPlus, np.float32](
            timeline=timeline,
            file=(carbon_tax_directory / "ResiduesProdFraction.out"),
        ).get(time).values,
        dtype=np.float64,
    )[:len(Region26.coords), :len(RCC.coords)]

    timer_total_potential = np.maximum(
        np.asarray(
            read_timer_residue_accounting(
                model=model,
                variable="TimerResiduesPotCurve",
                time=time
            ),
            dtype=np.float64,
        ),
        0.0,
    )

    exo_residue_potential = np.minimum(
        np.maximum(
            np.asarray(
                read_timer_residue_accounting(
                    model=model,
                    variable="ExoResiduesPotCurve",
                    time=time
                ),
                dtype=np.float64,
            ),
            0.0,
        ),
        timer_total_potential,
    )

    shared_residue_potential = timer_total_potential - exo_residue_potential

    timer_total_utilisation = np.clip(
        timer_total_utilisation,
        0.0,
        1.0,
    )

    timer_total_use = timer_total_utilisation * timer_total_potential

    timer_exo_use = np.minimum(
        timer_total_use,
        exo_residue_potential,
    )

    timer_shared_use = np.maximum(
        timer_total_use - timer_exo_use,
        0.0,
    )

    timer_shared_utilisation = np.divide(
        timer_shared_use,
        shared_residue_potential,
        out=np.zeros_like(shared_residue_potential, dtype=np.float64),
        where=shared_residue_potential > EPS,
    )

    timer_shared_utilisation = np.clip(
        timer_shared_utilisation,
        0.0,
        1.0,
    )

    diagnostics = {
        "timer_total_utilisation": timer_total_utilisation,
        "timer_total_potential_gj": timer_total_potential,
        "exo_residue_potential_gj": exo_residue_potential,
        "shared_residue_potential_gj": shared_residue_potential,
        "timer_total_use_gj": timer_total_use,
        "timer_exo_use_gj": timer_exo_use,
        "timer_shared_use_gj": timer_shared_use,
        "timer_shared_utilisation": timer_shared_utilisation,
    }

    diagnostics = {
        key: value.astype(np.float32)
        for key, value in diagnostics.items()
    }

    return (
        timer_shared_utilisation.astype(np.float32),
        diagnostics,
    )

def apply_proportional_residue_rationing(
        residue_supply_dm,
        timer_utilisation,
        reference_biochar_residue_use,
):

    residue_supply_dm = np.asarray(
        residue_supply_dm,
        dtype=np.float32,
    )

    timer_utilisation = np.clip(
        np.asarray(timer_utilisation, dtype=np.float32),
        0.0,
        1.0,
    )

    reference_biochar_residue_use = np.asarray(
        reference_biochar_residue_use, dtype=np.float32,
    )

    biochar_used_dm = np.sum(
        reference_biochar_residue_use,
        axis=1, dtype=np.float32,
    )

    total_supply = np.sum(
        residue_supply_dm,
        axis=1, dtype=np.float32,
    )

    biochar_used = np.sum(
        biochar_used_dm,
        axis=1, dtype=np.float32,
    )

    biochar_utilisation = np.divide(
        biochar_used,
        total_supply,
        out=np.zeros_like(total_supply, dtype=np.float32),
        where=total_supply > EPS,
    )

    biochar_utilisation = np.clip(
        biochar_utilisation,
        0.0,
        1.0,
    )

    total_claim = timer_utilisation + biochar_utilisation

    proportional_biochar_share = np.divide(
        biochar_utilisation,
        total_claim,
        out=np.zeros_like(total_claim, dtype=np.float32),
        where=total_claim > EPS,
    )

    proportional_timer_share = np.divide(
        timer_utilisation,
        total_claim,
        out=np.zeros_like(total_claim,dtype=np.float32),
        where=total_claim > EPS,
    )

    oversubscribed = total_claim > 1.0

    biochar_share = np.where(
        oversubscribed,
        proportional_biochar_share,
        np.maximum(0.0, 1.0 - timer_utilisation),
    ).astype(np.float32)

    timer_share = np.where(
        oversubscribed,
        proportional_timer_share,
        timer_utilisation,
    ).astype(np.float32)

    rationed_residue_supply_dm = (
        residue_supply_dm
        * biochar_share[:, None, :]
    ).astype(np.float32)

    return (
        rationed_residue_supply_dm,
        biochar_share,
        timer_share,
    )

def calculate_timer_residue_multipliers(
        raw_residue_supply_dm,
        timer_utilisation,
        timer_share,
        actual_biochar_used_dm,
        reference_biochar_residue_use,
):

    supply = np.asarray(raw_residue_supply_dm, dtype=np.float64)
        
    standalone_used = np.nansum(
        np.asarray(reference_biochar_residue_use,dtype=np.float64),
        axis=1
    )

    actual_used = np.asarray(actual_biochar_used_dm,dtype=np.float64)
    
    if actual_used.ndim == 4:
        actual_used = np.nansum(actual_used,axis=1)

    supply = np.nansum(supply,axis=1)
    standalone_used = np.nansum(standalone_used,axis=1)
    actual_used = np.nansum(actual_used,axis=1)

    biochar_utilisation = np.divide(
        standalone_used,
        supply,
        out=np.zeros_like(supply),
        where=supply > EPS,
    )

    actual_biochar_utilisation = np.divide(
        actual_used,
        supply,
        out=np.zeros_like(supply),
        where=supply > EPS,
    )

    timer_multiplier = np.where(
        (np.asarray(timer_utilisation) + np.clip(biochar_utilisation,0.0,1.0)) > 1.0,
        timer_share,
        1.0,
    ).astype(np.float32)

    timer_residual_multiplier = (
        1.0 - np.clip(actual_biochar_utilisation, 0.0, 1.0)
    ).astype(np.float32)

    return (
        timer_multiplier,
        timer_residual_multiplier,
    )