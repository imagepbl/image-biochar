import pym
import numpy as np
import xarray as xr
from pathlib import Path


data_dir = Path(
    r"X:\user\orhalem\image_biochar"
    r"\timer\data\scen\biochar\SSP2_M"
)

input_file = data_dir / "TimerResiduesPotCurve.nc"
output_file = data_dir / "PotResidues.SCN"

input_variable = "TimerResiduesPotCurve"
output_variable = "PotResidues"

def ensure_world_row(data):
    n_regions = data.sizes["region"]

    if n_regions == 26:
        region = np.asarray(data["region"].values)
        world_coord = np.max(region) + 1 if np.issubdtype(region.dtype, np.number) else "World"
        world = data.sum("region").expand_dims(region=[world_coord])
        return xr.concat([data, world], dim="region")

    if n_regions != 27:
        raise ValueError(f"Expected 26 or 27 regions, found {n_regions}.")

    world = data.isel(region=slice(0, 26)).sum("region")
    if not np.allclose(data.isel(region=26).values, world.values, rtol=1.0e-5, atol=1.0e5):
        raise ValueError("The 27th region is not the sum of regions 1..26.")

    return data


def main():
    with xr.open_dataset(input_file) as ds:
        data = ds[input_variable].transpose("time", "region", "bin").load()

    if data.sizes["bin"] != 10:
        raise ValueError(f"Expected 10 TIMER residue bins, found {data.sizes['bin']}.")

    if not np.all(np.isfinite(data.values)):
        raise ValueError(f"{input_variable} contains non-finite values.")

    if np.min(data.values) < -1.0:
        raise ValueError(f"{input_variable} contains materially negative values.")

    data = ensure_world_row(data.clip(min=0.0).astype(np.float32))
    years = np.rint(data["time"].values).astype(int)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    pym.write_mym(
        data=data.values,
        time=years,
        filename=str(output_file),
        variable_name=output_variable,
        table_format="wide",
        comment="! real PotResidues[NRC,NBC](t); Unit=GJ/yr; direct conversion of TimerResiduesPotCurve.nc",
        sep=",",
    )

    lines = output_file.read_text(encoding="utf-8").splitlines(keepends=True)
    output_file.write_text(
        "".join(line for i, line in enumerate(lines) if i == 0 or not line.startswith("!")),
        encoding="utf-8",
    )

    total_ej = data.isel(region=slice(0, 26)).sum(("region", "bin")).values / 1.0e9

    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Shape written: {data.shape} [time, region, bin]")
    print(f"Years: {years[0]} - {years[-1]}")
    print(f"Global physical residue supply: {np.min(total_ej):.3f} - {np.max(total_ej):.3f} EJ/yr")
    print("No biochar multiplier or other supply adjustment was applied.")


if __name__ == "__main__":
    main()
