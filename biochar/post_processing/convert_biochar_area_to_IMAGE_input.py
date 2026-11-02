"""
Script to convert the biochar area (separated by technology) to a file that can be read by IMAGE
"""
import pym
import numpy as np
import xarray as xr
from pathlib import Path

scenario_name = "biochar_c400_LT"

scratch_dir = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output\biochar_c400_LT"
)

output_root = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output\biochar2image"
)

output_dir = output_root / scenario_name
output_dir.mkdir(parents=True, exist_ok=True)

file_input = "BiocharAreaTech.nc"
file_out = "BIOCHAR_AREA.SCN"

input_path = scratch_dir / file_input
output_path = output_dir / file_out

step = 5

data = xr.open_dataset(input_path)

first_year = int(data["time"].min().item())

if first_year > 1970:
    missing_years = np.arange(
        1970,
        first_year,
        step,
        dtype=int
    )

    data_zeros = data.isel(time=[0]).reindex(
        time=missing_years,
        fill_value=0.0
    )

    data = xr.concat(
        [data_zeros, data],
        dim="time"
    )

data = data.transpose("time", "tech", "region")
biochar_area = data["BiocharAreaTech"]

pym.write_mym(
    data=biochar_area.values,
    time=data["time"].values,
    filename=str(output_path),
    variable_name="BIOCHAR_AREA",
    table_format="wide",
    comment=(
        "! real biochar_area[NBIOCHAR_TECH,NR] (t); "
        "Unit= ha; Label= Area for biochar application"
    ),
    sep=","
)

with open(output_path, "r", encoding="utf-8") as f:
    raw_text = f.readlines()

with open(output_path, "w", encoding="utf-8") as f:

    for i, line in enumerate(raw_text):

        if i == 0 or not line.startswith("!"):
            f.write(line)

print(f"IMAGE biochar input written to:\n{output_path}")