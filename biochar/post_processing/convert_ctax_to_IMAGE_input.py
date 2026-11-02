"""
Convert biochar carbon tax scenario input file to an IMAGE-land input file
"""
import pym
import prism
import numpy as np
from pathlib import Path

scenario_name = "biochar_c400_LT"
ctax_scen = "c400"

project_root = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism"
)

input_path = project_root / "timer" / "data" / "scen" / "biochar" / ctax_scen / "ctax.out"

output_root = (
    project_root
    / "cdr_output"
    / "biochar2image"
)

output_dir = output_root / scenario_name
output_dir.mkdir(parents=True, exist_ok=True)

file_out = "CARBON_TAX.SCN"
output_path = output_dir / file_out

_IMAGE_REGIONS = [
    "CAN",
    "USA",
    "MEX",
    "RCAM",
    "BRA",
    "RSAM",
    "NAF",
    "WAF",
    "EAF",
    "SAF",
    "WEU",
    "CEU",
    "TUR",
    "UKR",
    "STAN",
    "RUS",
    "ME",
    "INDIA",
    "KOR",
    "CHN",
    "SEAS",
    "INDO",
    "JAP",
    "OCE",
    "RSAS",
    "RSAF",
]

RegionTotalExp = prism.Dimension(
    "region",
    _IMAGE_REGIONS + ["World", "World2"]
)

timeline = prism.Timeline(
    start=prism.Q_(1969, "year"),
    end=prism.Q_(2100, "year"),
    stepsize=prism.Q_(1, "year")
)

data = prism.TimeVariable[RegionTotalExp](
    file=str(input_path),
    timeline=timeline
)

def to_numpy(values):
    if hasattr(values, "magnitude"):
        values = values.magnitude
    if hasattr(values, "values"):
        values = values.values
    return np.asarray(values)

carbon_tax = to_numpy(data._values)
time = to_numpy(data.timeline.to_array())

pym.write_mym(
    data=carbon_tax,
    time=time,
    filename=str(output_path),
    variable_name="CARBON_TAX",
    table_format="wide",
    comment=(
        "! real CARBON_TAX[NRGT] (t); "
        "Unit= ; Label=''"
    ),
    sep=","
)

with open(output_path, "r", encoding="utf-8") as f:
    raw_text = f.readlines()

with open(output_path, "w", encoding="utf-8") as f:
    for i, line in enumerate(raw_text):
        if i == 0 or not line.startswith("!"):
            f.write(line)


print(
    f"IMAGE carbon-tax input written to:\n"
    f"{output_path}"
)