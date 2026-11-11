# IMAGE Biochar

A research module for modelling biochar deployment and associated climate mitigation and agronomic benefits within [IMAGE (Integrated Model to Assess the Global Environment)](https://www.pbl.nl/en/image/home).

The model combines biomass residue supply, pyrolysis production technologies, soil carbon persistence, agricultural effects, and deployment costs. The main workflow operates at 5 arcminute resolution across 26 IMAGE regions, covering 2025 to 2100 in 5-year timesteps.

## Requirements

- A compatible IMAGE/TIMER Python environment, including the matching `prism` and `pym` libraries. This repository is used as the `timer.cdr` package within that environment.
- NumPy, pandas, SciPy, xarray, netCDF4, Dask, Matplotlib, Pint, PyYAML, and openpyxl.
- External IMAGE/TIMER scenario data, residue supply/cost curves, spatial inputs, and biochar parameters.

Input datasets and a dependency lockfile are not included.

## Setup

From the parent TIMER project root, clone the module into an unused `timer/cdr` directory:

```bash
git clone https://github.com/imagepbl/image-biochar.git timer/cdr
```

If `timer/cdr` already exists, integrate the source into your working project. Create a local configuration:

```bash
cp timer/cdr/config/biochar_scenarios.yaml timer/cdr/config/biochar_local.yaml
```

Update the following before running:

| File | Required configuration |
| --- | --- |
| `config/biochar_local.yaml` | Scenario input directories, output location, and entries under `runs`. |
| [cdr_util.py](cdr_util.py) | Shared spatial, land-use, biomass, residue, and biochar data paths. |

## Configuration and usage

The [example configuration](config/biochar_scenarios.yaml) defines the timeline, carbon price and land scenarios, default settings, and individual runs. Scenario-specific settings override the defaults. Available options include technology selection, carbon persistence, crop yield response, application rates, and residue rationing; see [biochar/config.py](biochar/config.py).

The supplied timeline is **2025–2100 in five-year steps**. Input time axes must match the model's five-year indexing assumptions.

Every entry in `runs` executes sequentially. Outputs go to `<output_root>/<run_name>/`.

## Outputs

The runner saves a resolved `scenario.yaml` before simulation and writes the main NetCDF outputs at the final timestep.

Analysis and figure scripts are available in [results/biochar/](results/biochar/).

## License

[MIT License](LICENSE). Copyright © 2026 IMAGE.
