import yaml
from copy import deepcopy
from pathlib import Path

SSP_TO_LAND_KEY = {"SSP1": "ssp1m", "SSP2": "ssp2m", "SSP3": "ssp3h"}
LAND_KEY_TO_SSP = {value: key for key, value in SSP_TO_LAND_KEY.items()}
LAND_DIRECTORY_FIELDS = (
    "biochar_land_directory",
    "landrent_directory",
    "residue_directory",
)

def deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

def load_biochar_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def resolve_ssp(run, run_config):
    raw_run_settings = run.get("settings", {})
    if "land_scenario" in raw_run_settings:
        requested = raw_run_settings["land_scenario"]
        if not isinstance(requested, str) or requested not in SSP_TO_LAND_KEY:
            raise ValueError(
                f"Invalid land_scenario={requested!r} in run {run['name']!r}. Choose exactly SSP1, SSP2, or SSP3"
            )
        ssp = requested
        return ssp, SSP_TO_LAND_KEY[ssp]

    if "land" in run:
        land_name = run["land"]
        return LAND_KEY_TO_SSP[land_name], land_name

    requested = run_config.get("settings", {}).get("land_scenario", "SSP2")
    if not isinstance(requested, str) or requested not in SSP_TO_LAND_KEY:
        raise ValueError(
            f"Invalid default land_scenario={requested!r}. Choose exactly SSP1, SSP2, or SSP3"
        )
    return requested, SSP_TO_LAND_KEY[requested]

def _land_paths(land_scenarios, land_key, ssp, run_name):
    land = land_scenarios[land_key]
    paths = {}
    for field in LAND_DIRECTORY_FIELDS:
        value = land.get(field)
        paths[field] = Path(value.strip())
    return paths

def resolve_runs(config):
    defaults = config.get("defaults", {})
    carbon_scenarios = config["carbon_tax_scenarios"]
    land_scenarios = config["land_scenarios"]
    resolved = []
    names = set()

    for run in config["runs"]:
        names.add(run["name"])

        run_config = deep_update(defaults, run)
        carbon_name = run["carbon_tax"]
        if carbon_name not in carbon_scenarios:
            raise KeyError(f"Unknown carbon tax scenario: {carbon_name}")

        ssp, land_name = resolve_ssp(run, run_config)
        land_paths = _land_paths(land_scenarios, land_name, ssp, run["name"])

        run_config["settings"]["land_scenario"] = ssp
        run_config["land"] = land_name
        run_config["land_name"] = land_name
        run_config.update(land_paths)

        carbon = carbon_scenarios[carbon_name]
        run_config["carbon_tax_name"] = carbon_name
        run_config["carbon_tax_directory"] = Path(carbon["directory"])

        reference_directory = carbon.get("residue_rationing_reference_directory")
        if reference_directory is not None:
            run_config["residue_rationing_reference_directory"] = Path(reference_directory)

        output_root = Path(run_config["output_root"])
        run_config["output_directory"] = Path(
            run_config.get("output_directory", output_root / run_config["name"])
        )
        resolved.append(run_config)

    return resolved

def _make_yaml_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _make_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_yaml_safe(item) for item in value]
    return value

def write_resolved_run_config(run_config, timeline_config, output_filename="scenario.yaml"):
    output_directory = Path(run_config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    scenario_config = _make_yaml_safe({
        "timeline": deepcopy(timeline_config),
        "scenario": deepcopy(run_config),
    })
    output_path = output_directory / output_filename
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(scenario_config, file, sort_keys=False, default_flow_style=False)
    return output_path