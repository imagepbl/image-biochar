"""
Script that turns vectorised netcdf files into UNF files
"""
import xarray as xr
import numpy as np
from utils.write_unf import WriteUNF

# Paths
scratch = "X:/user/orhalem/image_biochar/cdr_output/biochar2image/static_data"
outdir = "X:/user/scherrenbm/ModelDevelopment/IMAGE_Development/IMAGE_Biochar/Biochar_Data/unf_5min/"

# Files and how to rename them in the UNF-file
biochar_file_conversion = {
    #"n2o_effects_paddy.nc" : "GN2O_BIOCHAR_EMISSION_EFFECT_RICE",
    "ch4_effects_paddy.nc" : "GCH4_BIOCHAR_EMISSION_EFFECT_RICE",
    "n2o_effects_paddy.nc" : "GN2O_BIOCHAR_EMISSION_EFFECT_UPLAND",
    "yield_effects_paddy.nc" : "GBIOCHAR_YIELD_EFFECT_RICE",
    "yield_effects_upland.nc" : "GBIOCHAR_YIELD_EFFECT_UPLAND",
}

# Type of output data (generally going to be float (never use double precision!))
astype = "float32"

# We need to replace NaN with a value.
replace_nan_value = 0.0

# ==================================================================================================

# Load the data
for input_filename in biochar_file_conversion:

    # Get paths
    fullpath_input = scratch + input_filename
    filename_output = biochar_file_conversion[input_filename] # No UNF* or dimension at the end!

    # Load the data
    data = xr.open_dataarray(fullpath_input)

    # If 3 dimensions, only select the yield year
    if data.ndim == 3:
        print("Warning: only select first yield year of "+str(input_filename))
        data = data.sel({"yield_year":1})

    # Replace nan with 0
    data = data.fillna(replace_nan_value)
    
    # Convert to numpy
    data = data.to_numpy()

    # Write to UNF
    WriteUNF(data,filename_output,datatype =astype,path = outdir)
