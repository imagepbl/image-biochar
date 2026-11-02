import numpy as np

def ReadUNF(path,filename):
    """
    Loads UNF data, reshapes it to its dimensions and extracts the year
    """

    # Check the filename
    if 'unf' not in filename.lower():
        print('filename: "'+filename+'" is not an UNF filename')
    
    filename_chunk = filename.split('.')

    # -- Determine the datatype --------------------------------------------------------------------
    extention = filename_chunk[-1]

    if extention.lower() == 'unf0':
        data_type = 'float32'
    elif extention.lower() == 'unf1':
        data_type = 'int8'
    elif extention.lower() == 'unf2':
        data_type = 'int16'
    elif extention.lower() == 'unf4':
        data_type = 'int32'
    else:
        raise NotImplementedError('UNF-type "'+extention+'" not found. File cannot be processed')
    
    if len(filename_chunk) == 1:
        raise ValueError(
            'filename "'
            +filename+
            '" cannot be used to extract a variable name by pplitting based on "."'
        )

    # -- Extract the dimensions and year -----------------------------------------------------------
    dim,year = FindYearDimensionInUNF(filename)
        
    # --- Load UNF files ---------------------------------------------------------------------------
    with open(path+filename,'rb') as f:
        data = np.fromfile(f, data_type)

    # -- Reshape the data using the dims -----------------------------------------------------------

    # Determine the total dimension size without the gridded-indexes
    total_dimension_size = 1
    for d in dim:
        total_dimension_size *= d

    # Dimension sizes MUST to be an integer multiple of the data size
    if data.size / total_dimension_size != int(data.size / total_dimension_size):
        raise ValueError(
            ' "'+filename+'" size is "'+str(data.size)+
            '" which is not a multiple of dimension sizes: "'+total_dimension_size+'"'
        )

    # Determine the grid size (which is the remainder of the data / the dimensions)
    grid_size = int(data.size / total_dimension_size)

    # Reshape the data based on the dimension and grid sizes
    data = data.reshape([grid_size,*dim])

    print('Succesfully loaded "'+filename+'" number of datapoints: '+str(data.size))

    return data

def FindYearDimensionInUNF(filename):
    filename_chunk = filename.split('.')

    # -- Extract the dimensions and year -----------------------------------------------------------
    dim = []; year = []

    # Extract the year or dimension
    if len(filename_chunk) == 2:
        # No dimension present (only variable name and UNF)
        year = []
    else:
        # Dimension / year present
        for chunk in filename_chunk[1:-1]: # Skips variable name and extention
            
            # Convert dimension string to number
            try:
                dim_number = int(chunk)
            except:
                print(' > Dimension "'+chunk+'" of "'+filename+'" not a valid dimension size')
                continue

            # Check if dimension is a year
            if dim_number > 1850 and dim_number <= 2100:
                year.append(dim_number)

            elif dim_number > 2100: 
                # FAILSAVE
                raise Exception('Dimension "'+year(dim_number)+'" in filename "'+filename+'" may be too big!')

            else:
                # Not a year, has to be a dimension size
                dim.append(dim_number)
    
    return dim,year