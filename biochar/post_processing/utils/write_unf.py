import numpy as np
#%% ================================================================================================
def WriteUNF(data,variable_name,datatype = 'float32',year = '',path = ''):
    """
    Writes to UNF 
    """
    # -- Determine the datatype --------------------------------------------------------------------
    if datatype == 'float32':
        extention = 'UNF0'
    elif datatype == 'int8':
        extention = 'UNF1'
    elif datatype == 'int16':
        extention = 'UNF2'
    elif datatype == 'int32':
        extention = 'UNF4'
    else:
        raise Exception('Datatype "'+datatype+'" not found')
   
    data = data.astype(datatype)

    # -- Test shape --------------------------------------------------------------------------------
    """TODO:
    data.shape -> first dim must be a grid-size shape
    """
    # -- Determine the filename --------------------------------------------------------------------
    if data.ndim == 1:
        filename = path+variable_name+str(year)+'.'+extention
    elif data.ndim == 2:
        filename = path+variable_name+str(year)+'.'+str(data.shape[1])+'.'+extention
    else:
        raise NotImplementedError('UNF files cannot be saved for data with more than 2 dimensions')
        
    # --- write UNF file ---------------------------------------------------------------------------
    with open(filename, "wb") as f:
        np.array(data).tofile(f)

    print('Saved as: '+filename)
