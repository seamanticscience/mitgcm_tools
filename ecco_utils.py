import ecco_v4_py as ecco
import glob       as gb
import numpy      as np
import xarray     as xr
import xmitgcm    as xm
import datetime
import re 

def get_extra_ecco_metadata(version):
    
    if version == "LLC270":
        metadata = xm.utils.get_extra_metadata(domain="llc", nx=270)
        plot_dx = 1/3
        plot_dy = 1/3
    else:
        metadata = xm.utils.get_extra_metadata(domain="llc", nx=90)
        plot_dx = 1
        plot_dy = 1
    
    if version == "LLC270":
        metadata["datadir"] = "/nobackup/dcarrol2/v05_latest/darwin3/run/diags/budget"
        metadata["griddir"] = "/nobackup/dcarrol2/v05_latest/darwin3/run"
    elif version == "ECCOv4r4":
        metadata["datadir"] = "/nobackup/dcarrol2/v05_1deg_V4r4/darwin3/run/diags/budget"
        metadata["griddir"] = "/home3/jmlauder/Jupyter_notebooks/ECCOv4r4_grid"
    elif version == "ECCOv4r5":
        metadata["datadir"] = "/nobackup/dcarrol2/v05_1deg_V4r5/darwin3/run/diags/budget"
        metadata["griddir"] = "/home3/jmlauder/Jupyter_notebooks/ECCOv4r5_grid"
    elif version == "ECCOv4r5_v06":
        metadata["datadir"] = "/nobackup/jmlauder/v06/darwin3/run/diags/budget"
        metadata["griddir"] = "/nobackup/jmlauder/v06/darwin3/run"
    elif version == "ECCOv4r5_JRA55DO":
        metadata["datadir"] = "/nobackup/rsavelli/ECCO_V4r5/runoff_exp/darwin3/run/diags/budget"
        metadata["griddir"] = "/home3/jmlauder/Jupyter_notebooks/ECCOv4r5_grid"
    elif version == "ECCOv4r5_JRA55DO_ALL":
        metadata["datadir"] = "/nobackup/rsavelli/ECCO_V4r5/runoff_exp/darwin3/run_ALL/diags/budget"
        metadata["griddir"] = "/home3/jmlauder/Jupyter_notebooks/ECCOv4r5_grid"
    
    # Easy to extract metadata
    metadata["chunks"]  = {'k':-1, 'k_u':-1, 'k_l':-1, 'k_p1':-1,'j':-1,'j_g':-1,'i':-1,'i_g':-1,'face':-1}
    metadata['iters']   = [int(x) for x in [y[-15:-5] for y in sorted(gb.glob(metadata["datadir"]+'/average_2d.*.data'))]]
    metadata["refdate"] = "1992-1-1 12:0:0"
    return metadata, plot_dx, plot_dy

def getparm(path_to_namelist,usef90nml=True,flatten=True):
    """
       Read in Namelist file to a dictionary as strings or floats
    """
    
    if usef90nml:
        import f90nml

        parser = f90nml.Parser()
        parser.comment_tokens += '#'
        
        mydata = parser.read(path_to_namelist)
        
        # Want to flatten by removing namelist sections like "parm01, parm02, parm03....?
        if flatten:
            myparms={}
            for k in mydata.keys():
                myparms=dict(myparms, **mydata[k])
        else:
            myparms=mydata.todict()
    else: # Dont use f90nml (it works with data.diagnostics if fields(1:15,1) etc type statements are used)
        myparms={}
        it=0
        file=open(path_to_namelist, 'r').readlines()
        while it < len(file)-1:
            key=[]
            key1=[]
            value=[]
            line, _, comment = file[it].partition('#')
            if line.strip(): # non-blank line
                line, _, comment = line.partition('&')
                if line.strip(): # non-blank line
                    key1, _, value = line.partition('=')
                    key,_,loc=key1.strip(' .,\)\#').partition('(')
                    if key == 'fields': # Data.diagnostics specific
                        # Do some looking ahead to see if there are variables on the next line
                        while file[it+1].find('=') == -1:
                            it += 1 #Increment the counter
                            line, _, comment = file[it].partition('#')
                            if line.strip(): # non-blank line
                                line, _, comment = line.partition('&')
                                if line.strip(): # non-blank line
                                    value=value.strip(' \"\n')+line.strip(' \t')
                    try: 
                        if key.strip().lower() in myparms.keys(): # append value to a key in myvars 
                            myparms[key.strip().lower()].append(np.float(value.strip(' ,.\'\"\n')))
                        else: # Cannot append to a key that doesnt exist so create it as an array  
                            myparms[key.strip().lower()]=[np.float(value.strip(' ,.\'\"\n'))]
                    except ValueError:
                        if key.strip().lower() == 'fields':
                            if key.strip().lower() in myparms.keys(): # append value to a key in myvars 
                                myparms[key.strip().lower()].append(value.strip(' ,.\'\"\n').strip('\'').strip('\ ').replace(' ','').split('\',\''))
                            else: # Cannot append to a key that doesnt exist so create it as an array  
                                myparms[key.strip().lower()]=[value.strip(' ,.\'\"\n').strip('\'').strip('\ ').replace(' ','').split('\',\'')]
                        else:
                            if key.strip().lower() in myparms.keys(): # append value to a key in myvars 
                                myparms[key.strip().lower()].append(value.strip().strip(' ,.\'\"\n'))
                            else: # Cannot append to a key that doesnt exist so create it as an array  
                                myparms[key.strip().lower()]=[value.strip().strip(' ,.\'\"\n')]
            # Increment the counter
            it += 1
                            
    return myparms

def get_llc_face_connections():
    # https://gist.github.com/rabernat/a4158e23f50470f1a55be54910a2134b
    # normal (non-reverse) connections
    # W -> E, E -> W
    # S -> N, N -> S
    # W -> N, E -> S
    # S -> E, N -> W
    face_edge_link = {
        0:  {"N": (1,  "S",  1), "S": None, "E": (3 , "W",  1) , "W": (12 , "N",            -1)},
        1:  {"N": (2,  "S",  1), "S": (0  , "N",  1), "E": (4  , "W",   1), "W": (11,  "N", -1)},
        2:  {"N": (6,  "W", -1), "S": (1  , "N",  1), "E": (5  , "W",   1), "W": (10,  "N", -1)},
        3:  {"N": (4,  "S",  1), "S": None, "E": (9 , "S", -1) , "W":  (0 , "E",             1)},
        4:  {"N": (5,  "S",  1), "S": (3  , "N",  1), "E": (8  , "S",  -1), "W": ( 1,  "E",  1)},
        5:  {"N": (6,  "S",  1), "S": (4  , "N",  1), "E": (7  , "S",  -1), "W": ( 2,  "E",  1)},
        6:  {"N": (10, "W", -1), "S": (5  , "N",  1), "E": (7  , "W",   1), "W": ( 2,  "N", -1)},
        7:  {"N": (10, "S",  1), "S": (5  , "E", -1), "E": (8  , "W",   1), "W": ( 6,  "E",  1)},
        8:  {"N": (11, "S",  1), "S": (4  , "E", -1), "E": (9  , "W",   1), "W": ( 7,  "E",  1)},
        9:  {"N": (12, "S",  1), "S": (3  , "E", -1), "E": None, "W":  (8 , "E",             1)},
        10: {"N": (2,  "W", -1), "S": (7  , "N",  1), "E": (11 , "W",   1), "W": ( 6,  "N", -1)},
        11: {"N": (1,  "W", -1), "S": (8  , "N",  1), "E": (12 , "W",   1), "W": (10,  "E",  1)},
        12: {"N": (0,  "W", -1), "S": (9  , "N",  1), "E": None, "W": (11 , "E",             1)},
    }
    
    face_connections = {}
    for k in range(13):
        links  = face_edge_link[k]
        x0, x1 = links["W"], links["E"]
        y0, y1 = links["S"], links["N"]
        if x0:
            x0reverse = x0[1] not in ["E", "N"]
            x0ax      = "X" if x0[1] in ["E", "W"] else "Y"
            x0        = (x0[0],) + (x0ax, x0reverse)
        if x1:
            x1reverse = x1[1] not in ["W", "S"]
            x1ax      = "X" if x1[1] in ["E", "W"] else "Y"
            x1        = (x1[0],) + (x1ax, x1reverse)
        if y0:
            y0reverse = y0[1] not in ["N", "E"]
            y0ax      = "Y" if y0[1] in ["N", "S"] else "X"
            y0        = (y0[0],) + (y0ax, y0reverse)
        if y1:
            y1reverse = y1[1] not in ["S", "W"]
            y1ax      = "Y" if y1[1] in ["N", "S"] else "X"
            y1        = (y1[0],) + (y1ax, y1reverse)
        face_connections[k] = {"X": (x0, x1), "Y": (y0, y1)}
    return {"face": face_connections}

def get_xgcm_face_connections():
    # define the connectivity between faces
    return {'face':{
                    0: {'X':  ((12, 'Y', False), (3, 'X', False)),
                        'Y':  (None,             (1, 'Y', False))},
                    1: {'X':  ((11, 'Y', False), (4, 'X', False)),
                        'Y':  ((0, 'Y', False),  (2, 'Y', False))},
                    2: {'X':  ((10, 'Y', False), (5, 'X', False)),
                        'Y':  ((1, 'Y', False),  (6, 'X', False))},
                    3: {'X':  ((0, 'X', False),  (9, 'Y', False)),
                        'Y':  (None,             (4, 'Y', False))},
                    4: {'X':  ((1, 'X', False),  (8, 'Y', False)),
                        'Y':  ((3, 'Y', False),  (5, 'Y', False))},
                    5: {'X':  ((2, 'X', False),  (7, 'Y', False)),
                        'Y':  ((4, 'Y', False),  (6, 'Y', False))},
                    6: {'X':  ((2, 'Y', False),  (7, 'X', False)),
                        'Y':  ((5, 'Y', False),  (10, 'X', False))},
                    7: {'X':  ((6, 'X', False),  (8, 'X', False)),
                        'Y':  ((5, 'X', False),  (10, 'Y', False))},
                    8: {'X':  ((7, 'X', False),  (9, 'X', False)),
                        'Y':  ((4, 'X', False),  (11, 'Y', False))},
                    9: {'X':  ((8, 'X', False),  None),
                        'Y':  ((3, 'X', False),  (12, 'Y', False))},
                    10: {'X': ((6, 'Y', False),  (11, 'X', False)),
                         'Y': ((7, 'Y', False),  (2, 'X', False))},
                    11: {'X': ((10, 'X', False), (12, 'X', False)),
                         'Y': ((8, 'Y', False),  (1, 'X', False))},
                    12: {'X': ((11, 'X', False), None),
                         'Y': ((9, 'Y', False),  (0, 'X', False))}
                  }
           }

def get_xgcm_grid_metrics():
    return {
    ("X",)    : ["dxC", "dxG"],  # X distances
    ("Y",)    : ["dyC", "dyG"],  # Y distances
    ("Z",)    : ["dzW", "dzS", "dzC"],  # Z distances
    ("X", "Y"): ["rA", "rAz", "rAs", "rAw"],  # Areas
}

def get_xgcm_extra_attributes():
    return {
    "i": {
        "axis"              : "X",
        "c_grid_axis_shift" : 0.0,
        "long_name"         : "x-dimension of the c grid",
        "standard_name"     : "x_grid_index_at_c_location",
        "swap_dim"          : "XC",
    },
    "j": {
        "axis"              : "Y",
        "c_grid_axis_shift" : 0.0,
        "long_name"         : "y-dimension of the c grid",
        "standard_name"     : "y_grid_index_at_c_location",
        "swap_dim"          : "YC",
    },
    "k": {
        "axis"              : "Z",
        "c_grid_axis_shift" : 0.0,
        "long_name"         : "z-dimension of the w grid",
        "standard_name"     : "z_grid_index_at_c_location",
        "swap_dim"          : "Z",
    },
    "i_g": {
        "axis"              : "X",
        "c_grid_axis_shift" : -0.5,
        "long_name"         : "x-dimension of the v grid",
        "standard_name"     : "x_grid_index_at_v_location",
        "swap_dim"          : "XG",
    },
    "j_g": {
        "axis"              : "Y",
        "c_grid_axis_shift" : -0.5,
        "long_name"         : "y-dimension of the v grid",
        "standard_name"     : "y_grid_index_at_v_location",
        "swap_dim"          : "YG",
    },
    "XC": {
        "axis"              : "X",
        "c_grid_axis_shift" : 0.0,
        "long_name"         : "x-dimension of the c grid",
        "standard_name"     : "x_grid_index_at_c_location",
    },
    "YC": {
        "axis"              : "Y",
        "c_grid_axis_shift" : 0.0,
        "long_name"         : "y-dimension of the c grid",
        "standard_name"     : "y_grid_index_at_c_location",
    },
    "YG": {
        "axis"              : "Y",
        "c_grid_axis_shift" : -0.5,
        "long_name"         : "y-dimension of the v grid",
        "standard_name"     : "y_grid_index_at_v_location",
    },
    "XG": {
        "axis"              : "X",
        "c_grid_axis_shift" : -0.5,
        "long_name"         : "x-dimension of the v grid",
        "standard_name"     : "x_grid_index_at_v_location",
    },
    "Z": {  # This is the cell centre
        "axis"              : "Z",
        "c_grid_axis_shift" : 0.0,
        "long_name"         : "z-dimension of the w grid",
        "standard_name"     : "vertical coordinate of cell center",
    },
    "Zu": {  # This is the bottom interface
        "axis"              : "Z",
        "c_grid_axis_shift" : -0.5,
        "long_name"         : "z-dimension of the w grid",
        "standard_name"     : "vertical coordinate of upper cell interface",
    },
    "Zl": {  # This is the upper interface
        "axis"              : "Z",
        "c_grid_axis_shift" : 0.5,
        "long_name"         : "z-dimension of the w grid",
        "standard_name"     : "vertical coordinate of lower cell interface",
    },
    "Zp1": {  # This is the outer edges
        "axis"              : "Z",
        "c_grid_axis_shift" : -0.5,
        "long_name"         : "z-dimension of the w grid",
        "standard_name"     : "vertical coordinate of cell interface",
    },
}

def get_xgcm_extra_coordinates(ecco_grid):
    extra_coords = {
            "dzC":  ecco_grid["maskC"].astype(np.float32) * ecco_grid["drF"],
            "dzW":  ecco_grid["maskW"].astype(np.float32) * ecco_grid["drF"],
            "dzS":  ecco_grid["maskS"].astype(np.float32) * ecco_grid["drF"],
           "cvol": (ecco_grid["hFacC"] * ecco_grid["rA" ]*ecco_grid["drF"]).where(ecco_grid["maskC"]).transpose('k','face','j','i'),
           "uvol": (ecco_grid["hFacW"] * ecco_grid["rAw"]*ecco_grid["drF"]).where(ecco_grid["maskW"]).transpose('k','face','j','i_g'),
           "vvol": (ecco_grid["hFacS"] * ecco_grid["rAs"]*ecco_grid["drF"]).where(ecco_grid["maskS"]).transpose('k','face','j_g','i'),
        }
        
    ecco_grid = ecco_grid.assign_coords(extra_coords)
    
    # Have to add back the attributes for i, j, and k for some reason
    extra_attrs = get_xgcm_extra_attributes()
    for dim in extra_attrs.keys():
        ecco_grid[dim].attrs = extra_attrs[dim]
    return ecco_grid

def get_eccodarwin_initial_conditions(metadata,data_ptracers,tracer_ids=[1,2,3,4,5,6,7,18,19]):
    # Figure out if we need to load a pickup or a previous snapshot
    files = sorted(gb.glob(metadata["datadir"]+'/snap_3d.*.data'))
    idx   = files.index([m for m in files if re.search(str(metadata['iters'][0]), m)][0])

    if idx == 0:
        # Load pickup files
        extra_variables=dict()
        for ii in tracer_ids: # DIC, NO3, NO2, NH4, PO4, FeT, SiO2, ALK, O2
            extra_variables['pTr{0:02d}'.format(ii)] = dict(
                dims  = ['k','j','i'],
                nx    = metadata["nx"],
                ny    = metadata["ny"],
                nz    = metadata["nz"],
                attrs =dict(
                            standard_name = data_ptracers['ptracers_names'][ii-1],
                            units         = data_ptracers['ptracers_units'][ii-1],
                            coordinate    = "Z YC XC",
           )
        )  
        
        ecco_initial_conditions     = xm.open_mdsdataset(
                prefix              = ['pickup_ptracers'],
                iters               = data_ptracers['ptracers_iter0'], 
                data_dir            = metadata['datadir'].replace('diags/budget',''), 
                grid_dir            = metadata['griddir'], 
                delta_t             = metadata['deltat'],
                ref_date            = metadata['refdate'],
                nx                  = metadata["nx"],
                ny                  = metadata["ny"],
                nz                  = metadata["nz"],
                chunks              = metadata["chunks"],
                extra_variables     = extra_variables,
                ignore_unknown_vars = True, # only load in the tracers detailed in "extra_tracers", others are ignored
                read_grid           = True,
                geometry            = "llc",
                llc_method          = 'smallchunks',
                default_dtype       = np.float32
            ).rename(
                {'pTr{0:02d}'.format(ii):'TRAC{0:02d}'.format(ii) for ii in tracer_ids}
        )
        
        # Read in the pickup file - it's weird because nlevels is not 50 so that the file can contain 3d and 2d variables
        mitgcm_pickup = ecco.read_llc_to_tiles(
                    fdir        = metadata['datadir'].replace('diags/budget',''),
                    fname       = 'pickup.{0:010d}.data'.format(data_ptracers['ptracers_iter0']),
                    nk          = metadata["nz"]*9+3,
                    llc         = metadata["nx"],
                    use_xmitgcm = True,
                    filetype    = ">d",
                    less_output = True,
                ).squeeze()
        
        # Extract just the variables in the snapshot diagnostics, ETAN, THETA, and SALT
        ecco_initial_conditions['ETAN']=xr.DataArray(
                    np.squeeze(mitgcm_pickup[-3,:,:,:])[np.newaxis,:,:,:],
                    coords={
                        "time" : ecco_initial_conditions.time.values,
                        "face" : np.arange(len(metadata['face_facets'])),
                        "j"    : np.arange(metadata['nx']),
                        "i"    : np.arange(metadata['nx']),
                    },
                    dims=["time", "face", "j", "i"],
                )
        
        ecco_initial_conditions['THETA']=xr.DataArray(
                    np.squeeze(mitgcm_pickup[100:150,:,:,:])[np.newaxis,:,:,:],
                    coords={
                        "time" : ecco_initial_conditions.time.values,
                        "k"    : np.arange(metadata['nz']),
                        "face" : np.arange(len(metadata['face_facets'])),
                        "j"    : np.arange(metadata['nx']),
                        "i"    : np.arange(metadata['nx']),
                    },
                    dims=["time", "k", "face", "j", "i"],
                )
        
        ecco_initial_conditions['SALT']=xr.DataArray(
                    np.squeeze(mitgcm_pickup[150:200,:,:,:])[np.newaxis,:,:,:],
                    coords={
                        "time" : ecco_initial_conditions.time.values,
                        "k"    : np.arange(metadata['nz']),
                        "face" : np.arange(len(metadata['face_facets'])),
                        "j"    : np.arange(metadata['nx']),
                        "i"    : np.arange(metadata['nx']),
                    },
                    dims=["time", "k", "face", "j", "i"],
                )
        
        # Adjust initial time point 
        ecco_initial_conditions["time_midnight_jan1st"] = xr.DataArray(
            data=datetime.datetime.strptime(metadata['refdate'].replace("12:0:0","0:0:0"),'%Y-%m-%d %H:%M:%S'),
            dims=["time"],
            coords={"time": ecco_initial_conditions.time.values},
        )
        
        ecco_initial_conditions=ecco_initial_conditions.set_index(time="time_midnight_jan1st")
    else:
        # Read in the snapshot just before the first iteration listed
        ecco_initial_conditions = xm.open_mdsdataset(
            prefix        = ['snap_3d','snap_2d'],
            iters         = files[idx-1].replace(
                                            metadata['datadir'],""
                                       ).replace(
                                            "/snap_3d.",""
                                       ).replace(
                                            ".data",""
                                       ), 
            data_dir      = metadata['datadir'], 
            grid_dir      = metadata['griddir'], 
            delta_t       = metadata['deltat'],
            ref_date      = metadata['refdate'],
            nx            = metadata["nx"],
            ny            = metadata["ny"],
            nz            = metadata["nz"],
            chunks        = metadata["chunks"],
            read_grid     = True,
            geometry      = "llc",
            llc_method    = 'smallchunks',
            default_dtype = np.float32
        )
    return ecco_initial_conditions

def ecco_zonal_average(fld, lat_bins, grid, basin_name=None, basin_path='/home/jml1/.conda/envs/mitgcm/binary_data/'):
    """
    Compute weighted average of a quantity at each depth level
    across latitude(s), defined in lat_vals, in an LLC grid. 

    Uses xarray groupby_bins and weights calculated by surface area

    Parameters
    ----------
    fld : xarray DataArray
        3D spatial (+ time, optional) field
    lat_vals : float or list
        latitude value(s) specifying where to compute average
    coords : xarray Dataset
        only needs YC, and optional masks (defining wet points)
    basin_name : string, optional
        denote ocean basin over which to compute streamfunction
        If not specified, compute global quantity
        see get_basin.get_available_basin_names for options

    Returns
    -------
    ds_out : xarray Dataset
        with the main variable
            'average'
                average quantity across denoted latitude band at
                each depth level with dimensions 'time' (if in given dataset),
                'k' (depth), and 'lat'
    """
    # Get basin mask
    maskC = grid.maskC.load() if 'maskC' in grid.coords else xr.ones_like(fld).compute()

    if basin_name is not None:
        maskC = ecco.get_basin_mask(
            basin_name,
            maskC.rename({'face':'tile'}),
            basin_path=basin_path,
            less_output=True,
        ).rename({'tile':'face'})

    area = grid.rA.load()

    # These sums are the same for all lats, therefore precompute to save time
    tmp_c = fld.where(maskC).load()

    # Coordinate labels for the binned values
    xbins = np.arange(0,len(lat_bins))
    dbins = np.mean(np.diff(lat_bins))
    nbins = np.arange(dbins/2, len(lat_bins)-1)

    lat_labs = np.interp(nbins, xbins, lat_bins)

    da_mean = (
            (tmp_c*area).where(maskC)
        ).groupby_bins(
            "YC",
            lat_bins,
            labels=lat_labs,
        ).sum()/(
            (area).where(maskC)
        ).groupby_bins(
            "YC",
            lat_bins,
            labels=lat_labs,
        ).sum()
    return da_mean.assign_coords(z=('k',grid.Z.data)).swap_dims({'k':'z'}).rename({"YC_bins":"Latitude","z":"Depth"})
