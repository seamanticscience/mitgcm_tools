# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:07:24 2017

@author: jml1
"""
import glob    as gb
import getpass as gp
import numpy   as np
import pandas  as pd
import xarray  as xr
import xgcm

# %% REGRIDDING, AXES, AND FILE LOADING ROUTINES
def open_ncfile(file_pattern,doconform_axes=True,chunking=None,strange_axes=dict(),grid=[]):
    """
        Read in data from a netcdf file (file_pattern) using xarray, which can be chunked via dask by
          setting chunking to a dictionary of chunks (e.g. {'T':2,'X':10,'Y':10,'Z':2}).
        For compatability with xgcm, the axes may need to be conformed to certain specifications. 
          set conform_axes=False to override this. We can handle conversions between many axis names, but
          if there is a particularly difficult set (thanks, pkg/diagnostics) set the conversion within the
          "strange_axes" dictionary strange_axes={'Xnp':'XN','Ynp':'YN'}
    """
    if doconform_axes:
        if not grid:
            data=conform_axes(xr.open_dataset(gb.glob(file_pattern)[0],chunks=chunking),strange_ax=strange_axes)
        else:
            data=conform_axes(xr.open_dataset(gb.glob(file_pattern)[0],chunks=chunking),strange_ax=strange_axes,grd=grid)
    else:
        data=xr.open_dataset(gb.glob(file_pattern)[0],chunks=chunking)
    return data

def open_bnfile(fname,sizearr=(12,15,64,128),prec='>f4'):
    """ open_bnfile(fname,sizearr,prec) reads a binary file and returns it as a
        numpy array.
        
        fname is the file name,
        prec  is the precision and dtype argument ('>f4' works usually)
        sizearr is the anticipated size of the returned array
    """    
    
    try:
        binin=np.fromfile(fname,dtype=prec).reshape(sizearr)
    except ValueError:
        try:
            # Try dropping the last dimension of sizearr
            binin=np.fromfile(fname,dtype=prec).reshape(sizearr[1:])
        except ValueError:
            print('ValueError: cannot reshape array into shape '+np.str(sizearr))
            raise
    
    return binin
    
def loadgrid(fname='grid.glob.nc',basin_masks=True):
    """ loadgrid(fname,sizearr,prec) reads a netcdf grid file and returns it as a
        xarray, with a few additional items.
        
        fname is the file name,
    """     
    grd=xr.open_dataset(fname)
    
    if "T" in grd.coords:
        grd=grd.squeeze('T')
    
    # Preserve these arrays
    grd['lonc']=grd.XC
    grd['latc']=grd.YC
    grd['lonu']=grd.dxC*0+grd.XG[0,:]
    grd['latu']=grd.dxC*0+grd.YC[:,0]
    grd['lonv']=grd.dyC*0+grd.XC[0,:]
    grd['latv']=grd.dyC*0+grd.YG[:,0]
    grd['lonz']=grd.dxV*0+grd.XG[0,:]
    grd['latz']=grd.dxV*0+grd.YG[:,0]
    
    grd['cmask']=grd.HFacC.where(grd.HFacC>grd.HFacC.min())
    grd['umask']=grd.HFacW.where(grd.HFacW>grd.HFacW.min())
    grd['vmask']=grd.HFacS.where(grd.HFacS>grd.HFacS.min())
    grd['depth']=(grd.R_low*grd.cmask)
      
    grd['dzC'] = grd.HFacC * grd.drF #vertical cell size at tracer point
    grd['dzW'] = grd.HFacW * grd.drF #vertical cell size at u point
    grd['dzS'] = grd.HFacS * grd.drF #vertical cell size at v point
      
    # Reshape axes to have the same dimensions
    grd['cvol']=(grd.HFacC*grd.rA*grd.drF).where(grd.HFacC>=grd.HFacC.min())
    grd['uvol']=(grd.HFacW*grd.rAw*grd.drF).where(grd.HFacW>=grd.HFacW.min())
    grd['vvol']=(grd.HFacS*grd.rAs*grd.drF).where(grd.HFacS>=grd.HFacS.min())
    
    if basin_masks:
    # Get basin masks
        atlantic_mask, pacific_mask, indian_mask, so_mask, arctic_mask = oceanmasks(grd.lonc.transpose('X','Y').data,grd.latc.transpose('X','Y').data,grd.cmask.transpose('X','Y','Z').data)
        grd['cmask_atlantic'] = xr.DataArray(atlantic_mask, coords=[grd.X.data, grd.Y.data, grd.Z.data], dims=['X', 'Y', 'Z'])
        grd['cmask_pacific']  = xr.DataArray(pacific_mask , coords=[grd.X.data, grd.Y.data, grd.Z.data], dims=['X', 'Y', 'Z'])
        grd['cmask_indian']   = xr.DataArray(indian_mask  , coords=[grd.X.data, grd.Y.data, grd.Z.data], dims=['X', 'Y', 'Z'])
        grd['cmask_so']       = xr.DataArray(so_mask      , coords=[grd.X.data, grd.Y.data, grd.Z.data], dims=['X', 'Y', 'Z'])
        grd['cmask_arctic']   = xr.DataArray(arctic_mask  , coords=[grd.X.data, grd.Y.data, grd.Z.data], dims=['X', 'Y', 'Z'])
        grd['cmask_nh']       = grd.cmask.where(grd.coords['Y']>0)
        grd['cmask_sh']       = grd.cmask.where(grd.coords['Y']<=0)
    
        atlantic_mask, pacific_mask, indian_mask, so_mask, arctic_mask = oceanmasks(grd.lonu.transpose('Xp1','Y').data,grd.latu.transpose('Xp1','Y').data,grd.umask.transpose('Xp1','Y','Z').data)
        grd['umask_atlantic'] = xr.DataArray(atlantic_mask, coords=[grd.Xp1.data, grd.Y.data, grd.Z.data], dims=['Xp1', 'Y', 'Z'])
        grd['umask_pacific']  = xr.DataArray(pacific_mask , coords=[grd.Xp1.data, grd.Y.data, grd.Z.data], dims=['Xp1', 'Y', 'Z'])
        grd['umask_indian']   = xr.DataArray(indian_mask  , coords=[grd.Xp1.data, grd.Y.data, grd.Z.data], dims=['Xp1', 'Y', 'Z'])
        grd['umask_so']       = xr.DataArray(so_mask      , coords=[grd.Xp1.data, grd.Y.data, grd.Z.data], dims=['Xp1', 'Y', 'Z'])
        grd['umask_arctic']   = xr.DataArray(arctic_mask  , coords=[grd.Xp1.data, grd.Y.data, grd.Z.data], dims=['Xp1', 'Y', 'Z'])
        grd['umask_nh']       = grd.umask.where(grd.coords['Y']>0)
        grd['umask_sh']       = grd.umask.where(grd.coords['Y']<=0)
     
        atlantic_mask, pacific_mask, indian_mask, so_mask, arctic_mask = oceanmasks(grd.lonv.transpose('X','Yp1').data,grd.latv.transpose('X','Yp1').data,grd.vmask.transpose('X','Yp1','Z').data)
        grd['vmask_atlantic'] = xr.DataArray(atlantic_mask, coords=[grd.X.data, grd.Yp1.data, grd.Z.data], dims=['X', 'Yp1', 'Z'])
        grd['vmask_pacific']  = xr.DataArray(pacific_mask , coords=[grd.X.data, grd.Yp1.data, grd.Z.data], dims=['X', 'Yp1', 'Z'])
        grd['vmask_indian']   = xr.DataArray(indian_mask  , coords=[grd.X.data, grd.Yp1.data, grd.Z.data], dims=['X', 'Yp1', 'Z'])
        grd['vmask_so']       = xr.DataArray(so_mask      , coords=[grd.X.data, grd.Yp1.data, grd.Z.data], dims=['X', 'Yp1', 'Z'])
        grd['vmask_arctic']   = xr.DataArray(arctic_mask  , coords=[grd.X.data, grd.Yp1.data, grd.Z.data], dims=['X', 'Yp1', 'Z'])
        grd['vmask_nh']       = grd.vmask.where(grd.coords['Yp1']>0)
        grd['vmask_sh']       = grd.vmask.where(grd.coords['Yp1']<=0)
    
    grd.close()
    
    # These variable conflict with future axis names
    grd=grd.drop(['XC','YC','XG','YG'])
    
    # Attempt to conform axes to conventions
    grd=conform_axes(grd)

    # generate XGCM grid, with metrics for grid aware calculations
    # Have to make sure the metrics are properly masked
    # issue for area, but not volume... 
    grd['rA' ]=grd.rA * grd.HFacC.isel(ZC=0)
    grd['rAs']=grd.rAs* grd.HFacS.isel(ZC=0)
    grd['rAw']=grd.rAw* grd.HFacW.isel(ZC=0)
    # This is dodgy, but not sure what else to do...
    grd['rAz']=grd.rAz* (grd.HFacW*grd.HFacS).isel(ZC=0)

    metrics = {
        ('X',): ['dxC', 'dxG'], # X distances
        ('Y',): ['dyC', 'dyG'], # Y distances
        ('Z',): ['dzW', 'dzS', 'dzC'], # Z distances
        ('X', 'Y'): ['rA', 'rAz', 'rAs', 'rAw'] # Areas
        }
    xgrd = xgcm.Grid(grd,periodic=['X','Y'],metrics=metrics)
    
    return grd,xgrd
    
def conform_axes(dsin,strange_ax=dict(),grd=[]):
    """
       Make the netcdf file dimensions conform to the standards required by xgcm by:
        1.) Rename axes to better represent "Center, Outer, Left, Right"
        2.) Add "axis" and "c_grid_axis_shift" attributes to each axis.
        
        By default conform_axes recognises X, Y, Xp1, Yp1, Z, Zp1, Zu, and Zl axes, but 
          addtional axis names (like Zd000001 or Zmd000015 from pkg/diagnostics) can be supplied using
          the "strange_ax" input arguement, which renames the axis and draws coordinate values from  
          grid_xdset (xarray dataset of loaded "grid" files):
              
       oceDiag=mitgcm_tools.conform_axes(oceDiag_xdset,strange_axes={'Zmd000015':'ZC','Zld000015':'ZL'},grid=grid_xdset)
    """
        
    default_ax={'X':'XC','Y':'YC','Xp1':'XG','Yp1':'YG','Z':'ZC','Zp1':'ZG','Zu':'ZR','Zl':'ZL'}
    ax_conv_dict = dict(default_ax, **strange_ax)
    
    missing_ax=[]
    for ax in dsin.dims:
        try:
            dsin=dsin.rename({ax:ax_conv_dict[ax]})
        except KeyError:
            missing_ax.append(ax)
    
    if missing_ax != []:        
        print("The axes: "+','.join(missing_ax)+" could not be converted")
    
    missing_ax=[]
    for ax in dsin.dims:
        try:
            dsin[ax].attrs['axis']=ax[0]
            if ax[-1]=='C':
                dsin[ax].attrs['c_grid_axis_loc']  = 'Center'
                dsin[ax].attrs['c_grid_axis_shift']= 0.0 # Center
            elif ax[-1]=='G':
                dsin[ax].attrs['c_grid_axis_loc']  ='Outer'
                dsin[ax].attrs['c_grid_axis_shift']=-0.5 # Outer (n+1) Edges
            elif ax[-1]=='L':
                dsin[ax].attrs['c_grid_axis_loc']  ='Left'
                dsin[ax].attrs['c_grid_axis_shift']=-0.5 # Left
            elif ax[-1]=='R':
                dsin[ax].attrs['c_grid_axis_loc']  = 'Right'
                dsin[ax].attrs['c_grid_axis_shift']= 0.5 # Right
            else:
                missing_ax.append(ax)
        except KeyError:
            missing_ax.append(ax)
    
    if missing_ax != []:        
        print("Attributes could not be added for axes: "+','.join(missing_ax))

    # Check that all the dimensions have coordinate values. Should be able to import these from grid_data
    missing_ax=[]
    for ax in dsin.dims:
        if ax not in dsin.coords:
            missing_ax.append(ax)
            if not grd:
                # Be cheeky and reload the grid, which should have the axes needed
                grd=xr.open_dataset(gb.glob("grid*nc")[0]).squeeze('T')
                grd.close()
                grd=grd.drop(['XC','YC','XG','YG'])
                grd=conform_axes(grd)
            
            if dsin.dims[ax] > 1:
                try:
                    dsin=dsin.assign_coords(ax=grd.coords[ax])
                except NameError:
                    try:
                        dsin=dsin.assign_coords(ax=grid.coords[ax])
                    except NameError:
                        try:
                            dsin=dsin.assign_coords(ax=gridfile.coords[ax])
                        except NameError:
                            raise()                   
                    
                
            else:
                # Hack for the single layer diag files
                # This isnt working....
#                dsin=dsin.assign_coords(ax=dsin[ax])
                # Squeeze the singleton dimension
                dsin=dsin.squeeze(ax)
            
            if missing_ax != []:        
                    print("Coordinates added for axes: "+','.join(missing_ax))    
    
    # Bit annoying that "ax" gets added as a coordinate each time too! Zap it.
    if 'ax' in dsin.coords:
        dsin=dsin.drop('ax')

    return dsin

def oceanmasks(xc,yc,maskin):    
    from scipy.interpolate import griddata

    nzdim=0
    # Find if input dimensions are 3d or 2d
    if np.ndim(maskin)>2:
        nzdim=np.size(maskin,2)
        if np.ndim(xc)>2:
            xc=xc[:,:,0]
        if np.ndim(yc)>2:
            yc=yc[:,:,0]
    
    try: # Try to read from url
        url = "https://data.nodc.noaa.gov/woa/WOA13/MASKS/basinmask_01.msk"
        c = pd.read_csv(url,header=1)
        
        x = c.Longitude.values
        y = c.Latitude .values
        basinfile = c.Basin_0m.values
    except "HTTPError": # Fall back on the file downloaded
        mask_file='/Users/'+gp.getuser()+'/Dropbox_Work/Applications/MATLAB/mitgcm_toolbox/woa13_basinmask_01.msk'

        x = np.loadtxt(mask_file,delimiter=',',usecols=(1,),skiprows=2)
        y = np.loadtxt(mask_file,delimiter=',',usecols=(0,),skiprows=2)
        basinfile = np.loadtxt(mask_file,delimiter=',',usecols=(2,),skiprows=2)
        
    # Find out if the grid has been rotated and rotate so range is the same as input grid
    if (np.min(x)<0) != (np.min(xc)<0):
        x[x<0]=x[x<0]+360

    basinmask = griddata((x, y), basinfile, (xc,yc), method = 'nearest')

    basinmask[basinmask==12]=2 # Add the Sea of Japan to the Pacific
    basinmask[basinmask==56]=3 # Add Bay of Bengal to Indian Ocean
    basinmask[basinmask==53]=0 # Zero out Caspian Sea

    so_mask     = np.copy(basinmask)
    so_mask[so_mask!=10]=0
    so_mask[so_mask==10]=1
    arctic_mask = np.copy(basinmask)
    arctic_mask[arctic_mask!=11]=0
    arctic_mask[arctic_mask==11]=1

    # Divide Southern Ocean into Atlantic, Indian and Pacific Sectors
    tmp=basinmask[:,0:len(np.unique(yc[yc<=-45]))] 
    basinmask[:,0:np.size(tmp,1)]=np.transpose(np.tile(tmp[:,-1],[np.size(tmp,1),1]))
    atlantic_mask = np.copy(basinmask)
    atlantic_mask[atlantic_mask!=1]=0
    atlantic_mask[atlantic_mask==1]=1
    indian_mask   = np.copy(basinmask)
    indian_mask[indian_mask!=3]=0
    indian_mask[indian_mask==3]=1
    pacific_mask  = np.copy(basinmask)
    pacific_mask[pacific_mask!=2]=0
    pacific_mask[pacific_mask==2]=1
    
    # if input was 3d, then extent mask to 3d
    if nzdim>0:
        atlantic_mask = np.tile(atlantic_mask[:,:,np.newaxis],(1,1,nzdim))*maskin
        pacific_mask  = np.tile(pacific_mask [:,:,np.newaxis],(1,1,nzdim))*maskin
        indian_mask   = np.tile(indian_mask  [:,:,np.newaxis],(1,1,nzdim))*maskin
        so_mask       = np.tile(so_mask      [:,:,np.newaxis],(1,1,nzdim))*maskin
        arctic_mask   = np.tile(arctic_mask  [:,:,np.newaxis],(1,1,nzdim))*maskin
        
    return atlantic_mask, pacific_mask, indian_mask, so_mask, arctic_mask

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

# %% DATA ANALYSIS ROUTINES
def get_dicpco2(data_parms,data_dic,grid,path='./'):
    """get_dicpco2 loads output from the DIC package relating to the atmospheric
       pco2 boundary condition (constant, read in from a file, or dynamic).
       It interogates a bunch of different sources.
    """
    # Generate the initial dataset with xarray and info from data_parms to get runtimes/iterations
    try:
        # if "grid" contains the time axes from your datafiles already, great!
        # should come with "T" as a bonus coordinate!
        atm_box=grid.iter.to_dataset()
    except (KeyError, AttributeError):
        # Time/iter axes not included in the "grid" dataarray, try and work it out...
        try:
            run_iters= np.linspace(data_parms['niter0'],
                                  (data_parms['niter0']+data_parms['ntimesteps']),
                                   data_parms['ntimesteps']/(data_parms['tavefreq']/data_parms['deltatclock'])+1
                                  )[1:]
            run_T    = run_iters*data_parms['deltatclock']
            
            # make xarray dataset
            atm_box=xr.Dataset({"iter": (("T"), run_iters)},{"T":run_T})
        except (KeyError):
            # Perhaps try to use starttime/endtime parameters, after which I give up.
            try:
                run_T    = np.linspace(data_parms['starttime'],
                                       data_parms['endtime'],
                                       (data_parms['endtime']-data_parms['starttime'])/data_parms['tavefreq']+1
                                      )[1:]
                run_iters= run_T/data_parms['deltatclock']
                
                # make xarray dataset
                atm_box=xr.Dataset({"iter": (("T"), run_iters)},{"T":run_T})
            except:
                print("Could not determin time/iter axis values from available information")
                raise
    
    # Use some external functions to get atm_box output, you can call them seperately, but it's not recommended!
    try:
        if data_dic['dic_int1']==0: 
            # use default value of 278
            atm_box=read_dicpco2_default(atm_box)         

        elif data_dic['dic_int1']==1:
        # use the value read in from dic_pCO2
            atm_box=read_dicpco2_fixedvl(atm_box,data_dic)
            
        elif data_dic['dic_int1']==2:
        # use the value read in from a file (co2atmos.dat) 
            atm_box=read_dicpco2_prescri(atm_box,data_dic,path)
            
        elif data_dic['dic_int1']>=3:
            # Using the interactive atmospheric box which produces pickup files and may have diagnostics
            try:
                atm_box=read_dicpco2_pkgdiag(atm_box,data_dic,grid,path)
                
            except (AttributeError, IndexError, FileNotFoundError): # i.e. file does not exist or diagnostic is absent
                try:
                    atm_box=read_dicpco2_txtfile(atm_box,data_dic,path)
                
                except (AttributeError, IndexError, FileNotFoundError): # i.e. file does not exist or diagnostic is absent
                    atm_box=read_dicpco2_pickups(atm_box,data_parms,path)
                    
    except KeyError: # DIC_int1 is absent in data.dic so default value (278) is used
        atm_box=read_dicpco2_default(atm_box)

    return atm_box

def read_dicpco2_default(atm_box):
    # use the default pkg/dic value of 278 ppm
    atm_box['pco2']=(atm_box.iter*0+278e-6)           
    atm_box['molc']=(atm_box.iter*0+278e-6*1.77e20)
    
    return atm_box

def read_dicpco2_fixedvl(atm_box,data_dic):
    try:
        dicpco2=data_dic['dic_pco2']
    except KeyError: # i.e. param does not exist
        dicpco2=278e-6
        
    atm_box['pco2']=(atm_box.iter*0+dicpco2)           
    atm_box['molc']=(atm_box.iter*0+dicpco2*1.77e20)
    
    return atm_box

def read_dicpco2_prescri(atm_box,data_dic,path):
    # read from a file and linearly interpolate between file entries
    # (note:  dic_int2 = number entries to read
    #         dic_int3 = start timestep,
    #         dic_int4 = timestep between file entries)
    try:
        dicpco2=pd.read_csv(path+'co2atmos.dat',header=None,names=('C'),squeeze=True).to_xarray().rename(index='iter')
    except FileNotFoundError:
        dicpco2=pd.read_csv(path+'../input/co2atmos.dat',header=None,names=('C'),squeeze=True).to_xarray().rename(index='iter')
                   
    atmos_iter=np.arange(data_dic['dic_int3'],data_dic['dic_int3']+data_dic['dic_int4']*data_dic['dic_int2'],data_dic['dic_int4'])
    dicpco2['iter']=np.int32(atmos_iter)
    
    # Linearly interpolate values from the file onto the average timesteps in the output
    atm_box['pco2']=dicpco2.interp(iter=atm_box.iter,method='linear').reset_coords(drop=True)        
    atm_box['molc']=dicpco2.interp(iter=atm_box.iter,method='linear').reset_coords(drop=True)*1.77e20 
    
    return atm_box

def read_dicpco2_pkgdiag(atm_box,data_dic,grid,path):
    # Using the interactive atmospheric box which produces pickup files and may have diagnostics
    dic_surfdiag=open_ncfile(path+"dic_surfDiag.*.glob.nc",chunking={'T': 2},strange_axes={'Zmd000001':'ZC','Zd000001':'ZL'},grid=grid)
    try:
        # Taking the mean is fine because the atm box assigns the same value to each point (atm is well mixed)
        atm_box['pco2']=xr.DataArray(dic_surfdiag.DICATCO2.mean(('XC','YC')),coords=[atm_box.T])
    except AttributeError: # i.e. diagnostic is absent
        atm_box['molc']=xr.DataArray(dic_surfdiag.DICATCAR.mean(('XC','YC')),coords=[atm_box.T])
        atm_box['pco2']=xr.DataArray(dic_surfdiag.DICATCAR.mean(('XC','YC'))/1.77e20,coords=[atm_box.T])
    try:                
        atm_box['molc']=xr.DataArray(dic_surfdiag.DICATCAR.mean(('XC','YC')),coords=[atm_box.T])
    except AttributeError: # i.e. diagnostic is absent
        atm_box['pco2']=xr.DataArray(dic_surfdiag.DICATCO2.mean(('XC','YC')),coords=[atm_box.T])
        atm_box['molc']=xr.DataArray(dic_surfdiag.DICATCO2.mean(('XC','YC'))*1.77e20,coords=[atm_box.T])
    
    return atm_box

def read_dicpco2_txtfile(atm_box,data_dic,path):
    # Using the interactive atmospheric box which produces pickup files and may have diagnostics
    atm_file = pd.read_csv(gb.glob(path+"dic_atmos.*.txt")[0], delimiter="\s+",skiprows=0,header=None,
                            names=["iter","molc","pco2"],index_col='iter').to_xarray()
    atm_box=atm_box.set_coords('iter').swap_dims({'T':'iter'}).merge(atm_file,join='exact')\
                                      .swap_dims({'iter':'T'}).reset_coords('iter')

    return atm_box

def read_dicpco2_pickups(atm_box,data_parms,path):
    # Using the interactive atmospheric box which produces pickup files and may have diagnostics
    # No diagnostics, so have to load the pickups and interpolate linearly to the averageing time
    
    # Load the initial pickup at niter0 and append the others to it
    try: 
        dicpco2=xr.DataArray(open_bnfile(path+'pickup_dic_co2atm.'+'{0:010d}'
                        .format(np.int32(data_parms['niter0']))+'.data',sizearr=(1,2),prec='>f8').squeeze()) \
                        .to_dataset('dim_0').rename({0:'molc',1:'pco2'})
    except FileNotFoundError:
        dicpco2=xr.DataArray(open_bnfile(path+'../input/pickup_dic_co2atm.'+'{0:010d}'
                        .format(np.int32(data_parms['niter0']))+'.data',sizearr=(1,2),prec='>f8').squeeze()) \
                        .to_dataset('dim_0').rename({0:'molc',1:'pco2'})
    
    #Add the iteration axis
    dicpco2['iter']=np.int32(data_parms['niter0'])
    dicpco2=dicpco2.expand_dims('iter').set_coords('iter')
    
    for it in atm_box.iter.values:
        try:
            fname=path+'../run/pickup_dic_co2atm.'+'{0:010d}'.format(it)+'.data'
            tmp=xr.DataArray(open_bnfile(fname,sizearr=(1,2),prec='>f8').squeeze()) \
                    .to_dataset('dim_0').rename({0:'molc',1:'pco2'})
        except FileNotFoundError:
            fname=path+'../build/pickup_dic_co2atm.'+'{0:010d}'.format(it)+'.data'
            tmp=xr.DataArray(open_bnfile(fname,sizearr=(1,2),prec='>f8').squeeze()) \
                    .to_dataset('dim_0').rename({0:'molc',1:'pco2'})
        #Add the iteration axis
        tmp['iter']=np.int32(it)
        tmp=tmp.expand_dims('iter').set_coords('iter')            
        dicpco2=xr.concat((dicpco2,tmp),dim='iter')
        
    # The pickup files are snapshots of the pCO2, we want the "mean" between the snapshots
    atm_box['pco2']=dicpco2.pco2.rolling(iter=2,center=False).mean() \
        .dropna('iter').rename('pco2').reset_coords(drop=True).drop('iter').rename({'iter':'T'})
    atm_box['molc']=dicpco2.molc.rolling(iter=2,center=False).mean() \
        .dropna('iter').rename('molc').reset_coords(drop=True).drop('iter').rename({'iter':'T'})  
    
    return atm_box
                    
def get_macro_reference(fname,Rnp=16):
    from geopy.distance import geodesic as ge

# set macronutrient reference for the cost function from WOA13 annual climatology            
        
    woa=xr.open_dataset(fname,decode_times=False).squeeze('time').drop('time').rename({'lat':'YC','lon':'XC','depth':'ZC'}).transpose('XC','YC','ZC','nbounds') 
    woa=woa.assign_coords(XC=(woa.XC % 360)).roll(XC=(woa.dims['XC' ]//2),roll_coords=True).assign_coords(ZC=-woa.ZC)
    
    # Get axes - these are cell centres
    woa_lonc,woa_latc=np.meshgrid(woa.XC.values,woa.YC.values)
    #woa_dep=woa.ZC.values
    
    #woa_latg=np.unique(woa.lat_bnds)
    #woa_long=np.unique(woa.lon_bnds % 360)
    
    # Use geopy's geodesic to calulate dx and dy...note it's lat then lon input
    woa_dy=np.zeros((woa.dims['XC'],woa.dims['YC']))
    for jj in range(woa.dims['YC']):
        woa_dy[:,jj]=ge((woa.lat_bnds[jj][0],woa.XC.mean()),(woa.lat_bnds[jj][1],woa.XC.mean())).m
    woa['dy']=xr.DataArray(woa_dy,coords=[woa.XC, woa.YC], dims=['XC', 'YC'])
    
    woa_dx=np.zeros((woa.dims['XC'],woa.dims['YC']))
    for jj in range(woa.dims['YC']):
        woa_dx[:,jj]=ge((woa_latc[jj,0],woa.lon_bnds[0][0]),(woa_latc[jj,0],woa.lon_bnds[0][1])).m
    woa['dx']=xr.DataArray(woa_dx,coords=[woa.XC,woa.YC], dims=['XC','YC'])
    
    # Calulate dz (in metres)
    woa['dz']=xr.DataArray(np.diff(np.unique(woa.depth_bnds)),coords=[woa.ZC], dims=['ZC'])
    
    # Calculate volume
    woa['vol']=woa.dx * woa.dy * woa.dz
    
    # Get the phosphate data
    if Rnp==16:
        woa_nut=woa.n_an
    else:
        woa_nut=woa.p_an
   
    # Close the netcdf file
    woa.close()
    
    # Get basin masks
    woa['mask']=xr.DataArray(~np.isnan(woa_nut),coords=[woa.XC,woa.YC,woa.ZC], dims=['XC','YC','ZC'])
    #woa_atlantic_mask, woa_pacific_mask, woa_indian_mask, woa_so_mask, woa_arctic_mask = utils.oceanmasks(woa_lonc.T,woa_latc.T,woa.mask)
    
    return woa_nut

def get_micro_reference(fname):
# set Fe and L reference for the cost function from  GEOTRACES IDP 2017 
    import netCDF4  as nc
    import numpy.ma as nm
    
    idp  = nc.Dataset(fname, mode='r')
        
    # Variables of interest
    vars= {
    'Cruise':'metavar1', # Cruise
    'Press' :'var1',     # Pressure
    'Depth' :'var2',     # Depth (m)
    'Bottle':'var4',     # Bottle number
    'Bottle2':'var5',    # BODC Bottle number?
    'Firing':'var6',     # Firing Sequence
    'Theta' :'var7',     # CTDTEMP (°C) 
    'Salt'  :'var8',     # CTDSAL
    'OXY'   :'var20',    # Oxygen concentration (umol/kg)
    'OQC'   :'var20_QC', # OxygenQuality control flags
    'PO4'   :'var21',    # Phosphate (umol/kg)
    'PQC'   :'var21_QC', # Phosphate Quality control flags
    'SIT'   :'var23',    # Silicate (umol/kg)
    'SIQC'  :'var23_QC', # Silicate Quality control flags
    'NO3'   :'var24',    # Nitrate (umol/kg)
    'NQC'   :'var24_QC', # Nitrate Quality control flags
    'ALK'   :'var30',    # ALK (umol/kg)
    'AQC'   :'var30_QC', # ALK Quality control flags
    'DIC'   :'var31',    # DIC (umol/kg)
    'CQC'   :'var31_QC', # DIC Quality control flags
    'FeT'   :'var73',    # Fe (nmol/kg)
    'FQC'   :'var73_QC', # Fe Quality control flags
    'L1Fe'  :'var231',   # L1-Fe Ligand (nmol/kg)
    'L1QC'  :'var231_QC',# L1-Fe Quality control flags
    'L2Fe'  :'var233',   # L2-Fe Ligand (nmol/kg)
    'L2QC'  :'var233_QC',# L2-Fe Quality control flags
    }
    
    # size of arrays
    nsamp =idp.dimensions['N_SAMPLES' ].size
    nstat =idp.dimensions['N_STATIONS'].size
    #nchar =idp.dimensions['STRING6'].size
    
    # load variables
    idp_lon = np.transpose([idp.variables['longitude'][:] for _ in range(nsamp)])
    idp_lon = np.where(idp_lon>180, idp_lon-360, idp_lon)
    idp_lon_mod=idp_lon.copy()
    idp_lon_mod[idp_lon_mod<0]=idp_lon_mod[idp_lon_mod<0]+360
    
    
    idp_lat = np.transpose([idp.variables['latitude' ][:] for _ in range(nsamp)])
    idp_dep = -idp.variables[vars['Depth']][:]
    #idp_bot = idp.variables[vars['Bottle']][:] # get the bottle numbers for depth masking
    #umol= idp.variables[vars['PO4']].units
    #nmol=idp.variables[vars['FeT']].units
    
    # Use for later interpolation
    idpx=xr.DataArray(idp_lon_mod,dims=('N_STATIONS','N_SAMPLES'))
    idpy=xr.DataArray(idp_lat,dims=('N_STATIONS','N_SAMPLES'))
    idpz=xr.DataArray(idp_dep,dims=('N_STATIONS','N_SAMPLES'))
    
    #critdepth=np.zeros((nstat,nsamp)) # Going to ignore data points within 1000m of the bottom
    #for ii in range(nstat):
    #    if np.max(idp_dep[ii,:]) > 0.75:
    #        critdepth[ii,:]=np.max(idp_dep[ii,:])-1
    #    else:
    #        critdepth[ii,:]=np.max(idp_dep[ii,:])
            
    # Quality control flags are:
    # 1 Good:  Passed documented required QC tests
    # 2 Not evaluated, not available or unknown: Used for data when no QC test performed or the information on quality is not available
    # 3 Questionable/suspect: Failed non-critical documented metric or subjective test(s)
    # 4 Bad: Failed critical documented QC test(s) or as assigned by the data provider
    # 9 Missing data: Used as place holder when data are missing
    
    fqc = np.zeros((nstat,nsamp))
    tmp = idp.variables[vars['FQC']][:]
    for ii in range(nstat):
        for jj in range(nsamp):
            fqc[ii,jj]=np.double(tmp.data[ii,jj].tostring().decode("utf-8"))
    #idp_fe = nm.masked_where(np.logical_or(fqc>2,idp_dep>=critdepth),idp.variables[vars['FeT']][:])
    idp_fe = nm.masked_where(fqc>2,idp.variables[vars['FeT']][:])
    fref   = xr.DataArray(idp_fe,dims=('N_STATIONS','N_SAMPLES'))
    
    l1qc = np.zeros((nstat,nsamp))
    tmp = idp.variables[vars['L1QC']][:]
    for ii in range(nstat):
        for jj in range(nsamp):
            l1qc[ii,jj]=np.double(tmp.data[ii,jj].tostring().decode("utf-8"))
    #idp_l1 = nm.masked_where(np.logical_or(l1qc>2,idp_dep>=critdepth),idp.variables[vars['L1Fe']][:])
    idp_l1 = nm.masked_where(l1qc>2,idp.variables[vars['L1Fe']][:])
    idpxl1 = xr.DataArray(idp_l1,dims=('N_STATIONS','N_SAMPLES'))        
    
    l2qc = np.zeros((nstat,nsamp))
    tmp = idp.variables[vars['L2QC']][:]
    for ii in range(nstat):
        for jj in range(nsamp):
            l2qc[ii,jj]=np.double(tmp.data[ii,jj].tostring().decode("utf-8"))
    #idp_l2 = nm.masked_where(np.logical_or(l2qc>2,idp_dep>=critdepth),idp.variables[vars['L2Fe']][:])
    idp_l2 = nm.masked_where(l2qc>2,idp.variables[vars['L2Fe']][:])
    idpxl2 = xr.DataArray(idp_l2,dims=('N_STATIONS','N_SAMPLES'))
    
    # Add L1 nd L2 for total and sort out common mask
    #idp_lt = nm.masked_where(np.logical_and(idp_l1.mask,idp_l2.mask),idp_l1+idp_l2)
    lref   = idpxl1+idpxl2
    
    # close the file
    idp.close()
    
    # Get basin masks
    #idp_mask=np.ones(np.shape(idp_lon))
    #idp_atlantic_mask, idp_pacific_mask, idp_indian_mask, idp_so_mask, idp_arctic_mask = utils.oceanmasks(idp_lon_mod,idp_lat,idp_mask)
    
    return idpx, idpy, idpz, fref, lref

def calc_cost(modin,ref,stdev,iters=1):  
    if issubclass(type(modin), xr.core.dataarray.DataArray) or issubclass(type(ref), xr.core.dataarray.DataArray):
    # use the xarray-based methods  
        sumdims=[]      
        for ax in modin.dims:
            if ax.lower().find('t')==-1:
                sumdims.append(ax)
            
        cost=(np.power(modin-ref,2)/np.power(stdev,2)).sum(sumdims)
    else: # Use the old way using masked arrays or ndarrays
        if np.ndim(modin)<=1:
            iters=1
        else:
            iters=np.max(np.shape(modin))
        
        # number of boxes 
        #nbox=np.max(np.shape(ref))
        
        # Initial, quick n dirty, cost function, root-sum-squared anomalies
        #cost=np.sqrt(np.sum(np.power(modin.transpose()-np.tile(ref,(tlen,1)),2),axis=1)) # sum the rows
        
        # More advanced cost function where root-squared-anomalies are normalized by standard deviation of observations
        #   and then accumulated and divided by the number of boxes, after Radach and Moll (2006)
        #cost=np.sum(np.sqrt(np.power(modin.transpose()-np.tile(ref,(tlen,1)),2))/np.tile(stdev,(tlen,1)),axis=1)/nbox # sum the rows   
        
        # Something more like Omta et al., 2017
        cost=np.sum(np.power(modin.transpose()-np.tile(ref,(iters,1)),2)/np.tile(np.power(stdev,2),(iters,1)),axis=1)
    return cost
    
def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
  
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def wmean(arrin,vol_or_area,mask):
    import re
    
    regex = re.compile('X')
    xdim=tuple([string for string in arrin.dims if re.match(regex, string)])[0]

    regex = re.compile('Y')
    ydim=tuple([string for string in arrin.dims if re.match(regex, string)])[0]

    if (mask.ndim==2 and vol_or_area.ndim==2):
        # do area weighting
        wm=(arrin*vol_or_area*mask).sum([xdim,ydim])/(vol_or_area*mask).sum([xdim,ydim])
    elif (mask.ndim>=3 and vol_or_area.ndim>=3):
        # do volume weighting
        regex = re.compile('Z')
        zdim=tuple([string for string in arrin.dims if re.match(regex, string)])[0]
        wm=(arrin*vol_or_area*mask).sum([xdim,ydim,zdim])/(vol_or_area*mask).sum([xdim,ydim,zdim])
    return wm

def wsum(arrin,vol_or_area,mask):
    import re
    
    regex = re.compile('X')
    xdim=tuple([string for string in arrin.dims if re.match(regex, string)])[0]

    regex = re.compile('Y')
    ydim=tuple([string for string in arrin.dims if re.match(regex, string)])[0]

    if (mask.ndim==2 and vol_or_area.ndim==2):
        # do area integration
        wm=(arrin*vol_or_area*mask).sum([xdim,ydim])
    elif (mask.ndim>=3 and vol_or_area.ndim>=3):
        # do volume integration
        regex = re.compile('Z')
        zdim=tuple([string for string in arrin.dims if re.match(regex, string)])[0]
        wm=(arrin*vol_or_area*mask).sum([xdim,ydim,zdim])
    return wm

def saturated_oxygen(t,s,gsw=False):
    """Compute oxygen saturation from temperature and salinity
    Oxygen saturation value is the volume of oxygen gas absorbed from humidity-saturated
    air at a total pressure of one atmosphere, per unit volume of the liquid at the temperature
    of measurement (ml/l)
    """
    if gsw:
        import gsw
        gsw_o2sat=xr.apply_ufunc(gsw.O2sol_SP_pt,s,t,
                             dask='parallelized',
                             output_dtypes=[float])
        return gsw_o2sat*1.0245e-3 # Convert umol/kg to mol/m3
    else:
        oA0=  2.00907
        oA1=  3.22014 
        oA2=  4.05010 
        oA3=  4.94457 
        oA4= -2.56847E-1 
        oA5=  3.88767 
        oB0= -6.24523E-3 
        oB1= -7.37614E-3 
        oB2= -1.03410E-2 
        oB3= -8.17083E-3 
        oC0= -4.88682E-7

        aTT = 298.15-t
        aTK = 273.15+t
        aTS = np.log(aTT/aTK)
        aTS2= aTS*aTS 
        aTS3= aTS2*aTS
        aTS4= aTS3*aTS 
        aTS5= aTS4*aTS

        ocnew= np.exp(oA0 + oA1*aTS + oA2*aTS2 + oA3*aTS3 + oA4*aTS4 + oA5*aTS5
                      + s*(oB0 + oB1*aTS + oB2*aTS2 + oB3*aTS3) + oC0*(s*s))
#Saturation concentration of dissolved O2"/units="mol/m3" 
        return ocnew/22391.6*1000.0

def _calc_co2sys_tc(s,t,pz,at,atmpco2,pt,sit):
    import co2sys
    
    co=co2sys.calc_co2_system(s, t, 
                              pres    = pz, 
                              TA      = at,
                              pCO2    = atmpco2,
                              PO4     = pt,
                              Si      = sit,
                              K1K2    = "Millero_1995", 
                              KBver   = "Uppstrom", 
                              KSver   = "Dickson",
                              KFver   = "Dickson",
                              pHScale = 1
                              )
    return co.TC

def _calc_co2sys_pco2(s,t,pz,at,tc,pt,sit):
    import co2sys
    
    co=co2sys.calc_co2_system(s, t, 
                              pres    = pz, 
                              TA      = at,
                              TC      = tc,
                              PO4     = pt,
                              Si      = sit,
                              K1K2    = "Millero_1995", 
                              KBver   = "Uppstrom", 
                              KSver   = "Dickson",
                              KFver   = "Dickson",
                              pHScale = 1
                              )
    return co.pCO2

def calc_pco2(dic,alk,po4,sit,theta,salt,pressure):
    """
    calc_pco2(dic,alk,po4,sit,theta,salt,pressure)
    
    Compute seawater pCO2 concentration from dic and alkalinity
    at local pressure, temperature, salinity and nutrient conc.
    """
    pco2 = xr.apply_ufunc(_calc_co2sys_pco2,
                            salt,
                            theta,
                            pressure,
                            alk,
                            dic,
                            po4,
                            sit,
                            dask='parallelized', 
                            output_dtypes=[float],
                            )
    return pco2

def calc_carbon(pco2,alk,po4,sit,theta,salt,pressure):
    """
    calc_carbon(pco2,alk,po4,sit,theta,salt,pressure)
    
    Compute DIC concentration from seawater pCO2 and alkalinity
    at local pressure, temperature, salinity and nutrient conc.
    
    For saturated carbon, use preformed alkalinity and nutrients, as
    well at a 3d field of atmospheric pCO2.
    """
    dic = xr.apply_ufunc(_calc_co2sys_tc,
                            salt,
                            theta,
                            pressure,
                            alk,
                            pco2,
                            po4,
                            sit,
                            dask='parallelized', 
                            output_dtypes=[float],
                            )
    return dic
