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
import os
import sys
import xgcm

def find_in_search_path(pathname, matchFunc=os.path.isfile):
    """
       Look through the python search path for pathname and return location if successful.
       
       matchFunc lets you set the function to use. Default is os.path.isfile, which is great if
          you hit the file directly on the path, but gb.glob is a good choice if you have a vague
          idea that your file is on the path but arent exactly sure where (dont forget to include 
          a wildcard in pathname).
    
    """ 
    for dirname in sys.path:
        candidate = os.path.join(dirname, pathname)
        if matchFunc(candidate):
            return candidate
    raise FileNotFoundError("Can't find file %s" % pathname)

# %% REGRIDDING, AXES, AND FILE LOADING ROUTINES
def _mfd_preprocess(self):
    # Internal function for preprocessing tile files for merging (need to get rid of Xp! and Yp1 overlaps)
    
    # Have to make assumptions about domain decomposition 
    npx = self.attrs['nPx']
    npy = self.attrs['nPy']
    try:
        tile= self.attrs['tile_number']
        
        # "right edge" of grid should not be modified along XG axis for these tiles
        dont_knock_xmax=np.arange(npx,npx*npy+1,npx)
    
        # "top row" of grid should not be modified along YG axis for these tiles
        dont_knock_ymax=np.arange(npx*(npy-1)+1,npx*npy+1,1)
    except KeyError:
        # This will error out if it's a global file, but you don't want to preprocess this anyway
        tile=1
        dont_knock_xmax=[1]
        dont_knock_ymax=[1]
    
    if tile == "global":
        # This will error out if it's a global file, but you don't want to preprocess this anyway
        tile=1
        dont_knock_xmax=[1]
        dont_knock_ymax=[1]
        
    # Pkg/diagnostics hack for ZMD, ZUD, ZLD, and ZL dimensions that have no coordinate values
    diagdict=dict()
    if any(s.startswith('Zmd') for s in list(self.dims)):
        dim='Zmd{:06d}'.format(len(self.diag_levels.values))
        diagdict[dim]=self.diag_levels.values
    if any(s.startswith('Zld') for s in list(self.dims)):
        dim='Zld{:06d}'.format(len(self.diag_levels.values))
        diagdict[dim]=self.diag_levels.values
    if any(s.startswith('Zud') for s in list(self.dims)):
        dim='Zud{:06d}'.format(len(self.diag_levels.values))
        diagdict[dim]=self.diag_levels.values
    if any(s.startswith('Zd') for s in list(self.dims)):
        dim='Zd{:06d}'.format(len(self.diag_levels.values))
        diagdict[dim]=self.diag_levels.values 
    
    if diagdict:
        self=self.assign_coords(diagdict)
    
    # Define shorter Xp1 and Yp1 axes, only for interior tiles
    rendict=dict()
    if 'Xp1' in self.dims and tile not in dont_knock_xmax:
        newXp1=self['Xp1'].isel(Xp1=slice(0,-1)).values
        self['newXp1']=xr.DataArray(newXp1, coords=[newXp1],dims=['newXp1'])
        rendict["newXp1"]="Xp1"

    if 'Yp1' in self.dims and tile not in dont_knock_ymax:
        newYp1=self['Yp1'].isel(Yp1=slice(0,-1)).values
        self['newYp1']=xr.DataArray(newYp1, coords=[newYp1],dims=['newYp1'])
        rendict["newYp1"]="Yp1"

    # Process the variables on to shorter axes
    for var in self.variables:
        if var  not in ['newXp1', 'newYp1']:
            origdim=self[var].dims
            seldict=dict()
            coodict=dict()
            
            # Prepopulate coordinate dict with original dimensions
            for dim in origdim:
                coodict[dim]=self[dim].values
            # Remove horizontal dimensions to be replaced
            for dim in list(set(['X', 'Xp1', 'Y', 'Yp1']) & set(list(coodict.keys()))):
                del coodict[dim]
            
            if 'Yp1' in origdim and tile not in dont_knock_ymax:
                seldict["Yp1"]=slice(0,-1)
                coodict["newYp1"]=newYp1
            elif 'Yp1' in origdim and tile in dont_knock_ymax:
                coodict["Yp1"]=self['Yp1'].values
            elif 'Y' in origdim:
                coodict["Y"]=self['Y'].values
            
            if 'Xp1' in origdim and tile not in dont_knock_xmax:
                seldict["Xp1"]=slice(0,-1)
                coodict["newXp1"]=newXp1
            elif 'Xp1' in origdim and tile in dont_knock_xmax:
                coodict["Xp1"]=self['Xp1'].values
            elif 'X' in origdim:
                coodict["X"]=self['X'].values
            
            if seldict:
                self[var]=xr.DataArray(self[var].isel(seldict),
                                       coords=list(coodict.values()),
                                       dims=list(coodict.keys()))
        
    if rendict:
        return self.reset_coords(names=list(rendict.values()),drop=True).rename(rendict).chunk()
    else:
        return self.chunk()

def open_ncfile(file_pattern,doconform_axes=True,chunking=None,strange_axes=dict(),grid=[]):
    """
        Read in data from a netcdf file (file_pattern, we CAN handle tile files!)) using xarray, 
          which can be chunked via dask by setting chunking to a dictionary of chunks 
          (e.g. {'T':2,'X':10,'Y':10,'Z':2}).
        For compatability with xgcm, the axes may need to be conformed to certain specifications. 
          set conform_axes=False to override this. We can handle conversions between many axis names, but
          if there is a particularly difficult set (thanks, pkg/diagnostics) set the conversion within the
          "strange_axes" dictionary strange_axes={'Xnp':'XN','Ynp':'YN'}
    """
    if doconform_axes:
        if not grid:
            data=conform_axes(xr.open_mfdataset(gb.glob(file_pattern),preprocess=_mfd_preprocess,
                               decode_times=False,data_vars='minimal').chunk(chunking),strange_ax=strange_axes)
        else:
            data=conform_axes(xr.open_mfdataset(gb.glob(file_pattern),preprocess=_mfd_preprocess,
                               decode_times=False,data_vars='minimal').chunk(chunking),strange_ax=strange_axes,grd=grid)
    else:
        data=xr.open_mfdataset(gb.glob(file_pattern),preprocess=_mfd_preprocess,
                               decode_times=False,data_vars='minimal').chunk(chunking)
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
    
def loadgrid(fname='grid.glob.nc',basin_masks=True,doconform_axes=True,chunking=None):
    """ loadgrid(fname,sizearr,prec) reads a netcdf grid file and returns it as a
        xarray, with a few additional items.
        
        fname is the file name, could be a file pattern (we CAN handle tile files!)
    """     
    grd=xr.open_mfdataset(fname,preprocess=_mfd_preprocess,data_vars='minimal').chunk(chunking)
    
    if "T" in grd.dims:
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
        try:
            atlantic_mask, pacific_mask, indian_mask, so_mask, arctic_mask = oceanmasks(grd.lonc.transpose('X','Y').values,grd.latc.transpose('X','Y').values,grd.cmask.transpose('X','Y','Z').values)
                
            grd['cmask_atlantic'] = xr.DataArray(atlantic_mask,\
                                coords=[grd.X.values, grd.Y.values, grd.Z.values], dims=['X', 'Y', 'Z'])
            grd['cmask_pacific']  = xr.DataArray(pacific_mask ,\
                                coords=[grd.X.values, grd.Y.values, grd.Z.values], dims=['X', 'Y', 'Z'])
            grd['cmask_indian']   = xr.DataArray(indian_mask  ,\
                                coords=[grd.X.values, grd.Y.values, grd.Z.values], dims=['X', 'Y', 'Z'])
            grd['cmask_so']       = xr.DataArray(so_mask      ,\
                                coords=[grd.X.values, grd.Y.values, grd.Z.values], dims=['X', 'Y', 'Z'])
            grd['cmask_arctic']   = xr.DataArray(arctic_mask  ,\
                                coords=[grd.X.values, grd.Y.values, grd.Z.values], dims=['X', 'Y', 'Z'])
            grd['cmask_nh']       = grd.cmask.where(grd.coords['Y']>0)
            grd['cmask_sh']       = grd.cmask.where(grd.coords['Y']<=0)
            
            atlantic_mask, pacific_mask, indian_mask, so_mask, arctic_mask = oceanmasks(grd.lonu.transpose('Xp1','Y').values,grd.latu.transpose('Xp1','Y').values,grd.umask.transpose('Xp1','Y','Z').values)
            
            grd['umask_atlantic'] = xr.DataArray(atlantic_mask,\
                                coords=[grd.Xp1.values, grd.Y.values, grd.Z.values], dims=['Xp1', 'Y', 'Z'])
            grd['umask_pacific']  = xr.DataArray(pacific_mask ,\
                                coords=[grd.Xp1.values, grd.Y.values, grd.Z.values], dims=['Xp1', 'Y', 'Z'])
            grd['umask_indian']   = xr.DataArray(indian_mask  ,\
                                coords=[grd.Xp1.values, grd.Y.values, grd.Z.values], dims=['Xp1', 'Y', 'Z'])
            grd['umask_so']       = xr.DataArray(so_mask      ,\
                                coords=[grd.Xp1.values, grd.Y.values, grd.Z.values], dims=['Xp1', 'Y', 'Z'])
            grd['umask_arctic']   = xr.DataArray(arctic_mask  ,\
                                coords=[grd.Xp1.values, grd.Y.values, grd.Z.values], dims=['Xp1', 'Y', 'Z'])
            grd['umask_nh']       = grd.umask.where(grd.coords['Y']>0)
            grd['umask_sh']       = grd.umask.where(grd.coords['Y']<=0)
             
            atlantic_mask, pacific_mask, indian_mask, so_mask, arctic_mask = oceanmasks(grd.lonv.transpose('X','Yp1').values,grd.latv.transpose('X','Yp1').values,grd.vmask.transpose('X','Yp1','Z').values)
            
            grd['vmask_atlantic'] = xr.DataArray(atlantic_mask,\
                                coords=[grd.X.values, grd.Yp1.values, grd.Z.values], dims=['X', 'Yp1', 'Z'])
            grd['vmask_pacific']  = xr.DataArray(pacific_mask ,\
                                coords=[grd.X.values, grd.Yp1.values, grd.Z.values], dims=['X', 'Yp1', 'Z'])
            grd['vmask_indian']   = xr.DataArray(indian_mask  ,\
                                coords=[grd.X.values, grd.Yp1.values, grd.Z.values], dims=['X', 'Yp1', 'Z'])
            grd['vmask_so']       = xr.DataArray(so_mask      ,\
                                coords=[grd.X.values, grd.Yp1.values, grd.Z.values], dims=['X', 'Yp1', 'Z'])
            grd['vmask_arctic']   = xr.DataArray(arctic_mask  ,\
                                coords=[grd.X.values, grd.Yp1.values, grd.Z.values], dims=['X', 'Yp1', 'Z'])
            grd['vmask_nh']       = grd.vmask.where(grd.coords['Yp1']>0)
            grd['vmask_sh']       = grd.vmask.where(grd.coords['Yp1']<=0)
        except (FileNotFoundError,IOError,OSError,TypeError):
            print("Trouble with basin masking using oceanmasks")
        finally:    
            grd.close()
    
    if doconform_axes:
        # Attempt to conform axes to conventions
        # These variable conflict with future axis names
        grd=grd.drop(['XC','YC','XG','YG'])
    
        grd=conform_axes(grd)

        # generate XGCM grid, with metrics for grid aware calculations
        # Have to make sure the metrics are properly masked
        # issue for area, but not volume... 
        grd['rA' ]=grd.rA * grd.HFacC.isel(ZC=0)
        grd['rAs']=grd.rAs* grd.HFacS.isel(ZC=0)
        grd['rAw']=grd.rAw* grd.HFacW.isel(ZC=0)
        # This is dodgy, but not sure what else to do...
        grd['rAz']=grd.rAz* (grd.HFacW.interp(coords={'YC':grd['YG']},method='nearest')* \
                             grd.HFacS.interp(coords={'XC':grd['XG']},method='nearest')).isel(ZC=0)
    
        metrics = {
            ('X',): ['dxC', 'dxG'], # X distances
            ('Y',): ['dyC', 'dyG'], # Y distances
            ('Z',): ['dzW', 'dzS', 'dzC'], # Z distances
            ('X', 'Y'): ['rA', 'rAz', 'rAs', 'rAw'] # Areas
            }
        xgrd = xgcm.Grid(grd,periodic=['X','Y'],metrics=metrics)
    else:
        xgrd=[]
    
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

    # Check that all the dimensions have coordinate values (or want to override the coord values with strange_ax).
    missing_ax=dict()
    for ax in dsin.dims:
        if ax in list(strange_ax.values()) or ax not in dsin.coords:
            # Should be able to import these from grid_data
            if not grd:
                # Be cheeky and reload the grid, which should have the axes needed
                grd=xr.open_dataset(gb.glob("grid*nc")[0]).squeeze('T')
                grd.close()
                grd=grd.drop(['XC','YC','XG','YG'])
                grd=conform_axes(grd)
            if dsin.dims[ax] > 1:    
                missing_ax[ax]=grd.coords[ax]
            else:
                dsin=dsin.squeeze(ax).drop(ax)
    
    if missing_ax:        
        dsin=dsin.assign_coords(missing_ax)
        print("Coordinates added or altered for axes: "+','.join(list(missing_ax.keys())))    
    
    if 'diag_levels' in dsin.variables:
        dsin=dsin.drop_vars('diag_levels')

    return dsin

def oceanmasks(xc,yc,modelmask,woamask_file=''): 
    from scipy.interpolate import griddata
    from urllib.error      import HTTPError

    nzdim=0
    # Find if input dimensions are 3d or 2d
    if np.ndim(modelmask)>2:
        nzdim=np.size(modelmask,2)
        if np.ndim(xc)>2:
            xc=xc[:,:,0]
        if np.ndim(yc)>2:
            yc=yc[:,:,0]

    try:
        if gb.glob(woamask_file):
            # Use location of confirmed-existing user specified mask file
            locs=[woamask_file]
            found_by="user input location."
        else:
            # Try searching the python path for our maskfile
            locs=[gb.glob(find_in_search_path('*/woa13_basinmask_01.msk',matchFunc=gb.glob))[0]]
            found_by="searching python path."
    except FileNotFoundError:
        # Otherwise fall back on the web or hard coded locations
        locs=["https://data.nodc.noaa.gov/woa/WOA13/MASKS/basinmask_01.msk",
          "/Users/jml1/GitHub/Lauderdale_ligand_iron_microbe_feedback/woa13_basinmask_01.msk",
          "/Users/jml1/GitHub/Lauderdale_2016_GBC/woa13_basinmask_01.msk",
          "/Users/jml1/Dropbox_Work/Applications/MATLAB/mitgcm_toolbox/woa13_basinmask_01.msk"]
        found_by="hard coded location."
    finally:
        from urllib.error      import HTTPError
        # Initialize empty pandas data frame
        woamask=pd.DataFrame()
        # Cycle through location list to read the mask file
        for fname in locs:
            try:
                woamask = pd.read_csv(fname,header=1)
                print("read mask file at "+fname+" found by "+found_by)
                break
            except (HTTPError,FileNotFoundError,IOError,OSError):
                # If there's an error, just continue down the location list
                continue
        # test to see if the mask was actually successfully loaded nd raise an error if not
        if woamask.empty:
            raise FileNotFoundError("Can't find mask file on python search path, on the web, or at hard coded locations.")
        else:
            x = woamask.Longitude.values
            y = woamask.Latitude .values
            basinfile = woamask.Basin_0m.values
        
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
        atlantic_mask = np.tile(atlantic_mask[:,:,np.newaxis],(1,1,nzdim))*modelmask
        pacific_mask  = np.tile(pacific_mask [:,:,np.newaxis],(1,1,nzdim))*modelmask
        indian_mask   = np.tile(indian_mask  [:,:,np.newaxis],(1,1,nzdim))*modelmask
        so_mask       = np.tile(so_mask      [:,:,np.newaxis],(1,1,nzdim))*modelmask
        arctic_mask   = np.tile(arctic_mask  [:,:,np.newaxis],(1,1,nzdim))*modelmask
        
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
                    
def get_macro_reference(fname,Rnp=16,chunks={}):
    from geopy.distance import geodesic as ge

# set macronutrient reference for the cost function from WOA13 annual climatology            
        
    woa=xr.open_dataset(fname,decode_times=False,chunks=chunks).squeeze('time').drop('time').rename({'lat':'YC','lon':'XC','depth':'ZC'}).transpose('XC','YC','ZC','nbounds') 
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

def complexation(metal_tot, ligand_tot, beta):
        """
        Given a total metal concentration, ligand concentration, and beta value, this
        function will return the total concentration of free metal that is not bound to
        ligands (i.e. is free and prone to scavanging).
        Parameters
        ----------
        metal_tot : float 
            current total concentration of metal, value in mol per cubic meter 
        ligand_tot : float
            current total concentration of ligand, value in mol per cubic meter
        beta : float
            constant value, defines equilibrium position between free metal + ligand
            and the complexed form. Value in kg/mol
        Returns
        -------
        Float, current concentration of free metal.
        """
        term_1 = (metal_tot - 1/beta - ligand_tot)/2
        term_2 = ((beta*(ligand_tot - metal_tot + 1/beta)**2 + 4*metal_tot)/(4*beta))**(1/2)
    
        return term_1 + term_2

def read_geotraces_idp(fname,varsin=None,chunks={}):
    from contextlib import suppress
    
    # QC string on the variable name changed between IDP2017 and IDP2021
    if (fname.find('2017') != -1):
        # Quality control flags (2017) are:
        # 1 Good:  Passed documented required QC tests
        # 2 Not evaluated, not available or unknown: Used for data when no QC test performed or the information on quality is not available
        # 3 Questionable/suspect: Failed non-critical documented metric or subjective test(s)
        # 4 Bad: Failed critical documented QC test(s) or as assigned by the data provider
        # 9 Missing data: Used as place holder when data are missing   
        qcstr ='_QC'
        qcgood=2 
    elif (fname.find('2021') != -1):
        # Quality control flags (2021) are:
        # 48: no_quality_control
        # 49: good_value
        # 50: probably_good_value
        # 51: probably_bad_value
        # 52: bad_value
        # 53: changed_value
        # 54: value_below_detection
        # 55: value_in_excess
        # 56: interpolated_value
        # 57: missing_value
        # 65: value_phenomenon_uncertain
        # 66: nominal_value
        # 81: value_below_limit_of_quantification
        qcstr='_qc'
        qcgood=50
        
    # Read the netcdf file into xarray dataset
    tmp=xr.open_dataset(fname,chunks=chunks)

    if varsin is None:
        # We want to process the whole idp, which may take a few mins
        varlist = list(tmp.keys())
    else:
        # Need to add QC variables for processing
        varlist=[sub + '_QC' for sub in varsin] + varsin
    
    tmp['N_STATIONS']=np.arange(tmp.dims['N_STATIONS'])
    tmp['N_SAMPLES' ]=np.arange(tmp.dims['N_SAMPLES' ])
    
    long_name_dict=dict()
    drop_name_list=list()
    
    for var in tmp.keys():
        if (var in varlist) or (var.find('meta') == 0):
            #print(var)
            if "long_name" in tmp[var].attrs.keys():
                # Rename variables to their long names
                long_name_dict[var] = tmp[var].attrs['long_name'].lower()
                if (fname.find('2017') != -1): 
                    if (tmp[var].attrs['long_name'].lower() == 'ctdoxy'):
                        # Rename the CTD variables if they exist in the varlist
                        long_name_dict[list(long_name_dict.keys())[list(long_name_dict.values()).index('ctdoxy')]]='ctdoxy_d_conc_sensor'
                    elif (tmp[var].attrs['long_name'].lower() == 'ctdtmp'):
                        # Rename the CTD variables if they exist in the varlist
                        long_name_dict[list(long_name_dict.keys())[list(long_name_dict.values()).index('ctdtmp')]]='ctdtmp_t_value_sensor'
                    elif (tmp[var].attrs['long_name'].lower() == 'ctdsal'):
                        # Rename the CTD variables if they exist in the varlist
                        long_name_dict[list(long_name_dict.keys())[list(long_name_dict.values()).index('ctdsal')]]='ctdsal_d_conc_sensor'

            if (var.find(qcstr) != -1):
                # A quality control flag, that we eventuall want to drop
                drop_name_list.append(var)
                
                if tmp[var].dtype == 'O': 
                    # Convert "char" QC object into a float (IDP2017 only)
                    qc_flags=tmp[var].astype('float')
                else:
                    qc_flags=tmp[var]
        
                # Now apply the QC flag to the data variable
                tmp[var[:-3]] = tmp[var[:-3]].where(qc_flags<=qcgood)
                
                with suppress(KeyError):
                    # Also apply the QC flag to the STD variable
                    tmp[var[:-3]+'_STD'] = tmp[var[:-3]+'_STD'].where(qc_flags<=qcgood)
                 
            if (var.find('meta') == 0): 
                if (tmp[var].dtype == 'O' or str(tmp[var].dtype).startswith('|S')):
                    # Convert "char" metavar object into a string
                    tmp[var]=tmp[var].astype('str')
        else:
            drop_name_list.append(var)
       
    idp=tmp.rename(long_name_dict).drop(drop_name_list)
    
    # Expand dimensions of these variables
    idp['lon']=idp.longitude.broadcast_like(idp['depth'])
    idp['lat']=idp.latitude .broadcast_like(idp['depth'])
    
    # Get basin masks
    idp['mask']=xr.ones_like(idp['depth'])
    
    atlantic_mask, pacific_mask, indian_mask, so_mask, arctic_mask = oceanmasks(idp['lon'].values,idp['lat'].values,idp['mask'].values)
    
    idp['atlantic_mask'] = xr.DataArray(data=atlantic_mask, dims=["N_STATIONS", "N_SAMPLES"])
    idp['pacific_mask' ] = xr.DataArray(data=pacific_mask , dims=["N_STATIONS", "N_SAMPLES"])
    idp['indian_mask'  ] = xr.DataArray(data=indian_mask  , dims=["N_STATIONS", "N_SAMPLES"])
    idp['so_mask'      ] = xr.DataArray(data=so_mask      , dims=["N_STATIONS", "N_SAMPLES"])
    idp['arctic_mask'  ] = xr.DataArray(data=arctic_mask  , dims=["N_STATIONS", "N_SAMPLES"])
    return idp

def get_micro_reference(fname,chunks={}):
# set Fe and L reference for the cost function from  GEOTRACES IDP 2017 or 2021 
    try:
        df=pd.read_csv(fname.replace(".nc","_variables.txt"), delimiter='\t')
    
        # Variables of interest
        varlist=[df.loc[(df["Variable"].str.find(var)>=0)]['nc Variable'].tolist()[0] for var in 
                ["Latitude"               , # Latitude
                 "Longitude"              , # Longitude
                 "DEPTH"                  , # Depth (m)
                 "Fe_D_CONC_BOTTLE"       , # Fe (nmol/kg)
                 "L1Fe_D_CONC_BOTTLE"     , # L1-Fe Ligand (nmol/kg)
                 "L2Fe_D_CONC_BOTTLE"     , # L2-Fe Ligand (nmol/kg)
                 "L1Fe_D_LogK_BOTTLE"     , # L1-Fe stability coefficient (nmol/kg)
                 "L2Fe_D_LogK_BOTTLE"     , # L2-Fe stability coefficient (nmol/kg)
                 ]]
        if (fname.find('2021') != -1):
            # Ligand concentration if only one group of ligands was found (by the software)
            varlist.append(df.loc[(df["Variable"].str.find("LFe_D_CONC_BOTTLE")>=0)]['nc Variable'].tolist()[0])
            varlist.append(df.loc[(df["Variable"].str.find("LFe_D_LogK_BOTTLE")>=0)]['nc Variable'].tolist()[0])
    except FileNotFoundError:
        varlist=None
    
    # Given the file list (or no file list), load the IDP dataset
    idp=read_geotraces_idp(fname,varlist,chunks=chunks)
    
    #idp['depth']=-0.001*idp['depth'] # convert +ve metres to -ve km
        
    if (fname.find('2021') != -1):
        # Ligand concentration if only one group of ligands was found (by the software)
        # I'm adjusting the ligands again...some of these are v high (>10nmol/kg), but surely outliers
        idp['ltfe_d_conc_bottle'] = idp['l1fe_d_conc_bottle'].where(idp['l1fe_d_conc_bottle']<10.0).fillna(0.0) + \
                                    idp['l2fe_d_conc_bottle'].where(idp['l2fe_d_conc_bottle']<10.0).fillna(0.0) + \
                                    idp['lfe_d_conc_bottle' ].where(idp['lfe_d_conc_bottle' ]<10.0).fillna(0.0)
        idp['ltfe_d_conc_bottle'] = idp['ltfe_d_conc_bottle'].where(idp['ltfe_d_conc_bottle']>0.0)
    
        # Weighted mean of ligand stability coefficients
#        idp['ltfe_d_logk_bottle'] = (idp['l1fe_d_conc_bottle'].where(idp['l1fe_d_conc_bottle']<10.0).fillna(0.0)*idp['l1fe_d_logk_bottle'].fillna(0.0)+ \
#                                     idp['l2fe_d_conc_bottle'].where(idp['l2fe_d_conc_bottle']<10.0).fillna(0.0)*idp['l2fe_d_logk_bottle'].fillna(0.0)+ \
#                                     idp['lfe_d_conc_bottle' ].where(idp['lfe_d_conc_bottle' ]<10.0).fillna(0.0)*idp['lfe_d_logk_bottle' ].fillna(0.0))/\
#                                     idp['ltfe_d_conc_bottle']
    else:   
        # I'm adjusting the ligands again...some of these are v high (>10nmol/kg), but surely outliers
        idp['ltfe_d_conc_bottle'] = idp['l1fe_d_conc_bottle'].where(idp['l1fe_d_conc_bottle']<10.0).fillna(0.0) + \
                                    idp['l2fe_d_conc_bottle'].where(idp['l2fe_d_conc_bottle']<10.0).fillna(0.0)
    
#        idp['lfe_d_conc_bottle' ] = idp['ltfe_d_conc_bottle']*np.nan
        
        # Weighted mean of ligand stability coefficients
#        idp['ltfe_d_logk_bottle'] = (idp['l1fe_d_conc_bottle'].where(idp['l1fe_d_conc_bottle']<10.0).fillna(0.0)*idp['l1fe_d_logk_bottle'].fillna(0.0)+ \
#                                     idp['l2fe_d_conc_bottle'].where(idp['l2fe_d_conc_bottle']<10.0).fillna(0.0)*idp['l2fe_d_logk_bottle'].fillna(0.0))/\
#                                     idp['ltfe_d_conc_bottle']
        
    idp['ltfe_d_conc_bottle'] = idp['ltfe_d_conc_bottle'].where(idp['ltfe_d_conc_bottle']>0.0)    
#    idp['ltfe_d_logk_bottle'] = idp['ltfe_d_logk_bottle'].where(idp['ltfe_d_logk_bottle']>0.0)
    
    # I'm adjusting the iron again...some of these are either negative or v high (>10nmol/kg), but surely outliers
    idp['fe_d_conc_bottle'] = idp['fe_d_conc_bottle'].where(idp['fe_d_conc_bottle']>0.0).where(idp['fe_d_conc_bottle']<10.0)      

    fref = idp[['fe_d_conc_bottle',  'lat','lon','depth']].rename({'lat':'Latitude','lon':'Longitude','depth':'Depth'})
    lref = idp[['ltfe_d_conc_bottle','lat','lon','depth']].rename({'lat':'Latitude','lon':'Longitude','depth':'Depth'})

#    idp['fefree_d_conc_bottle']  = complexation(idp['fe_d_conc_bottle'], idp['ltfe_d_conc_bottle'], idp['ltfe_d_logk_bottle'])
#    idp['cufree_d_conc_bottle']  = complexation(idp['cu_d_conc_bottle'], idp['l1cu_d_conc_bottle'], idp['l1cu_d_logk_bottle'])
        
    return fref, lref

def calc_cost(modin,ref,stdev,iters=1,sumdims=['XC','YC','ZC']):  
    if issubclass(type(modin), xr.core.dataarray.DataArray) or issubclass(type(ref), xr.core.dataarray.DataArray):
    # use the xarray-based methods  
#        sumdims=[]      
#        for ax in modin.dims:
#            if ax.lower().find('t')==-1:
#                sumdims.append(ax)
            
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
