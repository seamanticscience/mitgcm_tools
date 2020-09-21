# mitgcm_tools
Python toolbox for working with MITgcm output. Depends heavily on the excellent `xarray` and `xgcm` packages.

### open_ncfile(file_pattern,doconform_axes=True,chunking=None,strange_axes=dict(),grid=[]):
        Read in data from a netcdf file (file_pattern) using xarray, which can be chunked via dask by
          setting chunking to a dictionary of chunks (e.g. {'T':2,'X':10,'Y':10,'Z':2}).
        For compatability with xgcm, the axes may need to be conformed to certain specifications. 
          set conform_axes=False to override this. We can handle conversions between many axis names, but
          if there is a particularly difficult set (thanks, `pkg/diagnostics`) set the conversion within the
          "strange_axes" dictionary strange_axes={'Xnp':'XN','Ynp':'YN'}

### open_bnfile(fname,sizearr=(12,15,64,128),prec='>f4'):
    open_bnfile(fname,sizearr,prec) reads a binary file and returns it as a
        numpy array.
        
        fname is the file name,
        prec  is the precision and dtype argument ('>f4' works usually)
        sizearr is the anticipated size of the returned array
    
### loadgrid(fname='grid.glob.nc'):
    loadgrid(fname,sizearr,prec) reads a netcdf grid file and returns it as a
        xarray, with a few additional items.
        
        fname is the file name,
  
### getparm(path_to_namelist,usef90nml=True,flatten=True):
        Read in Namelist file to a dictionary as strings or floats
        Works best with package `f90nml`
        
### get_dicpco2(data_parms,data_dic,grid,path='./'):
    get_dicpco2 loads output from the DIC package relating to the atmospheric
       pco2 boundary condition (constant, read in from a file, or dynamic).
       It interogates a bunch of different sources (text file, pickup, etc).
    
### saturated_oxygen(t,s,gsw=False):
    Compute oxygen saturation from temperature and salinity; best with `gsw` package.
    Oxygen saturation value is the volume of oxygen gas absorbed from humidity-saturated
    air at a total pressure of one atmosphere (mol/m3)
    
### calc_pco2(dic,alk,po4,sit,theta,salt,pressure):  
    Compute seawater pCO2 concentration from dic and alkalinity
    at local pressure, temperature, salinity and nutrient conc.
    Written to work with my python fork of the `co2sys` package.
    
### calc_carbon(pco2,alk,po4,sit,theta,salt,pressure):
    Compute DIC concentration from seawater pCO2 and alkalinity
    at local pressure, temperature, salinity and nutrient conc.
    
    For saturated carbon, use preformed alkalinity and nutrients, as
    well at a 3d field of atmospheric pCO2.
 
    
