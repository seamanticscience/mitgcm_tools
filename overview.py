# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 16:53:08 2019

@author: jml1

mitgcm_proc_ini - initial, first pass, processing on a model run for an 
overview of the physics and biogeochemistry 
"""

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from cycler import cycler
import getpass     as gp
import glob        as gb
import matplotlib  as mp
import netCDF4     as nc
import numpy       as np
import numpy.ma    as nm
import scipy       as sp
import xarray      as xr
import xgcm
import mitgcm_tools

# Import mitgcm_tools
#from importlib.machinery import SourceFileLoader
#mitgcm_tools = SourceFileLoader("mitgcm_tools",'/Users/'+gp.getuser()+'/Dropbox_Work/Applications/Python/mitgcm_tools/mitgcm_tools.py').load_module()

mp.rcParams['xtick.labelsize'] = 14
mp.rcParams['ytick.labelsize'] = 14 
mp.rcParams.update({'font.size': 14})
mp.rc('axes.formatter', useoffset=False) 

#%% Load grid data and xgcm grid instance
grid_data,xgrid=mitgcm_tools.loadgrid("grid.glob.nc")

# get the timestep information
tave=mitgcm_tools.open_ncfile("tave.*.glob.nc",chunking={'T': 1})
grid_data['T']=tave.coords["T"]
grid_data['Tyr']=tave.coords["T"]/(86400*360)
grid_data['iter']=tave.iter
tave.close()
grid_data=grid_data.chunk({'T':2})

try: 
    data_parms=mitgcm_tools.getparm('data') 
except FileNotFoundError:
    try: 
        data_parms=mitgcm_tools.getparm('data.orig')
    except FileNotFoundError:
        try: 
            data_parms=mitgcm_tools.getparm('../input/data.orig')
        except FileNotFoundError:
            data_parms=mitgcm_tools.getparm('../input/data')     

try: 
    grid_data['deltaT']=data_parms['deltat']
except KeyError: # i.e. param does not exist
    grid_data['deltaT']=data_parms['deltatclock']
    
try: 
    data_pkg=mitgcm_tools.getparm('data.pkg') 
except FileNotFoundError:
    data_pkg=mitgcm_tools.getparm('../input/data.pkg')        
if data_pkg['usegmredi']:
    try: 
        data_gmredi=mitgcm_tools.getparm('data.gmredi') 
    except FileNotFoundError:
        data_gmredi=mitgcm_tools.getparm('../input/data.gmredi')
if data_pkg['useptracers']:
    try: 
        data_ptracers=mitgcm_tools.getparm('data.ptracers') 
    except FileNotFoundError:
        data_ptracers=mitgcm_tools.getparm('../input/data.ptracers')
if data_pkg['usegchem']:
    try: 
        data_gchem=mitgcm_tools.getparm('data.gchem') 
    except FileNotFoundError:
        data_gchem=mitgcm_tools.getparm('../input/data.gchem')
if data_gchem['usedic']:
    try: 
        data_dic=mitgcm_tools.getparm('data.dic') 
    except FileNotFoundError:
        data_dic=mitgcm_tools.getparm('../input/data.dic')
if data_pkg['usediagnostics']: # This is not that great
    try: 
        data_diags=mitgcm_tools.getparm('data.diagnostics',usef90nml=False) 
    except FileNotFoundError:
        data_diags=mitgcm_tools.getparm('../input/data.diagnostics',usef90nml=False)

#%% Calculate transports and streamfunctions (including GM Transports if used)
d2rad = 0.017453 

try: 
    ocediag=mitgcm_tools.open_ncfile("oceDiag.*.glob.nc",chunking={'T': 1},strange_axes={'Zmd000015':'ZC','Zld000015':'ZL'})
    uvel=ocediag.UVEL
    vvel=ocediag.VVEL   
    wvel=ocediag.WVEL
    ocediag.close()
except (AttributeError, FileNotFoundError) as e: # i.e. file does not exist or diagnostic is absent
    print(e+' file does not exist or diagnostic is absent. Using alternative values.')
    tave=mitgcm_tools.open_ncfile("tave.*.glob.nc",chunking={'T': 1})
    uvel=tave.uVeltave
    vvel=tave.vVeltave
    wvel=tave.wVeltave
    tave.close()
    
gmdiag=mitgcm_tools.open_ncfile("gmDiag.*.glob.nc",chunking={'T': 1},strange_axes={'Zmd000015':'ZC','Zld000015':'ZL'})

try: 
    gmuvel=gmdiag.GM_U_EDD
    gmvvel=gmdiag.GM_V_EDD  
    gmwvel=gmdiag.GM_W_EDD
except AttributeError: # i.e. diagnostic does not exist
    try:    
        if data_gmredi['GM_AdvForm']:
            gmuvel=(gmdiag.GM_PsiX.differentiate('ZL').interp(ZL=vvel.ZC)*-1).chunk({'T':2})
            gmvvel=(gmdiag.GM_PsiY.differentiate('ZL').interp(ZL=vvel.ZC)*-1).chunk({'T':2})
        else:
            gmuvel=gmdiag.GM_PsiX.diff('ZL')/(2*grid_data.drF)
            gmvvel=gmdiag.GM_PsiY.diff('ZL')/(2*grid_data.drF)
    except AttributeError: # i.e. diagnostic does not exist
        if data_gmredi['GM_AdvForm']:
            gmuvel=gmdiag.GM_Kwx.diff('ZL')/grid_data.drF
            gmvvel=gmdiag.GM_Kwy.diff('ZL')/grid_data.drF
        else:
            gmuvel=gmdiag.GM_Kwx.diff('ZL')/(2*grid_data.drF)
            gmvvel=gmdiag.GM_Kwy.diff('ZL')/(2*grid_data.drF)
        
try:
    ures=gmdiag.GM_U_RES
    vres=gmdiag.GM_V_RES  
#    wres=gmdiag.GM_W_RES
except AttributeError: # i.e. diagnostic does not exist
    ures=uvel+gmuvel
    vres=vvel+gmvvel
#    wres=wvel+gmwvel

utrans=ures*(grid_data.HFacW*grid_data.dyG*grid_data.drF)
vtrans=vres*(grid_data.HFacS*grid_data.dxG*grid_data.drF)
#wtrans=wres*(grid_data.HFacC*grid_data.rA)

# Net Transport through Drake Passage
if grid_data.XG.max()>180:
    dp_lon=298
else:
    dp_lon=-62
    
TDP=(utrans*grid_data.umask_so).sel(XG=dp_lon,method='nearest').sum(['YC','ZC'])

#for it in range(tave.dims['T']):
#    div_uv[]= (utrans.diff('XG') + vtrans.diff('YG'))
# Meridional Eulerian SF: calculate first intergral and then second integral (partial summation) over levels
meul =  (  vvel*np.cos(grid_data.coords['YG']*d2rad)*grid_data.HFacS*grid_data.dxG*grid_data.drF).sum('XC').cumsum('ZC')
medd =  (gmvvel*np.cos(grid_data.coords['YG']*d2rad)*grid_data.HFacS*grid_data.dxG*grid_data.drF).sum('XC').cumsum('ZC')
mres =  (  vres*np.cos(grid_data.coords['YG']*d2rad)*grid_data.HFacS*grid_data.dxG*grid_data.drF).sum('XC').cumsum('ZC')

# Zonal Eulerian SF: calculate first intergral and then second integral (partial summation) over levels
zeul = -(  uvel*grid_data.HFacW*grid_data.dyG*grid_data.drF).sum('YC').cumsum('ZC')
zedd = -(gmuvel*grid_data.HFacW*grid_data.dyG*grid_data.drF).sum('YC').cumsum('ZC')
zres = -(  ures*grid_data.HFacW*grid_data.dyG*grid_data.drF).sum('YC').cumsum('ZC')

# Barotropic (depth-integrated) SF
baro_mask=grid_data.umask.interp(YC=grid_data.coords['YG'],method='linear')

ubaro=(((ures * grid_data.HFacW * grid_data.drF * grid_data.dyG).sum(dim='ZC').sel(YC=slice(None, None, -1)) .cumsum('YC')).sel(YC=slice(None, None, -1)).interp(YC=grid_data.coords['YG'],method='linear'))*baro_mask.isel(ZC=0)
vbaro=(  vres * np.cos(grid_data.coords['YG']*d2rad) * grid_data.HFacS * grid_data.drF * grid_data.dxG).sum(dim='ZC').cumsum('XC').interp(XC=grid_data.coords['XG'],method='linear')*baro_mask.isel(ZC=0)
baro =( ubaro+vbaro)

# Overturning strengths with time 
somoc=(mres.sel(YG=slice(-90,-30),ZC=slice(-50,-2000)).max(['YG','ZC']))
abmoc=(mres.sel(ZC=slice(-1000,-5000)).min(['YG','ZC']))
amoc =(mres.sel(YG=slice(0,90),ZC=slice(-50,-2000)).max(['YG','ZC']))

#%% Calculate Mean properties
ocediag=mitgcm_tools.open_ncfile("oceDiag.*.glob.nc",chunking={'T': 1},strange_axes={'Zmd000015':'ZC','Zld000015':'ZL'})

try: 
    theta=ocediag.THETA
    salt =ocediag.SALT
    ocediag.close()
except AttributeError: # i.e. diagnostic does not exist
    tave=mitgcm_tools.open_ncfile("tave.*.glob.nc",chunking={'T': 1})
    theta=tave.Ttave
    salt =tave.Stave
    tave.close()

# Should add EXF and other options to this 
surfdiag=mitgcm_tools.open_ncfile("surfDiag.*.glob.nc",chunking={'T': 1},strange_axes={'Zmd000001':'ZC','Zd000001':'ZL'})
try: 
    theta_surf_flux =surfdiag.TFLUX
    salt_surf_flux  =surfdiag.SFLUX*31104000*1e6
    theta_relax_flux=surfdiag.TRELAX
    salt_relax_flux =surfdiag.SRELAX*31104000*1e6
    theta_forc_flux =surfdiag.surForcT
    salt_forc_flux  =surfdiag.surForcS*31104000*1e6
    theta_qnet_flux =surfdiag.oceQnet
    salt_ocefw_flux =surfdiag.oceFWflx*31104000*35*1e6 # per year not per s
    theta_freez_flux=surfdiag.oceFreez
    salt_oces_flux  =surfdiag.oceSflux*31104000*1e6 # per year not per s
    
    surfdiag.close()
except AttributeError: # i.e. diagnostic does not exist
    tave=mitgcm_tools.open_ncfile("tave.*.glob.nc",chunking={'T': 1})
    theta_surf_flux=tave.tFluxtave
    salt_surf_flux =tave.sFluxtave
    
    # Set to NAN
    theta_relax_flux=tave.tFluxtave*np.nan
    salt_relax_flux =tave.sFluxtave*np.nan
    theta_forc_flux =tave.tFluxtave*np.nan
    salt_forc_flux  =tave.sFluxtave*np.nan
    theta_qnet_flux =tave.tFluxtave*np.nan
    salt_ocefw_flux =tave.sFluxtave*np.nan
    theta_freez_flux=tave.tFluxtave*np.nan
    salt_oces_flux  =tave.sFluxtave*np.nan
    tave.close()

#%%         
ptr_tave=mitgcm_tools.open_ncfile("ptr_tave.*.glob.nc",chunking={'T': 1},strange_axes={'Zmd000015':'ZC','Zld000015':'ZL'})

dicdiag=mitgcm_tools.open_ncfile("dicDiag.*.glob.nc",chunking={'T': 1},strange_axes={'Zmd000015':'ZC','Zld000015':'ZL'})

dic_surfdiag=mitgcm_tools.open_ncfile("dic_surfDiag.*.glob.nc",chunking={'T': 1},strange_axes={'Zmd000001':'ZC','Zd000001':'ZL'})

dic_tave=mitgcm_tools.open_ncfile("dic_tave.*.glob.nc",chunking={'T': 1})

#%% Process the Atmoshperic CO2 values and write out as a text file
if data_gchem['usedic']:
    atm_box=mitgcm_tools.get_dicpco2(data_parms,data_dic,grid_data)
    
#%% Set up plot axes
       
diag_drift1, ([meanstax,meangtax],[heatflax,fwflax]) = plt.subplots(figsize=(12, 8),ncols=2,nrows=2)

# Area-weighted Surface T
a1,=meanstax.plot(grid_data.Tyr,mitgcm_tools.wmean(theta.isel(ZC=0),grid_data.rA,grid_data.cmask.isel(ZC=0)),color='red',label='global T')
a2,=meanstax.plot(grid_data.Tyr,mitgcm_tools.wmean(theta.isel(ZC=0),grid_data.rA,grid_data.cmask_nh.isel(ZC=0)),color='red',linestyle='--',label='NH T')
a3,=meanstax.plot(grid_data.Tyr,mitgcm_tools.wmean(theta.isel(ZC=0),grid_data.rA,grid_data.cmask_sh.isel(ZC=0)),color='red',linestyle=':', label='SH T')
# second axes that shares the same x-axis
meanssax = meanstax.twinx()  
# Area-weighted Surface S
a4,=meanssax.plot(grid_data.Tyr,mitgcm_tools.wmean(salt.isel(ZC=0),grid_data.rA,grid_data.cmask.isel(ZC=0)),color='C0',label='global S')
a5,=meanssax.plot(grid_data.Tyr,mitgcm_tools.wmean(salt.isel(ZC=0),grid_data.rA,grid_data.cmask_nh.isel(ZC=0)),color='C0',linestyle='--',label='NH S')
a6,=meanssax.plot(grid_data.Tyr,mitgcm_tools.wmean(salt.isel(ZC=0),grid_data.rA,grid_data.cmask_sh.isel(ZC=0)),color='C0',linestyle=':', label='SH S')

#meantsax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#meantsax.set_xlim(left=-0.5,right=4.5)
#meantsax.set_ylim(bottom=0,top=50)
meanstax.set_title('Mean Surface Theta and Salt')

plt.legend(loc='upper center',handles = [a1,a2,a3,a4,a5,a6],ncol=3,columnspacing=1,
           bbox_to_anchor=(0.5, -0.1),fancybox=False, shadow=False)

# Volume-weighted full depth T
a1,=meangtax.plot(grid_data.Tyr,mitgcm_tools.wmean(theta,grid_data.cvol,grid_data.cmask),color='red',label='global T')
a2,=meangtax.plot(grid_data.Tyr,mitgcm_tools.wmean(theta,grid_data.cvol,grid_data.cmask_nh),color='red',linestyle='--',label='NH T')
a3,=meangtax.plot(grid_data.Tyr,mitgcm_tools.wmean(theta,grid_data.cvol,grid_data.cmask_sh),color='red',linestyle=':', label='SH T')
# second axes that shares the same x-axis
meangsax = meangtax.twinx()  
# Volume-weighted full depth T
a4,=meangsax.plot(grid_data.Tyr,mitgcm_tools.wmean(salt,grid_data.cvol,grid_data.cmask),color='C0',label='global S')
a5,=meangsax.plot(grid_data.Tyr,mitgcm_tools.wmean(salt,grid_data.cvol,grid_data.cmask_nh),color='C0',linestyle='--',label='NH S')
a6,=meangsax.plot(grid_data.Tyr,mitgcm_tools.wmean(salt,grid_data.cvol,grid_data.cmask_sh),color='C0',linestyle=':', label='SH S')
meangsax.set_title('Mean Global Theta and Salt')

plt.legend(loc='upper center',handles = [a1,a2,a3,a4,a5,a6],ncol=3,columnspacing=1,
           bbox_to_anchor=(0.5, -0.1),fancybox=False, shadow=False)

heatflax.plot(grid_data.Tyr,mitgcm_tools.wmean(theta_surf_flux,grid_data.rA,grid_data.cmask.isel(ZC=0)),label='tflux')
heatflax.plot(grid_data.Tyr,mitgcm_tools.wmean(theta_relax_flux,grid_data.rA,grid_data.cmask.isel(ZC=0)),label='trelax')
heatflax.plot(grid_data.Tyr,mitgcm_tools.wmean(theta_forc_flux,grid_data.rA,grid_data.cmask.isel(ZC=0)),label='tforc')
heatflax.plot(grid_data.Tyr,mitgcm_tools.wmean(theta_qnet_flux,grid_data.rA,grid_data.cmask.isel(ZC=0)),label='qnet')
heatflax.plot(grid_data.Tyr,mitgcm_tools.wmean(theta_freez_flux,grid_data.rA,grid_data.cmask.isel(ZC=0)),label='freez')
heatflax.set_ylim(bottom=-np.max(np.abs(heatflax.set_ylim()).round()),
                  top   = np.max(np.abs(heatflax.set_ylim()).round()))
heatflax.set_title('Surface Heat Forcing [w m$^{-1}$]')
heatflax.legend(loc='upper center',ncol=3,columnspacing=1,
           bbox_to_anchor=(0.5, -0.1),fancybox=False, shadow=False)

fwflax.plot(grid_data.Tyr,mitgcm_tools.wmean(salt_surf_flux,grid_data.rA,grid_data.cmask.isel(ZC=0)),label='sflux')
fwflax.plot(grid_data.Tyr,mitgcm_tools.wmean(salt_relax_flux,grid_data.rA,grid_data.cmask.isel(ZC=0)),label='srelax')
fwflax.plot(grid_data.Tyr,mitgcm_tools.wmean(salt_forc_flux,grid_data.rA,grid_data.cmask.isel(ZC=0)),label='sforc')
fwflax.plot(grid_data.Tyr,mitgcm_tools.wmean(salt_ocefw_flux,grid_data.rA,grid_data.cmask.isel(ZC=0)),label='socefw')
fwflax.plot(grid_data.Tyr,mitgcm_tools.wmean(salt_oces_flux,grid_data.rA,grid_data.cmask.isel(ZC=0)),label='soces')           
fwflax.set_ylim(bottom=-np.max(np.abs(fwflax.set_ylim()).round(decimals=-1)),
                top   = np.max(np.abs(fwflax.set_ylim()).round(decimals=-1)))
fwflax.set_title('Surface FW/salt forcing [mg m$^{-2}$ y$^{-1}$]')
fwflax.legend(loc='upper center',ncol=3,columnspacing=1,
           bbox_to_anchor=(0.5, -0.1),fancybox=False, shadow=False)

# Can adjust the subplot size
plt.subplots_adjust(wspace=0.4,hspace=0.6) 
plt.show()          
              
#%% Plot tracers concentrations
#diag_drift2, ([meancaax,meanfpax],[totcpax,ppax]) = plt.subplots(figsize=(12, 8),ncols=2,nrows=2)
if data_pkg['useptracers']:
    diag_drift2, ([meancaax, meanpdax],[meano2ax, meanfeax],[meanptrax1, meanptrax2]) = plt.subplots(figsize=(12, 16),ncols=2,nrows=3)
    #diag_drift2, ([meancaax, meanpdax],[meano2ax, meanfeax]) = plt.subplots(figsize=(12, 8),ncols=2,nrows=2)
        
    for ptr in np.arange(data_ptracers['ptracers_numinuse'],dtype=int):
        name=data_ptracers['ptracers_names'][ptr]
        unit=data_ptracers['ptracers_units'][ptr]
        var=ptr_tave[data_ptracers['ptracers_names'][ptr]]
        
        if (name[0:3]=='dic' or name[0:3]=='alk' or name[0:4]=='cpre' or name[0:4]=='apre'):
            # DIC and ALK variables on the same axis
            meancaax.plot(grid_data.Tyr,mitgcm_tools.wmean(var.isel(ZC=0),grid_data.rA,grid_data.cmask.isel(ZC=0)),label='surf '+name)
            meancaax.plot(grid_data.Tyr,mitgcm_tools.wmean(var,grid_data.cvol,grid_data.cmask),linestyle='--',label='global '+name)
            meancaax.legend(loc='upper center',ncol=2,columnspacing=1,
                bbox_to_anchor=(0.5, -0.1),fancybox=False, shadow=False)
            meancaax.set_ylabel('['+unit+']')
        elif (name[0:3]=='po4' or name[0:3]=='dop' or name[0:4]=='ppre'):
            meanpdax.plot(grid_data.Tyr,1e3*mitgcm_tools.wmean(var.isel(ZC=0),grid_data.rA,grid_data.cmask.isel(ZC=0)),label='surf '+name)
            meanpdax.plot(grid_data.Tyr,1e3*mitgcm_tools.wmean(var,grid_data.cvol,grid_data.cmask),linestyle='--',label='global '+name)
            meanpdax.legend(loc='upper center',ncol=2,columnspacing=1,
                bbox_to_anchor=(0.5, -0.1),fancybox=False, shadow=False)
            meanpdax.set_ylabel('[m'+unit+']')
        elif (name[0:3]=='o2' or name[0:3]=='do2' or name[0:4]=='opre'):
            meano2ax.plot(grid_data.Tyr,1e3*mitgcm_tools.wmean(var.isel(ZC=0),grid_data.rA,grid_data.cmask.isel(ZC=0)),label='surf '+name)
            meano2ax.plot(grid_data.Tyr,1e3*mitgcm_tools.wmean(var,grid_data.cvol,grid_data.cmask),linestyle='--',label='global '+name)
            meano2ax.legend(loc='upper center',ncol=2,columnspacing=1,
                bbox_to_anchor=(0.5, -0.1),fancybox=False, shadow=False)
            meano2ax.set_ylabel('[m'+unit+']')
        elif (name[0:3]=='fe' or name[0:4]=='fpre' or name[0:3]=='lig'):
            meanfeax.plot(grid_data.Tyr,1e6*mitgcm_tools.wmean(var.isel(ZC=0),grid_data.rA,grid_data.cmask.isel(ZC=0)),label='surf '+name)
            meanfeax.plot(grid_data.Tyr,1e6*mitgcm_tools.wmean(var,grid_data.cvol,grid_data.cmask),linestyle='--',label='global '+name)
            meanfeax.legend(loc='upper center',ncol=2,columnspacing=1,
                bbox_to_anchor=(0.5, -0.1),fancybox=False, shadow=False)
            meanfeax.set_ylabel('[u'+unit+']')
        else:
            # Then the rest of the ptracers
            mag=int(np.log10(np.max((var.isel(T=-1).isel(ZC=0).mean(['XC','YC']).compute(),1))))
            if ptr % 2 == 0: 
                # Plot even numbered ptracers on the left
                meanptrax1.plot(grid_data.Tyr,mitgcm_tools.wmean(var.isel(ZC=0),grid_data.rA,grid_data.cmask.isel(ZC=0))*(10**-mag),label=name+' ('+unit+')')
                meanptrax1.legend(loc='upper center',ncol=2,columnspacing=1,
                    bbox_to_anchor=(0.5, -0.1),fancybox=False, shadow=False)    
            else:
                # Plot odd numbered ptracers on the right
                meanptrax2.plot(grid_data.Tyr,mitgcm_tools.wmean(var,grid_data.cvol,grid_data.cmask)*(10**-mag),label=name)
                meanptrax2.legend(loc='upper center',ncol=2,columnspacing=1,
                    bbox_to_anchor=(0.5, -0.1),fancybox=False, shadow=False)
            
    # Can adjust the subplot size
    plt.subplots_adjust(wspace=0.25,hspace=0.6) 
    plt.show()
        
#%% Plot more bgc parameters
if data_gchem['usedic']:
    diag_drift3, ([prodax,meanpco2ax],[co2flax,atmco2ax]) = plt.subplots(figsize=(12, 8),ncols=2,nrows=2)

    prodax.plot(grid_data.Tyr,117*mitgcm_tools.wsum(dicdiag.DICBIOA,grid_data.cvol,grid_data.cmask)*360*86400*12e-15,color='green')
    prodax.set_title('Net global ocean production [GtC/yr]')
    co2flax.plot(grid_data.Tyr,mitgcm_tools.wsum(dic_surfdiag.DICTFLX*360*86400*12e-15,grid_data.rA,grid_data.cmask.isel(ZC=0)),color='red',label='Net CO2 Flux')
    co2flax.set_title('Net global ocean CO2 flux [GtC/yr]')
    a1,=atmco2ax.plot(grid_data.Tyr,atm_box.atm_pco2*1e6,color='red',label='ATM pCO2')
    a2,=atmco2ax.plot(grid_data.Tyr,mitgcm_tools.wmean(dic_surfdiag.DICPCO2*1e6,grid_data.rA,grid_data.cmask.isel(ZC=0)),label='Mean Ocean pCO2')
#    atmmolax = atmco2ax.twinx()  
#    a3,=atmmolax.plot(grid_data.Tyr,atm_box.atm_molc*12e-15,color='green',label='ATM molC')
#    plt.legend(loc='upper center',handles = [a1,a2,a3],ncol=2,columnspacing=1,
#           bbox_to_anchor=(0.5, -0.1),fancybox=False, shadow=False)
    # Can adjust the subplot size
    plt.subplots_adjust(wspace=0.25,hspace=0.4) 
    plt.show()
    
#%%
residfig, ([baro1ax,baro2ax],[meulax,mresax],[meddax,transax]) = plt.subplots(figsize=(12, 12),ncols=2,nrows=3)

cax1=baro1ax.contourf(grid_data.coords['XG'],grid_data.coords['YG'],ubaro.isel(T=-1)/1e6,cmap='RdBu_r',levels=(np.arange(-150,175,25)),extend='both')
for a in baro1ax.collections:
    a.set_edgecolor("face")
baro1ax.contour(grid_data.coords['XG'],grid_data.coords['YG'],ubaro.isel(T=-1)/1e6,levels=(np.arange(-150,175,25)),colors='black')
cbar1=residfig.colorbar(cax1,ax=baro1ax,ticks=np.arange(-150,200,50),extend='both')
cbar1.solids.set_edgecolor("face")
baro1ax.set_title('Barotropic Streamfunction\nuvel only [Sv]')
baro1ax.xaxis.set_ticks(np.arange(0, 420, 60))
baro1ax.yaxis.set_ticks(np.arange(-90, 120, 30))
baro1ax.set_facecolor('black')

cax2=baro2ax.contourf(grid_data.coords['XG'],grid_data.coords['YG'],baro.isel(T=-1)/1e6,cmap='RdBu_r',levels=(np.arange(-150,175,25)),extend='both')
for a in baro2ax.collections:
    a.set_edgecolor("face")
baro2ax.contour(grid_data.coords['XG'],grid_data.coords['YG'],baro.isel(T=-1)/1e6,levels=(np.arange(-150,175,25)),colors='black')
cbar2=residfig.colorbar(cax2,ax=baro2ax,ticks=np.arange(-150,200,50),extend='both')
cbar2.solids.set_edgecolor("face")
baro2ax.set_title('Barotropic Streamfunction\nuvel+vvel [Sv]')
baro2ax.xaxis.set_ticks(np.arange(0, 420, 60))
baro2ax.yaxis.set_ticks(np.arange(-90, 120, 30))
baro2ax.set_facecolor('black')

cax3=meulax.contourf(grid_data.coords['YG'],grid_data.coords['ZC']/1000,meul.isel(T=-1)*grid_data.vmask.mean('XC')/1e6,cmap='RdBu_r',levels=(np.arange(-50,55,5)),extend='both')
for a in meulax.collections:
    a.set_edgecolor("face")
meulax.contour(grid_data.coords['YG'],grid_data.coords['ZC']/1000,meul.isel(T=-1)*grid_data.vmask.mean('XC')/1e6,levels=(np.arange(-50,55,5)),colors='black')
cbar3=residfig.colorbar(cax3,ax=meulax,ticks=np.arange(-50,60,10),extend='both')
cbar3.solids.set_edgecolor("face")
meulax.set_title('Eulerian-mean overturning [Sv]')
meulax.xaxis.set_ticks(np.arange(-90, 120, 30))
meulax.yaxis.set_ticks(np.arange(-5, 1, 1))
meulax.set_facecolor('black')

cax4=mresax.contourf(grid_data.coords['YG'],grid_data.coords['ZC']/1000,mres.isel(T=-1)*grid_data.vmask.mean('XC')/1e6,cmap='RdBu_r',levels=(np.arange(-50,55,5)),extend='both')
for a in mresax.collections:
    a.set_edgecolor("face")
mresax.contour(grid_data.coords['YG'],grid_data.coords['ZC']/1000,mres.isel(T=-1)*grid_data.vmask.mean('XC')/1e6,levels=(np.arange(-50,55,5)),colors='black')
cbar4=residfig.colorbar(cax4,ax=mresax,ticks=np.arange(-50,60,10),extend='both')
cbar4.solids.set_edgecolor("face")
mresax.set_title('Residual mean overturning [Sv]')
mresax.xaxis.set_ticks(np.arange(-90, 120, 30))
mresax.yaxis.set_ticks(np.arange(-5, 1, 1))
mresax.set_facecolor('black')

cax5=meddax.contourf(grid_data.coords['YG'],grid_data.coords['ZC']/1000,medd.isel(T=-1)*grid_data.vmask.mean('XC')/1e6,cmap='RdBu_r',levels=(np.arange(-50,55,5)),extend='both')
for a in meddax.collections:
    a.set_edgecolor("face")
meddax.contour(grid_data.coords['YG'],grid_data.coords['ZC']/1000,medd.isel(T=-1)*grid_data.vmask.mean('XC')/1e6,levels=(np.arange(-50,55,5)),colors='black')
cbar5=residfig.colorbar(cax5,ax=meddax,ticks=np.arange(-50,60,10),extend='both')
cbar5.solids.set_edgecolor("face")
meddax.set_title('Eddy-induced overturning [Sv]')
meddax.xaxis.set_ticks(np.arange(-90, 120, 30))
meddax.yaxis.set_ticks(np.arange(-5, 1, 1))
meddax.set_facecolor('black')

# Transports
a1,=transax.plot(grid_data.Tyr,amoc/1e6,linestyle='-',label='AMOC')
a2,=transax.plot(grid_data.Tyr,somoc/1e6,linestyle='--',label='SOMOC')
a3,=transax.plot(grid_data.Tyr,abmoc/1e6,linestyle=':',label='AABW')
transax.set_ylim(bottom=-np.max(np.abs(transax.set_ylim()).round(decimals=-1)),
                 top   = np.max(np.abs(transax.set_ylim()).round(decimals=-1)))# second axes that shares the same x-axis
tdpax = transax.twinx()  
a4,=tdpax.plot(grid_data.Tyr,TDP/1e6,linestyle='--',label='TDP')

plt.legend(loc='upper center',handles = [a1,a2,a3,a4],ncol=4,columnspacing=1,
           bbox_to_anchor=(0.5, -0.1),fancybox=False, shadow=False)
#tdpax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#tdpax.set_xlim(left=-0.5,right=4.5)
#tdpax.set_ylim(bottom=0,top=50)
tdpax.set_title('Transports [Sv]')
tdpax.set_ylim(bottom=0,top=200)

# Can adjust the subplot size
plt.subplots_adjust(wspace=0.15,hspace=0.25) 
plt.show()

#%%
#hovmul, ([tempax,saltax],[dicax,phosax]) = plt.subplots(figsize=(12, 8),ncols=2,nrows=2)

