#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 19:09:27 2019

@author: jml1
"""

import mitgcm_tools
import pandas as pd
import datetime

now = datetime.datetime.now()

grid_data,xgrid=mitgcm_tools.loadgrid("grid.glob.nc")

# get the timestep information
tave=mitgcm_tools.open_ncfile("tave.*.glob.nc",chunking={'T': 1})
grid_data['T']=tave.coords["T"]
grid_data['Tyr']=tave.coords["T"]/(86400*360)
grid_data['iter']=tave.iter
tave.close()
grid_data=grid_data.chunk({'T':2})

try: 
    data_pkg=mitgcm_tools.getparm('data.pkg') 
except FileNotFoundError:
    data_pkg=mitgcm_tools.getparm('../input/data.pkg')       
    
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

#get_dicpco2 loads output from the DIC package relating to the atmospheric
#  pco2 boundary condition (constant, read in from a file, or dynamic).
atm_box=mitgcm_tools.get_dicpco2(data_parms,data_dic,grid_data)

# write out to file
atm_box.to_dataframe().to_csv('dic_atmos.'+now.strftime("%Y%m%d")+'.txt',columns=(('iter','molc','pco2')),sep="\t",index=False,header=False)