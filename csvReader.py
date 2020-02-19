#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:06:24 2020

@author: r2d2
"""


import numpy as np
import pandas as pd
import NanoImagingPack as nip
from phase_retrieval import fienup_phase_retrieval

'''
with open('Nervenzelle.ars', newline = '') as NervenData:                                                                                          
    data_reader = csv.reader(NervenData, delimiter='\t')
    for row in data_reader:
        print(row)
'''
cells='../ARS_Data/Nervenzelle.ars'
structure = '../ARS_Data/Structure.ars'
ecoli='../Ecoli_Data/ecoli_single_shot.ars'
ecoli_hdr = '../Ecoli_Data/ecoli_hdr.ars'

file = ecoli_hdr

ARS = np.array(pd.read_csv(file, 
                         sep='\t', header=None, 
                         skiprows=6,nrows=1040,usecols = [i for i in range(1040)],
                         dtype=np.float64))
Polar = np.array(pd.read_csv(file, 
                         sep='\t', header=None, 
                         skiprows=1048,nrows=1040,usecols = [i for i in range(1040)],
                         dtype=np.float64))
Phi = np.array(pd.read_csv(file, 
                         sep='\t', header=None, 
                         skiprows=2090,nrows=1040,usecols = [i for i in range(1040)],
                         dtype=np.float64))


nip.view(ARS)
region = ARS[467:577,477:587]
region = nip.extract(region, ARS.shape)
c = nip.ift(nip.ft(ARS)*np.conjugate(nip.ft(region)[::-1,::-1]))
nip.view(c)
loc_corr = np.where(c == c.max())
print(loc_corr)
loc_max = np.where(ARS == ARS.max())
print(loc_max)

loc = loc_max
'''

region = nip.extract(region,ARS.shape)
c = nip.ift(nip.ft(ARS)*np.conjugate(nip.ft(region)[::-1,::-1]))
c = np.real(nip.ift(nip.ft(ARS)*np.conjugate(nip.ft(ARS))))
nip.view(c)

'''
#mag=nip.abssqr(nip.extract(ARS,(800,800),(loc[0][0],loc[1][0])))
mag=nip.abssqr(nip.extract(ARS,(800,800)))
mag = nip.abssqr(ARS)

mask = nip.rr(mag.shape)<200

#result, evol = fienup_phase_retrieval(mag,mask=mask, steps=500,
#                                verbose=True)
#nip.view(evol)

