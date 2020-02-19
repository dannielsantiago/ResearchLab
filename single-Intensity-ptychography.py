import numpy as np
import NanoImagingPack as nip
import matplotlib.pyplot as plt
from phase_retrieval import fienup_phase_retrieval


ecoli = 'e-coli.png'
beads = 'beads_.png'
file = ecoli
obj = nip.readim(file)
obj = np.squeeze(obj)
obj = np.mean(obj, axis=0)
#obj = obj.max() - obj
nip.view(obj)

#illumination
sz = obj.shape
szn = np.array(sz)
ftradius = 5.0
myscales = ftradius / szn
asf = nip.jinc(szn, myscales)  # Fourier pattern for pinhole with radius ftradius
asf = asf / asf.midVal() * np.pi * ftradius ** 2 * nip.gaussian(asf.shape,sigma = 50) # normalize
psf = nip.abssqr(asf)
psf /= psf.max()
nip.view(psf)

'''
dx=-1
dy=-1
crop_shape=(200,200)
blocked_area = nip.rr(obj.shape)>1
blocked_area = nip.extract(blocked_area,crop_shape, centerpos=(blocked_area.shape[0]//2 + dy,blocked_area.shape[1]//2 + dx))
blocked_area = nip.extract(blocked_area,obj.shape,PadValue=1)
nip.view(blocked_area)
'''

pimg = nip.abssqr(nip.ft2d(obj*psf))
photons = 1000
nimg = nip.noise.poisson(pimg, NPhot=photons)

#pimg*=blocked_area
#pimg = nip.extract(np.abs(nip.ft2d(obj*mask)),crop_shape, centerpos=(obj.shape[0]//2 + dy,obj.shape[1]//2 + dx))
#mask = nip.extract(mask,pimg.shape)
support = nip.rr(obj.shape)<200

result, evol = fienup_phase_retrieval(nimg,mask=support, steps=500,
                                verbose=True)
nip.view(obj*psf)
nip.view(evol)
