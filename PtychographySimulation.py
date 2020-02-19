#%pdb # for debugging
import NanoImagingPack as nip
import numpy as np
from NanoImagingPack import v5


ecoli = 'ecoli_mini.png'
beads = 'beads_mini.png'
lena = 'lena.tif'
file = ecoli

obj = nip.extract(nip.readim(file),[128,128])
if 0:    
    obj = np.squeeze(obj)
    obj = np.mean(obj, axis=0)
nip.view(obj)

'''
define Illumination
'''
def make_pupil(obj_shape,ftradius):
    szn = np.array(obj_shape)
    myscales = ftradius / szn
    # Fourier pattern for pinhole with radius ftradius
    asf = nip.jinc(szn, myscales)  
    # normalize
    asf = asf / asf.midVal() * np.pi * ftradius ** 2 * nip.gaussian(asf.shape,sigma = 5) 
    # normlized to one in the disk
    pupil = np.real(nip.ft(asf)) / np.sqrt(np.prod(obj_shape))  
    return pupil

'''
create the illumination scan
'''
Sx = 4; Sy = 4
scan = nip.MakeScan(sz, [Sy, Sx])

pupil = make_pupil(obj.shape,ftradius=5.0)
illu = np.real(np.square(nip.ift2d(nip.ft2d(scan) * pupil)))

'''
define rough guess for support
'''
pupil2 = make_pupil(obj.shape,ftradius=7.0)
support = np.real(nip.abssqr(nip.ift2d(nip.ft2d(scan) * pupil2)))

nip.view(illu)
nip.view(support)
'''
simulate diffraction in object
'''
pimg = nip.abssqr(nip.ft2d(illu * obj))
MaxPhotons = 1000
nimg = nip.noise.poisson(pimg,NPhot=MaxPhotons)

#block_area=nip.rr(obj.shape)>5
#block_area = nip.extract(block_area,obj.shape,centerpos=(block_area.shape[0]//2 + 4,block_area.shape[1]//2 + 4),PadValue=1)
#block_area
#nimg *=block_area
#nimg[:,64,64]=0.1 #Blocking center pixel (:,64,64)
'''
simulate shifting of diffraction pattern over CCD
'''
dx=0
dy=0
nimg = nip.extract(nimg,obj.shape,centerpos=(obj.shape[0]//2 + dy,obj.shape[1]//2 + dx),PadValue=0)

'''
Applying the algorithm
'''

Niter = 50
AbsAmp = np.sqrt(nimg)
estimate = np.real(np.mean(nip.ift2d(AbsAmp),axis=0))
#myv = v5(estimate)
evolution = None

for n in range(Niter):
    #plt.clf()
    predicted = nip.ft2d(estimate * support)
    replaced = AbsAmp * np.exp(1j*np.angle(predicted))
    estimate = np.mean(nip.ift2d(replaced), axis=0)
    estimate = np.real(estimate)
    estimate[estimate < 0] = 0
    #myv.ReplaceData(estimate, e=0, title=n)
    if evolution is None:
        evolution = np.expand_dims(estimate, axis=0)
    else:
        evolution = np.concatenate((evolution,np.expand_dims(estimate,axis=0)))
    print('Iteration '+str(n))
nip.view(evolution)

