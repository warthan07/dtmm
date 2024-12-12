"""Cholester lens example.
"""

import dtmm
import numpy as np

dtmm.conf.set_verbose(2)
dtmm.conf.set_betamax(0.9)

#: pixel size in nm
PIXELSIZE = 500

THICKNESS = 8 #um

RADIUS = 20 #um

PITCH = 460/1.50/PIXELSIZE #pitch in pixelsize units
#RADIUS = RADIUS * 1000/P


NO = 1.5
NE = 1.65


#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 321,128,128
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,9)
#: create some experimental data (stack) left-handed cholesteric
optical_data = [dtmm.cholesteric_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 20, pitch = 7, no = 1.5, ne = 1.65, nhost = 1.5)] #approx 50*7*1.5 nm bragg reflection


Z = np.arange(0,THICKNESS, THICKNESS/NLAYERS)*1000/PIXELSIZE #z coordinate in pixel units
LAYER_THICKNESS = THICKNESS/NLAYERS *1000 / PIXELSIZE # in pixelsize units

RADIUS_PIXEL = RADIUS * 1000 / PIXELSIZE 
THICKNESS_PIXEL = THICKNESS * 1000 / PIXELSIZE 

zz,yy,xx = np.meshgrid(Z,np.linspace(-HEIGHT/2,HEIGHT/2,HEIGHT),np.linspace(-WIDTH/2,WIDTH/2,WIDTH),indexing = "ij")

zz = zz + RADIUS_PIXEL - THICKNESS_PIXEL

r = (zz**2+yy**2+xx**2)**0.5

mask = r > RADIUS_PIXEL

d = np.ones(NLAYERS) * LAYER_THICKNESS

angle = 2*np.pi*Z/PITCH 

eps = np.ones((NLAYERS, HEIGHT, WIDTH, 3))
eps[...,0] = NO**2
eps[...,1] = NO**2
eps[...,2] = NE**2

for i in range(NLAYERS):
    eps[i,...,2][mask[i]] = NO**2

ang = np.zeros((NLAYERS, HEIGHT, WIDTH, 3))
ang[...,1] = np.pi/2
ang[...,2] = -angle[...,None,None]


optical_data = [(d,eps,ang)]

#: create right-handed polarized input light
beta = 0.0 #make it off-axis 
#window = dtmm.aperture((HEIGHT, WIDTH),0.9,0.)
window = None

#jones = dtmm.jonesvec((1,1j)) 
jones = None

focus= 20 #this will focus field diaphragm in the middle of the stack
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, jones = jones, beta = beta,
                       pixelsize = PIXELSIZE, n = 1.5, focus = focus, window = window) 

#: transfer input light through stackt
field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = beta, phi = 0,
# %%
                                     diffraction = 1, method = "4x4",  smooth = 0.03,

                                     reflection = 2,nin = 1.5, nout = 1.5,norm = 2, npass = 75)

#: visualize output field
viewer1 = dtmm.field_viewer(field_data_out, mode = "t",n = 1.5, intensity = 0.5, focus = -20) 
fig1,ax1 = viewer1.plot()
ax1.set_title("Transmitted field")

#: residual back propagating field is close to zero
viewer2 = dtmm.field_viewer(field_data_out, mode = "r",n = 1.5)
fig2,ax2 = viewer2.plot()
ax2.set_title("Residual field")

viewer3 = dtmm.field_viewer(field_data_in, mode = "r", n = 1.5, polarization_mode = "mode", polarizer = "LCP", analyzer = "LCP")
fig3,ax3 = viewer3.plot()
ax3.set_title("Reflected field")

import matplotlib.pyplot as plt
plt.show()
