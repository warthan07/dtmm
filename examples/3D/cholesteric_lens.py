"""Reflections/transmission microscopy of a cholesteric lens

In this example, we build a thin cholesteric lens placed on a flat surface (glass)
and have no index matching on the top surface. We enter from air into lens (or glass)
and observe reflection and transmisson spectra.

Because of the additional reflections caused by the air-LC interface and the lensing
effects, the reflection spectra is quite different compared to an experiment 
performed in an index-matching medium.

For lensing to work, the pixelsize parameter needs to be sufficiently low.
If we use index-matching on the other hand, pixelsize may be larger.

We set up a plane wave at normal incidence and compute transmission and reflection
spectra using an iterative procedure. Note that the algorithm should always
converge to an exact solution, but the convergence rate increases exponentially 
with thicness of the cholesteric. We use 15 passes.

"""

#the usual imports
import dtmm
import numpy as np

# for verbose printitng
dtmm.conf.set_verbose(2)

# needs to be set high enough, must not exceed the minimum value of refractive index  
# that we use in our layered material. Since we enter from air with n=1, we must
# restrict beta < 1. Lowering beta may improve convergence, but it reduces resolution
# and accuracy of the solution.
dtmm.conf.set_betamax(0.9)

#START CONFIGURATION-----------------------------------------------

#: pixel size in nm
PIXELSIZE = 200

#: lens thickness, for thicker lenses, you must increase NPASS and NLAYERS
THICKNESS = 3 #um

# radius of the curvature of the lens
RADIUS = 20 #um

# cholesteric pitch, bandgap at arround 460 nm
PITCH = 460/1.50/PIXELSIZE #pitch in pixelsize units

# ordinary refractive index
NO = 1.5
# extraordinary refractive index
NE = 1.65

#: refractive index of the cover material (air)
NIN = 1. 
#: refractive index of the substrate material (glass)
NOUT = 1.5

#: height and width of the compute box in pixels, 
# increase this number for better rresolution and accuracy
HEIGHT, WIDTH = 128,128

#: number of layers. Increase this number to increase the accuracy, or if you increase lens thickness
NLAYERS = 321

#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,19)

#: the beta parameter of the planewave 0.0 means normal incidence
BETA = 0.0 

#: number of light passes - number of iterations, should be an odd number
NPASS = 7


#STOP CONFIGURATION-----------------------------------------------------
# computed parameters below, do not change these manually.

# z coordinate in pixelsize units
Z = np.arange(0,THICKNESS, THICKNESS/NLAYERS)*1000/PIXELSIZE #z coordinate in pixel units

# layer thickness in pixelsize units
LAYER_THICKNESS = THICKNESS/NLAYERS *1000 / PIXELSIZE # in pixelsize units

# RADIUS in pixel units
RADIUS_PIXEL = RADIUS * 1000 / PIXELSIZE 

# total thickness iub pixel units
THICKNESS_PIXEL = THICKNESS * 1000 / PIXELSIZE 

print("Using layer thickness of ", LAYER_THICKNESS * PIXELSIZE, " nm" )

# build coordinates meshgrid arrays
zz,yy,xx = np.meshgrid(Z,np.linspace(-HEIGHT/2,HEIGHT/2,HEIGHT),np.linspace(-WIDTH/2,WIDTH/2,WIDTH),indexing = "ij")

# we set the z coordinate of the bottom layer to RADIUS_PIXEL - THICKNESS_PIXEL
# the origin of the coordinate system is in the center of the reference sphere.
zz = zz + RADIUS_PIXEL - THICKNESS_PIXEL

# distance from the origin of the reference sphere.
r = (zz**2+yy**2+xx**2)**0.5

# mask for all voxels that have the distance from the origin above the sphere
mask = r > RADIUS_PIXEL

# for optical data contruction...
d = np.ones(NLAYERS) * LAYER_THICKNESS

# azimuthal angle of the director
angle = 2*np.pi*Z/PITCH 

# the eigenvalues of the dilectric tensor inside the LC
eps = np.ones((NLAYERS, HEIGHT, WIDTH, 3))
eps[...,0] = NO**2
eps[...,1] = NO**2
eps[...,2] = NE**2

# now set the immersion material
for i in range(NLAYERS):
    eps[i,...,0][mask[i]] = NIN**2
    eps[i,...,1][mask[i]] = NIN**2
    eps[i,...,2][mask[i]] = NIN**2

# Euler angles for the rotation of the eigenframe
# note that we do not need to mask these outside of the lens because for 
# isotropic material, angles do not matter

ang = np.zeros((NLAYERS, HEIGHT, WIDTH, 3))
ang[...,0] = 0. # does not matter for uniaxial material
ang[...,1] = np.pi/2 # theta angle, director is in plane
ang[...,2] = -angle[...,None,None] # azimuthal angle - define the helix

optical_data = [(d,eps,ang)]

field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                            beta = BETA, pixelsize = PIXELSIZE, n = NIN) 

#: transfer input light through stackt
field_data_out = dtmm.transfer_field(field_data_in, optical_data, 
                                     beta = BETA,
                                     phi = 0,
                                     diffraction = 1, # set to 1 for fast calculation
                                     method = "4x4",  #must be 4x4 
                                     smooth = 0.03, # between 0 and 1, can imporove convergence
                                     reflection = 2, # must be 2,3 or 4.
                                     nin = NIN, 
                                     nout = NOUT,
                                     norm = 2,  # must be 2 for cholesterics
                                     npass = NPASS # must increase this value for thick samples
                                     )

#: visualize output field, transmission mode
viewer1 = dtmm.field_viewer(field_data_out, mode = "t",n = NOUT, intensity = 0.5, focus = -20) 
fig1,ax1 = viewer1.plot()
ax1.set_title("Transmitted field")

#: residual back propagating field is close to zero
viewer2 = dtmm.field_viewer(field_data_out, mode = "r",n = NOUT)
fig2,ax2 = viewer2.plot()
ax2.set_title("Residual field")

#: visualize output field, reflection mode
viewer3 = dtmm.field_viewer(field_data_in, mode = "r", n = NIN, polarization_mode = "mode", polarizer = "LCP", analyzer = "LCP")
fig3,ax3 = viewer3.plot()
ax3.set_title("Reflected field")

import matplotlib.pyplot as plt
plt.show()
