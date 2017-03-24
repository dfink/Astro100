
# coding: utf-8
import numpy as np, astropy.units as u, matplotlib.pyplot as plt
from astropy.analytic_functions import blackbody_lambda, blackbody_nu
from mpl_toolkits.axes_grid1 import make_axes_locatable
################################################################

# start here
def get_files(path, obj, name): #Creates a list of fits files 
    import glob
    elist = glob.glob(path) #Create a list of file names
    explist = [] #Create a list for the specified file types
    for f in elist:
        hdulist = fits.open(f)
        scihead = hdulist[0].header #Read the header
        if scihead[str(obj)] == str(name): #Look in the header for the type of file
            explist.append(f)
    return explist

def phot(image, x, y, rad, skyrad): #Pass the image, x/y coordinates of the star, aperture radius, sky annulus [r1,r2]
        x0, x1 = x-skyrad[1], x+skyrad[1] #Define X/Y coordinates for the box around the target star
        y0, y1 = y-skyrad[1], y+skyrad[1]
        box = image[x0:x1+1,y0:y1+1]
        (nx,ny) = np.shape(box)
        xbox = (np.arange(nx*ny).reshape(nx,ny)) % nx + (x-skyrad[1]) #
        #Create an array like this:
        # 0 1 2 3 ... nx
        # 0 1 2 3
        # 0 1 2 3
        # ...
        ybox = (np.arange(nx*ny).reshape(nx,ny)) / nx + (y-skyrad[1])
        #Create an array like this:
        # 0 0 0 0 ...
        # 1 1 1 1
        # 2 2 2 2
        # ...
        # ny
        #The numbers in xbox and ybox are like units of distance 
        # that we use in r2 below
        
        r2 = (x-xbox)**2 + (y-ybox)**2
        aper = np.where(r2 < rad*rad)
        #aper = np.where(r2 < rad*rad, 1, 0)
        #sky = np.where(r2 > skyrad[0]**2, 1), & np.where(r2 < skyrad[1]**2, 1, 0)
        sky = np.where(r2 > skyrad[0]**2) & np.where(r2 < skyrad[1]**2)
        #### aper and sky specify where in the box cut out that xbox and ybox are within
        #### r2 or between the two sky radii in skyrad
        # aper_ind and sky_ind create 2D arrays of 1 and 0 to indicate True and False for where
        # the np.where() statement, here, they also indicate the number of pixels in those regions
        # when the np.where() statement is 1 (line 44)
        
        background = np.median(image[sky])
        Ns, Na = len(np.where(sky_ind == 1)[0]), len(np.where(aper_ind == 1)[0])
        #Number of pixels and the sky and aperture
        sig_s, sig_a = np.std(box[sky]), np.std(box[aper])
        err = np.sqrt( ( Na*sig_s/np.sqrt(Ns) )**2 + ( np.sqrt(Na)*sig_a )**2 )
        box -= background #Subtract the sky background from the box
        star_counts = np.sum(box[aper])
        return star_counts, err
    
### In serious need of refining ####    
def find_peaks(img):
    y_pix_vals = np.zeros(len(img[0,:]))
    for i in range(len(img[0,:])):
        col = img[:,i] #Slice out a column
        y_pix_ind = np.where(col == np.max(col))[0] #Find the maxima of the column
        if len(y_pix_ind) > 1: #If multiple maxima are found, take the average of their locations
            y_pix_vals[i] = np.mean(y_pix_ind)
        else:
            y_pix_vals[i] = y_pix_ind
    return y_pix_vals
### In serious need of refining ####    

# Sigma-clip data over a user-specified number of times (nloops)
# deg is the order of polynomial being fit to the data (deg = 2 is a second order polynomial)
# Returns best polynomial parameters (line_params) to sigma-clipped data,
# use line_params in np.polyval(line_params, xdata)
def sigma_clip(x,y, deg = 1, nloops = 15):
    y_sig_arr = np.arange(0,nloops,1.0)
    for i in range(1,nloops):# Sigma clipping
        line_params = np.polyfit(x, y, deg)
        trc_fnc = np.polyval(line_params, x)
        y_sig = 5.0*np.std(y-trc_fnc)
        y_sig_arr[i] = y_sig
        delta_y = y-trc_fnc
        clipped_ind = np.where(np.abs(delta_y) <= y_sig)[0]
        #Where the difference in the line vs data is within 5 sigma
        y = y[clipped_ind] #Reset to te newly clipped data
        x = x[clipped_ind]
        if np.around(y_sig_arr[i],3) == np.around(y_sig_arr[i-1],3):
            print 'Converged after '+str(i)+' iterations'
            break
    return line_params

def lines(nx, center, width, amp): #Create spectral lines in mock data
    x =np.arange(nx).astype(float)
    spec = np.zeros(nx)
    for i in np.arange(len(center)):
        spec += amp[i]*np.exp(-(center[i]-x)**2.0/(2.0*width[i]**2.0))/(np.sqrt(2*np.pi)*width[i])
    return spec
    
def mockspec(nx = 500, ny = 100, sky = None, obj = None, noise = None, trace = None): #Create a mock 2D image
    im = np.zeros((ny,nx))
    y, sig = np.arange(ny), 2.0
    y0 = np.zeros(nx)+ny/2
    if trace != None: #Curve the diffraction order
        x = np.arange(nx)
        y0 = np.polyval(trace, x)
    for i in np.arange(nx):
        im[:,i] = obj[i]*np.exp(-(y-y0[i])**2/(2*sig**2))
        if sky != None: #Add sky lines
            im[:,i] += sky[i]
    if noise != None: #Add background noise
        im += np.random.randn(nx*ny).reshape(ny,nx)*noise
    return im

def mockobj(nx, temp): #Create a mock 1D spectrum
    x = np.arange(nx)
    l = x*10+5000
    wave = l*u.AA
    temperature = temp*u.K
    flux = blackbody_nu(wave, temperature).value
    return flux

def cosmic_ray(img, nrays): #Add cosmic rays to a 2D image
    nx, ny = img.shape[1], img.shape[0]
    for i in range(nrays):
        x_pos = np.random.randint(low = 0, high = nx, size = 1) 
        y_pos = np.random.randint(low = 0, high = ny, size = 1)
        cr  = np.random.randint(low = 10, high = 15, size = 1)
        img[y_pos, x_pos] = cr
    return img

def rmbias(img, bias_st, bias_fin): #Take te median of the overscan region and subtract the median and the region from 2D image
    img = img.astype(float)
    ncol, nrow = len(img[0,:]), len(img[:,0])
    over_scan = img[:,bias_st:bias_fin] #Discerned by eye
    for i in range(nrow):
        img[i,:] -= np.median(over_scan[i])
    dark = np.arange(bias_st,nrow*ncol,ncol) 
    for j in range(bias_fin):      #CUT is the user-input approximation of which horizontal pixel
        img = np.delete(img, dark, axis = 1)
    return img

def boximg(img): #Create a smoothed image to subtract from the raw image to make the trace stand out more
    box_img = np.zeros(img.shape)
    for j in range(len(img[:,0])):
        row = img[j,:]
        box_img[j,:] = convolve(row, Box1DKernel(5))
    box_img -= np.roll(box_img, -20, axis = 0)
    return box_img
    
