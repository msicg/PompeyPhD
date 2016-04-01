#1d_file.py - plots 1 dimensional spherically averaged profiles of the density, temperature, etc, for a specified
#             radius in kpc centred on the highest density point in the simulation.
#   
#             Just specify the Enzo outfile(s) after 1d_file.py to be plotted, and then the
#             length of the profile (in kpc)
###########################################################################
import sys #allows you to import the command line arguments
import yt
import matplotlib.pyplot as plt
import numpy as np

for i in range(1,len(sys.argv)-1):

# Loads in the data, find the max density  
      
    #ds = yt.load(sys.argv[i])
    #ds.print_stats()
    #val, loc = ds.find_max("density")
    #Coords of the MBH inserted at z=17.26
    #loc = np.array([0.5103157529, 0.5099027380, 0.5093864920])
    #print loc
    #dx = (ds.index.get_smallest_dx()).in_units('kpc')
    #print 'smallest dx in kpc = ', dx

# An array of the fields

    field = ('density', 'Temperature', 'H_fraction', 'H_p1_fraction', 'He_fraction', 'He_p1_fraction', 'He_p2_fraction', 'H_m1_fraction', 'H2_fraction', 'H2_p1_fraction')

# Creates a sphere for the averaging of a radius specified in the command line arguments

    #my_sphere = ds.sphere(loc, (float(sys.argv[len(sys.argv)-1]),'kpc'))

# Allows profiles for each field to be saved to a variable with a different name   
    name = dict()
    profile = dict()
    for k in range(0, max(np.shape(field))):
	#print k
        #print field[k]
        name[k] = 'plot'+str(k)
        #print name[k]
        profile[k] = 'profile'+str(k)
        #print profile[k]
  
# Creates the plots the profiles for the various quantities needed to calculate the NIR luminosity
# and saves the plots

    #	name[k] = yt.ProfilePlot(my_sphere, "radius", field[k], weight_field="cell_mass", n_bins=1000) 
    #	name[k].set_unit('radius', 'kpc')
    #   name[k].set_xlim(1, float(sys.argv[len(sys.argv)-1]))
    #	name[k].save()    

# Writes the spherically averaged quantity as function of radius to text files

    	#profile[k] = name[k].profiles[0]
        #data=open(sys.argv[i]+'_'+field[k]+'_1d_profile.txt', 'w')
    	#for j in range(0, max(np.shape(profile[k].x))):
        #     if float(profile[k][field[k]][j]) != 0:     	
        #     	data.write(str(float(profile[k].x[j])))
        #     	data.write('\t')
    	#     	data.write(str(float(profile[k][field[k]][j])))
    	#     	data.write('\n')
        #data.close()
   
# Imports the data for a specified field written to a text file

    afield = field[9]
    
    array = np.loadtxt('RD0004/RedshiftOutput0004_'+afield+'_1d_profile.txt')
    x = array[:,0]
    y = array[:,1]

# Does some smoothing
    
    from scipy.signal import savgol_filter
    ynew = savgol_filter(y, 5, 1)
    
# Plots the data imported from the text file

    plt.plot(x,y,x,ynew)
    plt.yscale('log')    
    plt.xscale('log')
    plt.show()

# Writes the smoothed data of the field imported above to text files

    data=open(sys.argv[i]+'_'+afield+'_smooth_1d_profile.txt', 'w')
    for j in range(0, max(np.shape(x))):           
    	data.write(str(float(x[j])))
    	data.write('\t')
    	data.write(str(float(ynew[j])))
    	data.write('\n')
    data.close()

# Saves the smoothed plot

    plt.plot(x, ynew)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(sys.argv[i]+'_'+afield+'_smooth_1d_profile.png')

################################################################################
# Tries to smooth or fit curves to the data

# FT of the data, removing the high frequencies, then iFT back to real space

#import scipy.fftpack

    #w = scipy.fftpack.rfft(y)
    #spectrum = w**2
    #print int((max(np.shape(w)))*0.1)
    #for q in range (int((max(np.shape(w)))*0.2), int(max(np.shape(w)))):
    #    w[q] = 0
    #y_new = scipy.fftpack.irfft(w)

# Calculates a moving average of the data

    #def movingav(y, window_size):
        #window = np.ones(window_size)/window_size
        #return np.convolve(y, window, mode='same')
    #y_smooth = movingav(y,5)

# Tries to smooth the data using a Savitzky-Golay filter

#from scipy.signal import savgol_filter

    #ynew = savgol_filter(y, 3, 1)

# 1d interpolation to fit a curve to the points

#from scipy.interpolate import interp1d
#from scipy.interpolate import spline
#from scipy.interpolate import splev, splrep
#from scipy.interpolate import UnivariateSpline
#from scipy.interpolate import Rbf
#matplotlib.use('Agg') Backend for creating high quality images

    #interp = scipy.interp1d(x,y)
    #interp = interp1d(x,y, 'cubic')
    #spline = spline(x,y)

    #tck = splrep(x,y)
    
    #spl = UnivariateSpline(x, y)
    #tck = splrep(x,y, s=0.0000000001)
    #f = interp1d(x, y, kind='cubic')
    #xnew = np.linspace(1, 49, num=2000, endpoint=True)
    #ynew = splev(xnew, tck)
    #spl.set_smoothing_factor(0.01)
    #rbfi = Rbf(x, y)
    
# Fit a function to a curve   

#from scipy.optimize import curve_fit

    #def func(x,a, b, c):
	#return 1/(a*x**5+b*x**4+c)
    #popt, pcov = curve_fit(func, x, y)
    #yfit = func(xnew, popt[0], popt[1], popt[2])
