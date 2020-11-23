import sys
import numpy as np
import scipy
import matplotlib
import PyDiff as pd
np.set_printoptions(threshold=sys.maxsize)

def DataGen(S0,S02,D,D2,ImageSize,bvalues,noise):
    Im = np.zeros([480,480,len(bvalues)])
    
    i = 0
  
    for b in np.arange(len(bvalues)):
        Im[:,:,b] = S0*np.exp(-(D*bvalues[i]))
        i = i+1
            
    ind = int(len(bvalues))
    
    
    # Add some noise
    Im = Im + np.random.rand(480,480,ind)*noise
    pixels = np.abs(Im)
    return pixels

          
        
    
    
    
S0 = 1200
S02 = 0.2*S0
D = 3e-3
D2 = 3e-2
noise = 0
bvalues = np.array([1,2,3,4,5,6,7,8,9,10])
ImageSize = np.array([480,480])
px = DataGen(S0,S02,D,D2,ImageSize,bvalues,noise)

S0Fit = pd.fit_ADC(px,bvalues)
s0 = S0Fit[0]
adc = S0Fit[1]
print(s0)
    