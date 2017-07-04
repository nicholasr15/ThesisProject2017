########################################
#Nicholas Randall                      
#Code for Master Thesis 2017           
########################################
import matplotlib.pyplot as plt
import plotly.plotly as py
import numpy as np
get_ipython().magic('matplotlib notebook')
# The code come from 
#https://plot.ly/matplotlib/fft/#basic-fft-plot-with-matplotlib
# Learn about API authentication here: 
#https://plot.ly/python/getting-started
# Find your api_key here: 
#https://plot.ly/settings/api


# In[13]:

Fs = 150.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector


# In[14]:

ff = 5;   # frequency of the signal
y = np.sin(2*np.pi*ff*t)


# In[15]:

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))] # one side frequency range


# In[16]:

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(int(n/2))] # one side of the FFT range


# In[17]:

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

#plot_url = py.plot_mpl(fig, filename='mpl-basic-fft')