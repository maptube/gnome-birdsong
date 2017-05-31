#!/usr/bin/env python

#pip install PySoundFile
#https://pypi.python.org/pypi/PySoundFile/0.8.1
#pip install matplotlib
#https://matplotlib.org/users/pyplot_tutorial.html
#als lookup melfcc.m

#this is useful: https://dsp.stackexchange.com/questions/32076/fft-to-spectrum-in-decibel

import numpy as np
import soundfile as sf
from math import cos, pi, floor
import matplotlib.pyplot as plt
import matplotlib.cm as cm

###############################################################################

"""
Hamming Window: w(n)=0.54-0.46 cos (2PI*n/N), 0<=n<=N
NOTE: apparently, numpy has np.hamming(N) function to do exactly this
@param N number of samples to create window for
@returns a list of Hamming window weights with N values in it
"""
def hamming(N):
    w = []
    for n in range(N):
        w.append(0.54-0.46*cos(2*pi*n/N))
    return w

###############################################################################

"""
Plot a spectrogram using matplotlib.
@param s is the output of np.fft.fft which contains the imaginary and real parts
"""
def plotSpectrogram(s,filename):
    #todo: need to compute mag and phase here for plotting - contents of "s" param are imaginary and real parts
    grid = np.array(s) #turn the list of lists into a numpy array we can plot
    x, y = np.shape(grid)
    #mag = np.array([np.linalg.norm(elem) for elem in np.nditer(grid)])
    #mag = np.reshape(x,y)
    mag = np.empty([x,y])
    #x2, y2 = np.shape(mag)
    #print "x=",x,"y=",y
    #print "x2=",x2,"y2=",y2
    for xi in range(0,x):
        for yi in range(0,y):
            mag[xi][yi]=np.linalg.norm(grid[xi][yi])
    #print "x=",x,"y=",y
    #x2, y2 = np.shape(mag)
    #print "x2=",x2,"y2=",y2
    #print grid[0][0],mag[0][0]

    #colmap = cm.Greys
    #colmap = cm.gist_yarg
    #colmap = cm.gist_gray
    #colmap = cm.binary
    #colmap=cm.gist_rainbow
    #colmap = cm.copper
    #colmap=cm.gnuplot
    colmap=cm.gnuplot2
    plt.imshow(np.transpose(mag), origin="lower", aspect="auto", cmap=colmap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, x-1])
    plt.ylim([0, y])

    #plt.show()
    plt.savefig(filename)
    
    #xmin=0.0
    #xmax=float(len(s))
    #ymin=0.0
    #ymax=float(len(s[0]))
    #plt.imshow(grid, extent=(xmin, xmax, ymax, ymin), interpolation='nearest', cmap=cm.gist_rainbow)
    #plt.show()

###############################################################################

def dbfsfft(x,fftsize):
    """Compute spectrogram in db relative to full scale"""
    ref=1.0
    N = len(x)
    wdw = hamming(N)
    x = x * wdw
    spec = np.fft.rfft(x,fftsize) #real part only
    #spec = np.fft.fft(x,fftsize)
    spec_mag = np.abs(spec)*2/np.sum(wdw) #magnitude scaling by window
    spec_dbfs = 20 * np.log10(spec_mag/ref) #conversion to db rel full scale

    #todo: return frequency bands as well? need sampling frequency for this
    #print "len spec_dbfs=",len(spec_dbfs)
    return spec_dbfs

###############################################################################

def main():
    #matplotlib test
    #grid=np.array([[1, 2, 3], [4, 5, 6], [7,8,9]], np.int32)
    #x=np.array([0,1,2])
    #y=np.array([0,1,2])
    #plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()), interpolation='nearest', cmap=cm.gist_rainbow)
    #plt.show()

    
    #define constants which determine how the learning works
    #assume sample rate of 44100
    sampleRate = 44100 #this should come from the sample itself really
    windowSeconds = 20.0/1000.0 #window time in seconds
    windowSamples = int(floor(windowSeconds*sampleRate)) #number of samples in a window
    windowOverlap = 0.5 #degree of overlap between sample windows 0.5 means 50% overlap
    windowSampleOverlap = int(floor(windowOverlap*windowSamples)) #how many samples the overlap contains
    fftSize = 512 #number of frequency bins in the ftt spectrogram
    #todo: you might want to print this lot out at the start of each run?
    print "sampleRate=",sampleRate
    print "windowSeconds=",windowSeconds
    print "windowSamples=",windowSamples
    print "windowOverlap=",windowOverlap
    print "windowSampleOverlap=",windowSampleOverlap
    print "fftSize=",fftSize
    ###
    
    ham = hamming(windowSamples) # define a Hamming window covering the sample window

    ###

    #this is the data load and process
    data, datasamplerate = sf.read('training-data/xccoverbl/xccoverbl_2014_269/xc25119.flac')
    datalen = len(data)
    print "data length=",datalen," sample rate=",datasamplerate
    #todo: check datasamplerate==sampleRate?

    #spectrogram computation on entire waveform file

    spectrogram = []
    
    #now go through each block in turn
    n=0
    while (n<datalen):
        #window=[0]*windowSamples
        #i=0
        #for m in range(int(n),int(min(datalen,n+windowSamples))):
        #    window[i]=data[m]*ham[i]
        #    i+=1
        window = data[n:n+windowSamples]

        #at the point we have window[] containing the data with the hamming weights applied
        #the next part of the analysis is the spectrogram slice

        #perform RMS check on data here for frames which are silent...
        
        #spec = np.fft.fft(window,fftSize)
        spec = dbfsfft(window,fftSize)
        #spec_db = spec+120 #scale dbfs to db
        spectrogram.append(spec)
            
        n=n+windowSamples-windowSampleOverlap

    #that's the spectrogram computed, now we need to stack spectrogram frames and learn from them
    #TODO: here!
    print "spectrogram feature frames: ",len(spectrogram)
    print np.shape(spectrogram)
    plotSpectrogram(spectrogram,'spec_xc25119.png')
    
    #spec = np.fft.fft(data,512)
    #print len(spec)
    #rms = [np.sqrt(np.mean(block**2)) for block in
    #   sf.blocks('training-data/xccoverbl/xccoverbl_2014_269/xc25119.flac', blocksize=1024, overlap=512)]
    #print rms



###############################################################################
if __name__ == "__main__":
    main()
