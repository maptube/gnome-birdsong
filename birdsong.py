#!/usr/bin/env python

#pip install PySoundFile
#https://pypi.python.org/pypi/PySoundFile/0.8.1

import numpy as np
import soundfile as sf
from math import cos, pi

###############################################################################

"""
Hamming Window: w(n)=0.54-0.46 cos (2PI*n/N), 0<=n<=N
@param N number of samples to create window for
@returns a list of Hamming window weights with N values in it
"""
def hamming(N):
    w = []
    for n in range(N):
        w.append(0.54-0.46*cos(2*pi*n/N))
    return w

###############################################################################

def main():
    #define constants which determine how the learning works
    #assume sample rate of 44100
    sampleRate = 44100 #this should come from the sample itself really
    windowSeconds = 20/1000 #window time in seconds
    windowSamples = windowSeconds*sampleRate #number of samples in a window
    windowOverlap = 0.5 #degree of overlap between sample windows 0.5 means 50% overlap
    windowSampleOverlap = windowOverlap*windowSamples #how many samples the overlap contains
    fftSize = 512 #number of frequency bins in the ftt spectrogram
    #todo: you might want to print this lot out at the start of each run?
    ###
    
    ham = hamming(windowSamples) # define a Hamming window covering the sample window

    ###

    #this is the data load and process
    data, datasamplerate = sf.read('training-data/xccoverbl/xccoverbl_2014_269/xc25119.flac')
    datalen = len(data)
    print "data length=",datalen," sample rate=",datasamplerate
    #todo: check datasamplerate==sampleRate?

    #now go through each block in turn
    n=0
    while (n<datalen):
        window=[0]*windowSamples
        i=0
        for m in range(n:min(datalen,n+windowSamples)):
            window[i]=data[m]*ham[i]
            i+=1

        #at the point we have window[] containing the data with the hamming weights applied
        #the next part of the analysis is the spectrogram slice

        #perform RMS check on data here for frames which are silent...
        
        spec = np.fft.fft(window,fftSize)
        #need to save this somewhere
            
        n+=windowSamples-windowSampleOverlap

    #that's the spectrogram computed, now we need to stack spectrogram frames and learn from them
    #TODO: here!
    
    #spec = np.fft.fft(data,512)
    #print len(spec)
    #rms = [np.sqrt(np.mean(block**2)) for block in
    #   sf.blocks('training-data/xccoverbl/xccoverbl_2014_269/xc25119.flac', blocksize=1024, overlap=512)]
    #print rms



###############################################################################
if __name__ == "__main__":
    main()
