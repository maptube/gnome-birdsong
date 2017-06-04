#!/usr/bin/env python

#pip install PySoundFile
#https://pypi.python.org/pypi/PySoundFile/0.8.1
#pip install matplotlib
#https://matplotlib.org/users/pyplot_tutorial.html
#als0 lookup melfcc.m

#this is useful: https://dsp.stackexchange.com/questions/32076/fft-to-spectrum-in-decibel

#this is VERY good on MFCC: http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

import numpy as np
#from collections import deque
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
def plotSpectrogram(s,freq,filename):
    #todo: need to compute mag and phase here for plotting - contents of "s" param are imaginary and real parts
    grid = np.array(s) #turn the list of lists into a numpy array we can plot
    x, y = np.shape(grid)
    
    #colmap = cm.Greys
    #colmap = cm.gist_yarg
    #colmap = cm.gist_gray
    #colmap = cm.binary
    #colmap=cm.gist_rainbow
    #colmap = cm.copper
    #colmap=cm.gnuplot
    colmap=cm.gnuplot2
    plt.imshow(np.transpose(grid), origin="lower", aspect="auto", cmap=colmap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (KHz)")
    plt.xlim([0, x-1])
    plt.ylim([0, y])
    plt.yticks(
        [0,y/4,y/2,3*y/4,y],
        [freq[0]/1000.0,freq[y/4]/1000.0,freq[y/2]/1000.0,freq[3*y/4]/1000.0,freq[y-1]/1000.0]
    )
    #plt.yticks(np.arange(0,y,y/4),np.arange(freq[0]/1000.0,freq[y-1]/1000.0,(freq[y-1]-freq[0])/(4*1000)))
    plt.tight_layout() #it cuts the y label off otherwise

    #plt.show()
    F=plt.gcf()
    DPI=float(F.get_dpi())
    F.set_size_inches(1280.0/DPI,960.0/DPI)
    plt.subplots_adjust(top=0.88) #stupid thing cuts the top off if you don't do this - it's fine if you don't adjust the size though
    plt.savefig(filename)
    plt.close() #if you don't close it, then it's the same object next time around!
    
    #xmin=0.0
    #xmax=float(len(s))
    #ymin=0.0
    #ymax=float(len(s[0]))
    #plt.imshow(grid, extent=(xmin, xmax, ymax, ymin), interpolation='nearest', cmap=cm.gist_rainbow)
    #plt.show()

###############################################################################

"""Plot a simple time domain signal"""
def plotSignal(x):
    t=np.arange(0,len(x),1)
    plt.plot(t,x)
    plt.xlabel('sample')
    plt.ylabel('x')
    plt.title('Signal')
    plt.grid(True)
    plt.show()
    

###############################################################################

def dbfsfft(x,fftsize,sampleRate):
    """Compute spectrogram (fft real) for this time window. Uses Hamming window function."""
    ref=1.0 #full scale reference - would be 65535 for int values
    N = len(x)
    wdw = hamming(N)
    #plotSignal(wdw)
    xrms=rms(x)
    x = x * wdw
    #plotSignal(x)
    spec = np.fft.rfft(x,fftsize) #real part only fft.fft would do the img mirror
    freq = np.arange((fftsize / 2) + 1) / (float(fftsize) / sampleRate) #need frequency bins for plotting
    #find the magnitude of the complex numbers in spec
    spec_mag = np.abs(spec)*2/np.sum(wdw) #magnitude scaling by window: np.abs(s) is amplitude spectrum, np.abs(s)**2 is power
    spec_dbfs = 20 * np.log10(spec_mag/ref) #conversion to db rel full scale
    #print "max,min=",np.max(spec_dbfs),np.min(spec_dbfs),np.max(spec_mag),np.min(spec_mag),xrms,np.sum(wdw)

    #todo: return frequency bands as well? need sampling frequency for this
    #print "len spec_dbfs=",len(spec_dbfs)
    return freq, spec_dbfs

###############################################################################

#def dbfs(spec_mag):
#    """Compute db relative to full scale based on the magnitudes in spec"""
#    ref=1.0 #as spec samples are floats, would be 65535 for 16 bit ints
#    spec_dbfs = []
#    #x, y = np.shape(spec_mag)
#    for spec in spec_mag:   
#        dbfs = 20 * np.log10(spec/ref+0.0000001) # conversion to db rel full scale
#        spec_dbfs.append(dbfs)
#    #print "max,min=",np.max(spec_dbfs),np.min(spec_dbfs),np.max(spec_mag),np.min(spec_mag),xrms,np.sum(wdw)
#    return spec_dbfs

###############################################################################

def rms(x):
    """Compute RMS value using time domain signal in x"""
    rms = np.sqrt(np.mean([x[i]**2 for i in range(len(x))]))
    return rms

###############################################################################

def spectrogramMedianFilter(spec):
    #todo: this is wrong - this isn't the median!
    """
    Normalise spectrogram power for each frequency band by subtracting the median from each band in turn.
    Any values below zero are set to zero.
    """
    #TODO:
    #x-axis is a list of the fft window (y-axis), we need min and max along x, which is along the frames
    xs, ys = np.shape(spec)
    #minmax = []
    #smin1=[]
    #for y in range(0,ys):
    #    smin1.append(1000000)
    #    for x in range(0,xs):
    #        if spec[x][y]<smin1[y]:
    #            smin1[y]=spec[x][y]
    #print "smin1=",smin1
    #
    #smin2=1000000
    #smin2=np.amin(spec,axis=0)
    #print "smin2=",smin2

    #min and max across frames (x-axis)
    #smin = np.amin(spec,axis=0)
    #smax = np.amax(spec,axis=0)
    smedian = np.median(spec,axis=0)
    #now take the median value away from the data across each frequency band
    for y in range(0,ys):
        for x in range(0,xs):
            spec[x][y]=spec[x][y]-smedian[y]
            if (spec[x][y]<0):
                spec[x][y]=0
    
    return spec

###############################################################################

def hzToMel(hz):
    """Convert frequency in Hz into a mel frequency - NOTE: other mel conversion scales exist"""
    #return 2595.0*np.log10(1.0+hz/700.0)
    return 1125.0*np.log(1.0+hz/700.0) #alternative representation

def melToHz(mel):
    #return 700.0*(pow(10,mel/2592.0)-1.0)
    return 700*(np.exp(mel/1125.0)-1.0)

def triangleWindow(N):
    w = []
    L = N #could use L+N, L=N+1 or L=N-1
    for n in range(N):
        w.append(1-np.abs((n-(N-1)/2)/(L/2)))
    return w

def melSpectrum(spec,freq,num_mel_bands):
    """
    Compute a mel frequency spectrum.
    NOTE: we don't want mel frequency cepstral coefficients (MFCCs), but the actual mel
    frequency power spectrogram. This should work better with the feature learning stage
    that comes next.
    @param freq Frequency bands in spec (Hz)
    @param spec The power spectrogram that we're going to transform
    @param num_mel_bands The number of mel frequency bands to map the frequencies to
    @returns the mel bands and the transformed spectrogram
    """
    #compute mel spaced filterbank (window)
    N = len(freq)
    minF = freq[0]
    maxF = freq[N-1]
    minMel = hzToMel(minF)
    maxMel = hzToMel(maxF)
    #print "minF=",minF," maxF",maxF
    #evenly spaced mel bands
    melBank = np.arange(minMel,maxMel,(maxMel-minMel)/float(num_mel_bands))
    print "melBank=",melBank
    #convert melBank back into Hz and then to sample numbers
    hzBank = [melToHz(m) for m in melBank]
    print "hzBank=",hzBank
    #to compute index in fft, i=(hz-minF)/((maxF-minF)/N)
    hzBanki = [np.floor((hz-minF)/((maxF-minF)/N)) for hz in hzBank]
    print "hzBanki=",hzBanki
    
    #TODO: now need "num_mel_bands" mel window filters (triangles) centred on the mel
    #frequencies and overlapping
    melFBank = []
    for m in range(0,num_mel_bands):
        #todo: find min and max sample indices, create window, push to melFBank
        if m==0:
            #initial case, peak on zero with no lower left part
            midi = hzBank[m]
            highi = hzBanki[m+1]
            #window highi*2 samples and centred on origin
            wdw = triangleWindow(int((highi-midi)*2))
            wdw = np.array(wdw)
            wdw = np.pad(wdw,pad_width=(0, num_mel_bands), mode='constant')
            #wdw = np.shift(wdw,-(highi-midi))
            melFBank.append(wdw)
            plotSignal(wdw)
        else:
            lowi=hzBanki[m-1]
            midi=hzBanki[m]
            #highi=hzBanki[m+1]           
            
    
    
    #wdw = triangleWindow(N)
    #mel_spec = []
    #xs, ys = np.shape(spec)
    #for y in range(0,ys):
    #    for x in range(0,num_mel_bands):
            

###############################################################################

def main():
    #define constants which determine how the learning works
    #assume sample rate of 44100
    sampleRate = 44100 #this should come from the sample itself really
    windowSeconds = 20.0/1000.0 #window time in seconds
    windowSamples = int(floor(windowSeconds*sampleRate)) #number of samples in a window
    windowOverlap = 0.5 #degree of overlap between sample windows 0.5 means 50% overlap
    windowSampleOverlap = int(floor(windowOverlap*windowSamples)) #how many samples the overlap contains
    fftSize = 512 #number of frequency bins in the ftt spectrogram
    numMelBands = 16 #number of mel frequency bands for the mel spectrogram
    #todo: you might want to print this lot out at the start of each run?
    print "sampleRate=",sampleRate
    print "windowSeconds=",windowSeconds
    print "windowSamples=",windowSamples
    print "windowOverlap=",windowOverlap
    print "windowSampleOverlap=",windowSampleOverlap
    print "fftSize=",fftSize
    print "numMelBands=",numMelBands
    ###
    
    #ham = hamming(windowSamples) # define a Hamming window covering the sample window

    ###

    #this is the data load and process
    data, datasamplerate = sf.read('training-data/xccoverbl/xccoverbl_2014_269/xc25119.flac')
    datalen = len(data)
    print "data length=",datalen," sample rate=",datasamplerate
    #todo: check datasamplerate==sampleRate?
    #plotSignal(data) #plot original time domain signal

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
        window = data[n:n+windowSamples] #if this runs off the end then we pad with zeroes
        N = len(window)
        if N<windowSamples:
            window = np.pad(window, pad_width=(0, windowSamples-N), mode='constant')
        #plotSignal(window)


        #perform RMS check on data here for frames which are silent...
        
        #compute spectrogram in db relative to full scale
        freq, spec = dbfsfft(window,fftSize,sampleRate)
        #spec_db = spec+120 #scale dbfs to db
        spectrogram.append(spec)
            
        n=n+windowSamples-windowSampleOverlap

    #that's the spectrogram computed, now we need to stack spectrogram frames and learn from them
    #TODO: here!
    print "spectrogram feature frames: ",len(spectrogram)
    print np.shape(spectrogram)
    plotSpectrogram(spectrogram,freq,'spec_xc25119.png')
    spectrogramF = spectrogramMedianFilter(spectrogram)
    plotSpectrogram(spectrogramF,freq,'spec_med_xc25119.png')
    spectrogramMels = melSpectrum(spectrogramF,freq,numMelBands)
    
    #spec = np.fft.fft(data,512)
    #print len(spec)
    #rms = [np.sqrt(np.mean(block**2)) for block in
    #   sf.blocks('training-data/xccoverbl/xccoverbl_2014_269/xc25119.flac', blocksize=1024, overlap=512)]
    #print rms



###############################################################################
if __name__ == "__main__":
    main()
