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
import os

from spectrogram import Spectrogram

###############################################################################

##"""
##Hamming Window: w(n)=0.54-0.46 cos (2PI*n/N), 0<=n<=N
##NOTE: apparently, numpy has np.hamming(N) function to do exactly this
##@param N number of samples to create window for
##@returns a list of Hamming window weights with N values in it
##"""
##def hamming(N):
##    w = []
##    for n in range(N):
##        w.append(0.54-0.46*cos(2*pi*n/N))
##    return w

###############################################################################

##"""
##Plot a spectrogram using matplotlib.
##@param s is the output of np.fft.fft which contains the imaginary and real parts
##"""
##def plotSpectrogram(s,freq,filename):
##    #todo: need to compute mag and phase here for plotting - contents of "s" param are imaginary and real parts
##    grid = np.array(s) #turn the list of lists into a numpy array we can plot
##    x, y = np.shape(grid)
##    
##    #colmap = cm.Greys
##    #colmap = cm.gist_yarg
##    #colmap = cm.gist_gray
##    #colmap = cm.binary
##    #colmap=cm.gist_rainbow
##    #colmap = cm.copper
##    #colmap=cm.gnuplot
##    colmap=cm.gnuplot2
##    plt.imshow(np.transpose(grid), origin="lower", aspect="auto", cmap=colmap, interpolation="none")
##    plt.colorbar()
##
##    plt.xlabel("time (s)")
##    plt.ylabel("frequency (KHz)")
##    plt.xlim([0, x-1])
##    plt.ylim([0, y])
##    plt.yticks(
##        [0,y/4,y/2,3*y/4,y],
##        [freq[0]/1000.0,freq[y/4]/1000.0,freq[y/2]/1000.0,freq[3*y/4]/1000.0,freq[y-1]/1000.0]
##    )
##    #plt.yticks(np.arange(0,y,y/4),np.arange(freq[0]/1000.0,freq[y-1]/1000.0,(freq[y-1]-freq[0])/(4*1000)))
##    plt.tight_layout() #it cuts the y label off otherwise
##
##    #plt.show()
##    F=plt.gcf()
##    DPI=float(F.get_dpi())
##    F.set_size_inches(1280.0/DPI,960.0/DPI)
##    plt.subplots_adjust(top=0.88) #stupid thing cuts the top off if you don't do this - it's fine if you don't adjust the size though
##    plt.savefig(filename)
##    plt.close() #if you don't close it, then it's the same object next time around!
##    
##    #xmin=0.0
##    #xmax=float(len(s))
##    #ymin=0.0
##    #ymax=float(len(s[0]))
##    #plt.imshow(grid, extent=(xmin, xmax, ymax, ymin), interpolation='nearest', cmap=cm.gist_rainbow)
##    #plt.show()

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

##def dbfsfft(x,fftsize,sampleRate):
##    """
##    Compute spectrogram (fft real) for this time window. Uses Hamming window function.
##    @param x signal
##    @param fftsize number of bands in fft e.g. 512
##    @param sampleRate rate signal was sampled at, required to calculate the frequency bands in hz
##    @returns the frequency bands in hz, the db full scale spectrum (log) and the raw magnitude spectrum (lin)
##    """
##    ref=1.0 #full scale reference - would be 65535 for int values
##    N = len(x)
##    wdw = hamming(N)
##    #plotSignal(wdw)
##    xrms=rms(x)
##    x = x * wdw
##    #plotSignal(x)
##    spec = np.fft.rfft(x,fftsize) #real part only fft.fft would do the img mirror
##    freq = np.arange((fftsize / 2) + 1) / (float(fftsize) / sampleRate) #need frequency bins for plotting
##    #find the magnitude of the complex numbers in spec
##    spec_mag = np.abs(spec)*2/np.sum(wdw) #magnitude scaling by window: np.abs(s) is amplitude spectrum, np.abs(s)**2 is power
##    spec_dbfs = 20 * np.log10(spec_mag/ref) #conversion to db rel full scale
##    #print "max,min=",np.max(spec_dbfs),np.min(spec_dbfs),np.max(spec_mag),np.min(spec_mag),xrms,np.sum(wdw)
##
##    #print "len spec_dbfs=",len(spec_dbfs)
##    return freq, spec_dbfs, spec_mag

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

##def spectrogramMedianFilter(spec):
##    """
##    Normalise spectrogram power for each frequency band by subtracting the median from each band in turn.
##    Any values below zero are set to zero.
##    """
##    #TODO:
##    #x-axis is a list of the fft window (y-axis), we need min and max along x, which is along the frames
##    xs, ys = np.shape(spec)
##    #minmax = []
##    #smin1=[]
##    #for y in range(0,ys):
##    #    smin1.append(1000000)
##    #    for x in range(0,xs):
##    #        if spec[x][y]<smin1[y]:
##    #            smin1[y]=spec[x][y]
##    #print "smin1=",smin1
##    #
##    #smin2=1000000
##    #smin2=np.amin(spec,axis=0)
##    #print "smin2=",smin2
##
##    #min and max across frames (x-axis)
##    #smin = np.amin(spec,axis=0)
##    #smax = np.amax(spec,axis=0)
##    smedian = np.median(spec,axis=0)
##    #now take the median value away from the data across each frequency band
##    for y in range(0,ys):
##        for x in range(0,xs):
##            spec[x][y]=spec[x][y]-smedian[y]
##            if (spec[x][y]<0):
##                spec[x][y]=0
##    
##    return spec

###############################################################################

##def hzToMel(hz):
##    """Convert frequency in Hz into a mel frequency - NOTE: other mel conversion scales exist"""
##    #return 2595.0*np.log10(1.0+hz/700.0)
##    return 1125.0*np.log(1.0+hz/700.0) #alternative representation

##def melToHz(mel):
##    #return 700.0*(pow(10,mel/2592.0)-1.0)
##    return 700*(np.exp(mel/1125.0)-1.0)

##def triangleWindow(N):
##    w = []
##    L = float(N+1) #could use L=N, L=N+1 or L=N-1, using L=N+1 gives me a triangle window for small N
##    for n in range(N):
##        w.append(1.0-np.abs((n-(N-1)/2.0)/(L/2.0)))
##    return w

##def createMelFilters(freq,num_mel_bands):
##    """
##    Create a filterbank of triangle filters which is used to convert the regular FFT into a mel
##    frequency range. This is done using a bank of window filters (frequency domain) which act on
##    the FFT power bands to convert them to mel power bands. In simple terms, the mel bands get
##    increasingly spaced out as the frequency increases (log scale).
##    @param freq The list of frequency bands in the input fft
##    @param num_mel_banks The number of mel power bands that we want to create
##    @returns the mel frequency bands AND
##    a list of num_mel_banks in length where each item is a list of len(freq) in length
##    containing a window function used to create that particular mel frequency power from the
##    original fft.
##    TODO: the final filter is centred a lot lower than the top frequency - you might want to change this
##    or maybe not? You could add a final filter centred on the top frequency that only has a lower part
##    of the triangle to it.
##    """
##    N = len(freq)
##    minF = freq[0]
##    maxF = freq[N-1]
##    minMel = hzToMel(minF)
##    maxMel = hzToMel(maxF)
##    #print "minF=",minF," maxF",maxF
##    #evenly spaced mel bands NOTE: maxMel+1 so that it includes maxMel as the final value. We need this
##    #for the min,max pairs to make the filterbank. It results in len(melBank)=num_mel_bands+1 and similarly
##    #for hzBank and hzBanki so the final value in all of them corresponds to maxF (frequency)
##    melBank = np.arange(minMel,maxMel+1,(maxMel-minMel)/float(num_mel_bands))
##    #print "melBank=",melBank
##    #convert melBank back into Hz and then to sample numbers
##    hzBank = [melToHz(m) for m in melBank]
##    #print "hzBank=",hzBank
##    #to compute index in fft, i=(hz-minF)/((maxF-minF)/N)
##    hzBanki = [np.floor((hz-minF)/((maxF-minF)/N)) for hz in hzBank]
##    #print "hzBanki=",hzBanki
##    
##    #TODO: now need "num_mel_bands" mel window filters (triangles) centred on the mel
##    #frequencies and overlapping
##    melFBank = []
##    for m in range(0,num_mel_bands):
##        #todo: find min and max sample indices, create window, push to melFBank
##        if m==0:
##            #initial case, peak on zero with no lower left part
##            midi = int(hzBank[m])
##            highi = int(hzBanki[m+1])
##        else:
##            #lowi=hzBanki[m-1]
##            midi=int(hzBanki[m])
##            highi=int(hzBanki[m+1]) #NOTE: len(hzBanki)=num_mel_bands+1, so this is valid
##        #Now make a window containing zeros and a triangle window from the low, mid and high i.
##        #The triangle window is then placed into the zero window so it's peak is on midi
##        lowi=midi-(highi-midi)
##        if lowi<0:
##            lowi=0
##        #now make a window function of length "fftSize" freq bands, with a triangle filter covering the mel bit
##        wdw = np.zeros(N)
##        #print "window: lowi=",int(lowi)," midi=",int(midi)," highi=",int(highi)
##        Ntri=highi-lowi #number of samples in the tri filter window 
##        triwdw = triangleWindow(Ntri)
##        #print "Ntri=",Ntri,"triwdw=",triwdw
##        wdw[lowi:highi] = triwdw
##        melFBank.append(wdw)
##        #plotSignal(wdw)
##    #OK, so at this point melFBank contains the triangle filers (f domain) that we need to transform
##    #the fft into a mel fft.
##    return melBank, melFBank

##def melSpectrum(spec,freq,num_mel_bands):
##    """
##    Compute a mel frequency spectrum.
##    NOTE: we don't want mel frequency cepstral coefficients (MFCCs), but the actual mel
##    frequency power spectrogram. This should work better with the feature learning stage
##    that comes next.
##    @param freq Frequency bands in spec (Hz)
##    @param spec The power spectrogram that we're going to transform
##    @param num_mel_bands The number of mel frequency bands to map the frequencies to
##    @returns the mel frequency bands and the transformed spectrogram
##    """
##    #Compute mel spaced filterbank (window). This is to apply to the fft power spectra.
##    melFreq, melFBank = createMelFilters(freq,num_mel_bands)
##    
##
##    #TO REMEMBER: do you need to normalise the energy in each filter window? They don't all have the
##    #same area.
##
##    #todo: apply the filters to all the frames and return data
##    mel_spec = []
##    xs, ys = np.shape(spec)
##    for x in range(0,xs):
##        fft=spec[x]
##        mel_fft = []
##        for y in range(0,num_mel_bands):
##            #dot product between mel filter bank and fft
##            mel_fft.append(np.dot(melFBank[y],fft))
##        mel_spec.append(mel_fft)
##
##    return melFreq, mel_spec

###############################################################################

##def logSpectrogram(spec):
##    """
##    Apply a log scaling function to a set of spectrogram frames.
##    This is so that I can compare log and lin spectrums.
##    @param spec list of frames containing a list of power spectrum magnitudes
##    @returns the spectrum with the log function applied to everything
##    """
##    xs, ys = np.shape(spec)
##    logspec=[]
##    for x in range(0,xs):
##        fft=spec[x]
##        logfft = [20*np.log10(v+0.0001) for v in fft]
##        logspec.append(logfft)
##    return logspec
    

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
    #numMelBands = 16 #number of mel frequency bands for the mel spectrogram
    #todo: you might want to print this lot out at the start of each run?
    print "sampleRate=",sampleRate
    print "windowSeconds=",windowSeconds
    print "windowSamples=",windowSamples
    print "windowOverlap=",windowOverlap
    print "windowSampleOverlap=",windowSampleOverlap
    print "fftSize=",fftSize
    #print "numMelBands=",numMelBands
    ###

    #housekeeping - where everything is and where it goes
    inFileTrainManifest = "training-data/xccoverbl/xcmeta.csv" #manifest file for all training data sound files
    inDirData = "training-data/xccoverbl/xccoverbl_2014_269" #location of the wave files
    outDirSpectrogram = "spectrograms" #where spectrogram plots can get put so they're all together - basically debug/testing
    

    ###

    #go through the training manifest and generate spectrograms for all the files
    #id	gen	sp	en	rec	cnt	lat	lng	type	lic
    #132608	Acanthis	flammea	Common Redpoll	Jarek Matusiak	Poland	50.7932	15.4995	female, male, song	http://creativecommons.org/licenses/by-nc-sa/3.0/
    #Results in [ id="132608", gen="Acanthis", sp="flammea", en="Common Redpoll", rec="Jarek Matusiak",
    #                   cnt="Poland", lat="50.7932", lng="15.4995", type="female, male, song",
    #                   lic="http://creativecommons.org/licenses/by-nc-sa/3.0/" ]
    #todo: need to make a set out of the common names so I can give them an id
    with open(inFileTrainManifest) as f:
        next(f) #skip header line
        for line in f:
            fields = line.split('\t')
            fileid = fields[0]
            commonName = fields[3]
            filename = os.path.join(inDirData,fileid+".flac")
            print commonName," ",filename
            #todo: pick up file, make the spectrogram from it and save spec, plus serialise the data for training

    #spectrogram computation on entire waveform file

    spectrogramDBFS = [] #this is the db relative full scale power spectra
    spectrogramMag = [] #and this is the raw linear power spectra
    
    #now go through each block in turn
##    n=0
##    while (n<datalen):
##        #window=[0]*windowSamples
##        #i=0
##        #for m in range(int(n),int(min(datalen,n+windowSamples))):
##        #    window[i]=data[m]*ham[i]
##        #    i+=1
##        window = data[n:n+windowSamples] #if this runs off the end then we pad with zeroes
##        N = len(window)
##        if N<windowSamples:
##            window = np.pad(window, pad_width=(0, windowSamples-N), mode='constant')
##        #plotSignal(window)
##
##
##        #perform RMS check on data here for frames which are silent...
##        
##        #compute spectrogram in db relative to full scale
##        freq, spec, specmag = dbfsfft(window,fftSize,sampleRate)
##        #spec_db = spec+120 #scale dbfs to db
##        spectrogramDBFS.append(spec) #this is the log db relative full scale version (normal one plotted)
##        spectrogramMag.append(specmag) #this is the raw magnitude spectrum directly off the fft
##            
##        n=n+windowSamples-windowSampleOverlap

#new bit

    genSpec = Spectrogram()
    genSpec.load('training-data/xccoverbl/xccoverbl_2014_269/xc25119.flac')
    spectrogramDBFS = genSpec.spectrogramDBFS
    spectrogramMag = genSpec.spectrogramMag
    freq=genSpec.freq #these are the frequency bands that go with the above
    

#end of new bit
    

    #that's the spectrogram computed, now we need to stack spectrogram frames and learn from them
    #TODO: here!
    print "spectrogram feature frames: ",len(spectrogramDBFS)
    print "spectrogram=",np.shape(spectrogramDBFS)
    #At this point we have a number of possibilities. There is the spectrumDBFS, which is the DB relative
    #full scale log magnitude power spectrum, or the raw linear magnitude power spectrum which is just from
    #the raw fft data directly (magnitude of the im, re vector). Either of these can be median filtered
    #and then converted to a mel scale.
    genSpec.plotSpectrogram(spectrogramDBFS,freq,'spec_dbfs_xc25119.png')
    genSpec.plotSpectrogram(spectrogramMag,freq,'spec_mag_xc25119.png')
    spectrogramDBFS_Filt = genSpec.spectrogramMedianFilter(spectrogramDBFS)
    spectrogramMag_Filt = genSpec.spectrogramMedianFilter(spectrogramMag)
    genSpec.plotSpectrogram(spectrogramDBFS_Filt,freq,'spec_dbfs_med_xc25119.png')
    genSpec.plotSpectrogram(spectrogramMag_Filt,freq,'spec_mag_med_xc25119.png')
    #now you can do a mel frequency one off of either spectrogram[Mag|DBFS] or spectrogram[Mag|DBFS]_Filt (median filtered)
    melfreq, spectrogramMels = genSpec.melSpectrum(spectrogramMag_Filt,freq)
    print "mels=",np.shape(spectrogramMels)
    genSpec.plotSpectrogram(spectrogramMels,melfreq,"spec_mel_mag_med_xc25119.png")
    spectrogramMelsLog = genSpec.logSpectrogram(spectrogramMels) #this applies a log function to the magnitudes
    genSpec.plotSpectrogram(spectrogramMelsLog,melfreq,"spec_mel_log_mag_med_xc25119.png")
    #TODO: need velocity and acceleration features
    #and rms normalisation
    #and silence removal
    
    #spec = np.fft.fft(data,512)
    #print len(spec)
    #rms = [np.sqrt(np.mean(block**2)) for block in
    #   sf.blocks('training-data/xccoverbl/xccoverbl_2014_269/xc25119.flac', blocksize=1024, overlap=512)]
    #print rms



###############################################################################
if __name__ == "__main__":
    main()
