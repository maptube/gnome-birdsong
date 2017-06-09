#!/usr/bin/env python

import numpy as np
#from collections import deque
import soundfile as sf
from math import cos, pi, floor
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Spectrogram:
    def __init__(self):
        #what do you need?
        self.sampleRate = 44100 #this should come from the sample itself really
        self.windowSeconds = 20.0/1000.0 #window time in seconds
        self.windowSamples = int(floor(self.windowSeconds*self.sampleRate)) #number of samples in a window
        self.windowOverlap = 0.5 #degree of overlap between sample windows 0.5 means 50% overlap
        self.windowSampleOverlap = int(floor(self.windowOverlap*self.windowSamples)) #how many samples the overlap contains
        self.fftSize = 512 #number of frequency bins in the ftt spectrogram
        self.numMelBands = 16 #number of mel frequency bands for the mel spectrogram
        #todo: you might want to print this lot out at the start of each run?
        print("sampleRate=",self.sampleRate)
        print("windowSeconds=",self.windowSeconds)
        print("windowSamples=",self.windowSamples)
        print("windowOverlap=",self.windowOverlap)
        print("windowSampleOverlap=",self.windowSampleOverlap)
        print("fftSize=",self.fftSize)
        print("numMelBands=",self.numMelBands)
        self.spectrogramDBFS = []
        self.spectrogramMag = []
        self.freq=[] #frequency bands to go with above (Hz)

################################################################################

    def load(self,audiofilename):
        #re-initialise data
        self.spectrogramDBFS = []
        self.spectrogramMag = []
        self.freq=[] #frequency bands to go with above (Hz)
        #load audio and process
        data, datasamplerate = sf.read(audiofilename)
        datalen = len(data)
        print("data length=",datalen," sample rate=",datasamplerate)

        #now go through each block in turn
        n=0
        while (n<datalen):
            window = data[n:n+self.windowSamples] #if this runs off the end then we pad with zeroes
            N = len(window)
            if N<self.windowSamples:
                window = np.pad(window, pad_width=(0, self.windowSamples-N), mode='constant')
            #plotSignal(window)


            #perform RMS check on data here for frames which are silent...
        
            #compute spectrogram in db relative to full scale
            freq, spec, specmag = self.dbfsfft(window)
            #spec_db = spec+120 #scale dbfs to db
            self.freq=freq
            self.spectrogramDBFS.append(spec) #this is the log db relative full scale version (normal one plotted)
            self.spectrogramMag.append(specmag) #this is the raw magnitude spectrum directly off the fft
            
            n=n+self.windowSamples-self.windowSampleOverlap

################################################################################

    """
    Hamming Window: w(n)=0.54-0.46 cos (2PI*n/N), 0<=n<=N
    NOTE: apparently, numpy has np.hamming(N) function to do exactly this
    @param N number of samples to create window for
    @returns a list of Hamming window weights with N values in it
    """
    def hamming(self,N):
        w = []
        for n in range(N):
            w.append(0.54-0.46*cos(2*pi*n/N))
        return w

################################################################################        


    def dbfsfft(self,x):
        """
        Compute spectrogram (fft real) for this time window. Uses Hamming window function.
        @param x signal
        @param fftsize number of bands in fft e.g. 512
        @param sampleRate rate signal was sampled at, required to calculate the frequency bands in hz
        @returns the frequency bands in hz, the db full scale spectrum (log) and the raw magnitude spectrum (lin)
        """
        ref=1.0 #full scale reference - would be 65535 for int values
        N = len(x)
        wdw = self.hamming(N)
        #plotSignal(wdw)
        #xrms=rms(x)
        x = x * wdw
        #plotSignal(x)
        spec = np.fft.rfft(x,self.fftSize) #real part only fft.fft would do the img mirror
        freq = np.arange((self.fftSize / 2) + 1) / (float(self.fftSize) / self.sampleRate) #need frequency bins for plotting
        #find the magnitude of the complex numbers in spec
        spec_mag = np.abs(spec)*2/np.sum(wdw) #magnitude scaling by window: np.abs(s) is amplitude spectrum, np.abs(s)**2 is power
        spec_dbfs = 20 * np.log10(spec_mag/ref) #conversion to db rel full scale
        #print "max,min=",np.max(spec_dbfs),np.min(spec_dbfs),np.max(spec_mag),np.min(spec_mag),xrms,np.sum(wdw)

        #print "len spec_dbfs=",len(spec_dbfs)
        return freq, spec_dbfs, spec_mag

################################################################################

    def spectrogramMedianFilter(self,spec):
        """
        Normalise spectrogram power for each frequency band by subtracting the median from each band in turn.
        Any values below zero are set to zero.
        """
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
    
################################################################################

    def hzToMel(self,hz):
        """Convert frequency in Hz into a mel frequency - NOTE: other mel conversion scales exist"""
        #return 2595.0*np.log10(1.0+hz/700.0)
        return 1125.0*np.log(1.0+hz/700.0) #alternative representation

    def melToHz(self,mel):
        #return 700.0*(pow(10,mel/2592.0)-1.0)
        return 700*(np.exp(mel/1125.0)-1.0)

    def triangleWindow(self,N):
        w = []
        L = float(N+1) #could use L=N, L=N+1 or L=N-1, using L=N+1 gives me a triangle window for small N
        for n in range(N):
            w.append(1.0-np.abs((n-(N-1)/2.0)/(L/2.0)))
        return w

    def createMelFilters(self,freq):
        """
        Create a filterbank of triangle filters which is used to convert the regular FFT into a mel
        frequency range. This is done using a bank of window filters (frequency domain) which act on
        the FFT power bands to convert them to mel power bands. In simple terms, the mel bands get
        increasingly spaced out as the frequency increases (log scale).
        @param freq The list of frequency bands in the input fft
        @param num_mel_banks The number of mel power bands that we want to create
        @returns the mel frequency bands AND
        a list of num_mel_banks in length where each item is a list of len(freq) in length
        containing a window function used to create that particular mel frequency power from the
        original fft.
        TODO: the final filter is centred a lot lower than the top frequency - you might want to change this
        or maybe not? You could add a final filter centred on the top frequency that only has a lower part
        of the triangle to it.
        """
        N = len(freq)
        minF = freq[0]
        maxF = freq[N-1]
        minMel = self.hzToMel(minF)
        maxMel = self.hzToMel(maxF)
        #print "minF=",minF," maxF",maxF
        #evenly spaced mel bands NOTE: maxMel+1 so that it includes maxMel as the final value. We need this
        #for the min,max pairs to make the filterbank. It results in len(melBank)=num_mel_bands+1 and similarly
        #for hzBank and hzBanki so the final value in all of them corresponds to maxF (frequency)
        melBank = np.arange(minMel,maxMel+1,(maxMel-minMel)/float(self.numMelBands))
        #print "melBank=",melBank
        #convert melBank back into Hz and then to sample numbers
        hzBank = [self.melToHz(m) for m in melBank]
        #print "hzBank=",hzBank
        #to compute index in fft, i=(hz-minF)/((maxF-minF)/N)
        hzBanki = [np.floor((hz-minF)/((maxF-minF)/N)) for hz in hzBank]
        #print "hzBanki=",hzBanki
    
        #now need "num_mel_bands" mel window filters (triangles) centred on the mel
        #frequencies and overlapping
        melFBank = []
        for m in range(0,self.numMelBands):
            #find min and max sample indices, create window, push to melFBank
            if m==0:
                #initial case, peak on zero with no lower left part
                midi = int(hzBank[m])
                highi = int(hzBanki[m+1])
            else:
                #lowi=hzBanki[m-1]
                midi=int(hzBanki[m])
                highi=int(hzBanki[m+1]) #NOTE: len(hzBanki)=num_mel_bands+1, so this is valid
            #Now make a window containing zeros and a triangle window from the low, mid and high i.
            #The triangle window is then placed into the zero window so it's peak is on midi
            lowi=midi-(highi-midi)
            if lowi<0:
                lowi=0
            #now make a window function of length "fftSize" freq bands, with a triangle filter covering the mel bit
            wdw = np.zeros(N)
            #print "window: lowi=",int(lowi)," midi=",int(midi)," highi=",int(highi)
            Ntri=highi-lowi #number of samples in the tri filter window 
            triwdw = self.triangleWindow(Ntri)
            #print "Ntri=",Ntri,"triwdw=",triwdw
            wdw[lowi:highi] = triwdw
            melFBank.append(wdw)
            #plotSignal(wdw)
        #OK, so at this point melFBank contains the triangle filers (f domain) that we need to transform
        #the fft into a mel fft.
        return melBank, melFBank

    def melSpectrum(self,spec,freq):
        """
        Compute a mel frequency spectrum.
        NOTE: we don't want mel frequency cepstral coefficients (MFCCs), but the actual mel
        frequency power spectrogram. This should work better with the feature learning stage
        that comes next.
        @param freq Frequency bands in spec (Hz)
        @param spec The power spectrogram that we're going to transform
        @param num_mel_bands The number of mel frequency bands to map the frequencies to
        @returns the mel frequency bands and the transformed spectrogram
        """
        #Compute mel spaced filterbank (window). This is to apply to the fft power spectra.
        melFreq, melFBank = self.createMelFilters(freq)
    

        #TO REMEMBER: do you need to normalise the energy in each filter window? They don't all have the
        #same area.

        #apply the filters to all the frames and return data
        mel_spec = []
        xs, ys = np.shape(spec)
        for x in range(0,xs):
            fft=spec[x]
            mel_fft = []
            for y in range(0,self.numMelBands):
                #dot product between mel filter bank and fft
                mel_fft.append(np.dot(melFBank[y],fft))
            mel_spec.append(mel_fft)

        return melFreq, mel_spec

################################################################################

    def logSpectrogram(self,spec):
        """
        Apply a log scaling function to a set of spectrogram frames.
        This is so that I can compare log and lin spectrums.
        @param spec list of frames containing a list of power spectrum magnitudes
        @returns the spectrum with the log function applied to everything
        """
        xs, ys = np.shape(spec)
        logspec=[]
        for x in range(0,xs):
            fft=spec[x]
            logfft = [20*np.log10(v+0.0001) for v in fft]
            logspec.append(logfft)
        return logspec
################################################################################

    """
    Plot a spectrogram using matplotlib.
    @param s is the output of np.fft.fft which contains the imaginary and real parts
    """
    def plotSpectrogram(self,s,freq,filename):
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

################################################################################

