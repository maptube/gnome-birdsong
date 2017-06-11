#!/usr/bin/env python3

#pip install PySoundFile
#https://pypi.python.org/pypi/PySoundFile/0.8.1
#pip install matplotlib
#https://matplotlib.org/users/pyplot_tutorial.html
#also lookup melfcc.m

#this is useful: https://dsp.stackexchange.com/questions/32076/fft-to-spectrum-in-decibel

#this is VERY good on MFCC: http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

#PCA Whitening: http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/

#https://pythonprogramming.net/tensorflow-introduction-machine-learning-tutorial/

import numpy as np
#from collections import deque
import soundfile as sf
from math import cos, pi, floor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
#import cPickle
import pickle

import tensorflow as tf

from spectrogram import Spectrogram


###############################################################################

def plotSignal(x):
    """Plot a simple time domain signal"""
    t=np.arange(0,len(x),1)
    plt.plot(t,x)
    plt.xlabel('sample')
    plt.ylabel('x')
    plt.title('Signal')
    plt.grid(True)
    plt.show()

###############################################################################

def rms(x):
    """Compute RMS value using time domain signal in x"""
    rms = np.sqrt(np.mean([x[i]**2 for i in range(len(x))]))
    return rms


###############################################################################

def buildXCTrainingVectors(inDirData,outDirSpectrogram,outDirTrainingVectors):
    #go through the training manifest and generate spectrograms for all the files
    #id	gen	sp	en	rec	cnt	lat	lng	type	lic
    #132608	Acanthis	flammea	Common Redpoll	Jarek Matusiak	Poland	50.7932	15.4995	female, male, song	http://creativecommons.org/licenses/by-nc-sa/3.0/
    #Results in [ id="132608", gen="Acanthis", sp="flammea", en="Common Redpoll", rec="Jarek Matusiak",
    #                   cnt="Poland", lat="50.7932", lng="15.4995", type="female, male, song",
    #                   lic="http://creativecommons.org/licenses/by-nc-sa/3.0/" ]

    birdNameToId = loadSpeciesClassification()
    
    genSpec = Spectrogram()
    with open(inFileTrainManifest) as f:
        next(f) #skip header line
        count=0
        for line in f:
            fields = line.split('\t')
            fileid = fields[0]
            commonName = fields[3]
            speciesid = birdNameToId[commonName]
            filename = os.path.join(inDirData,'xc'+fileid+'.flac')
            print(count," ",commonName," ",speciesid," ",filename)
            sys.stdout.flush() #otherwise it very annoyingly doesn't write anything until after the loop is finished!
            #pick up audio file, make the spectrogram from it and save spec, plus serialise the data for training
            genSpec.load(filename)
            #spectrogram computation on entire waveform file
            spectrogramDBFS = genSpec.spectrogramDBFS #this is the db relative full scale power spectra
            spectrogramMag = genSpec.spectrogramMag #and this is the raw linear power spectra
            freq=genSpec.freq #these are the frequency bands that go with the above
            #now plot the data and save the training frames
            print("spectrogram feature frames: ",len(spectrogramDBFS))
            print("spectrogram=",np.shape(spectrogramDBFS))
            #At this point we have a number of possibilities. There is the spectrumDBFS, which is the DB relative
            #full scale log magnitude power spectrum, or the raw linear magnitude power spectrum which is just from
            #the raw fft data directly (magnitude of the im, re vector). Either of these can be median filtered
            #and then converted to a mel scale.
            genSpec.plotSpectrogram(spectrogramDBFS,freq,os.path.join(outDirSpectrogram,'xc'+fileid+'_spec_dbfs.png'))
            genSpec.plotSpectrogram(spectrogramMag,freq,os.path.join(outDirSpectrogram,'xc'+fileid+'_spec_mag.png'))
            spectrogramDBFS_Filt = genSpec.spectrogramMedianFilter(spectrogramDBFS)
            spectrogramMag_Filt = genSpec.spectrogramMedianFilter(spectrogramMag)
            genSpec.plotSpectrogram(spectrogramDBFS_Filt,freq,os.path.join(outDirSpectrogram,'xc'+fileid+'_spec_dbfs_med.png'))
            genSpec.plotSpectrogram(spectrogramMag_Filt,freq,os.path.join(outDirSpectrogram,'xc'+fileid+'_spec_mag_med.png'))
            #now you can do a mel frequency one off of either spectrogram[Mag|DBFS] or spectrogram[Mag|DBFS]_Filt (median filtered)
            #melfreq, spectrogramMels = genSpec.melSpectrum(spectrogramMag_Filt,freq)
            #print "mels=",np.shape(spectrogramMels)
            #genSpec.plotSpectrogram(spectrogramMels,melfreq,os.path.join(outDirSpectrogram,'xc'+fileid+'_spec_mel_mag_med.png'))
            #spectrogramMelsLog = genSpec.logSpectrogram(spectrogramMels) #this applies a log function to the magnitudes
            #genSpec.plotSpectrogram(spectrogramMelsLog,melfreq,os.path.join(outDirSpectrogram,'xc'+fileid+'_spec_mel_log_mag_med.png'))
            #and rms normalisation
            #and silence removal
            #TODO: need to serialise the data from the spectrogram here so we can learn from it
            
            #Save vector data needed for learning
            #TODO: NEED the training class in the filename here
            output = open(os.path.join(outDirTrainingVectors,speciesid+'_xc'+fileid+'_dbfs.pkl'), 'wb')
            pickle.dump(spectrogramDBFS, output)
            output.close()
            output = open(os.path.join(outDirTrainingVectors,speciesid+'_xc'+fileid+'_mag.pkl'), 'wb')
            pickle.dump(spectrogramMag, output)
            output.close()
            output = open(os.path.join(outDirTrainingVectors,speciesid+'_xc'+fileid+'_freq.pkl'), 'wb')
            pickle.dump(freq, output)
            output.close()
            
            count=count+1

###############################################################################

def loadXCTrainingVectors(dirTrainingVectors):
    #todo:
    #filenames are 0_xc41428_[dbfs|mag|freq].pkl
    #The zero is the species classification
    #The xc... is the xeno canto identifier
    #dbfs is a bd relative full scale spectrogram, while mag is the magnitude (linear)
    #freq is the list of frequencies in the spectrogram
    #returns some sort of structure containing spectrogram segments and target species
    for filename in os.listdir(dirTrainingVectors):
        fields = filename.split('_')
        target = fields[0]
        xcid = fields[1]
        obtype = fields[2]
        if obtype=='dbfs':
            print("Training vector file: ",filename)
            pickle.load(os.path.join(dirTrainingVectors,filename)) #it's a list of spectrogram frames
            #todo: now we need to create vectors from the data in the spectrogram
            #do a 3 way rms cluster

    return ""

###############################################################################

def loadSpeciesClassification():
    """
    Read the 'birdspecies.csv' file (hardcoded - BAD!!!) which contains the lookup
    between the common species name e.g. "Sedge Warbler" and a species id which I've
    defined e.g. 74
    returns a dictionary lookup between name and species id
    """
    birdNameToId = {}
    with open('birdspecies.csv') as f:
        for line in f:
            fields = line.split(',')
            birdId = fields[0]
            commonName = fields[1].rstrip() #they have \n on the end
            birdNameToId[commonName]=birdId
    return birdNameToId


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
    print("sampleRate=",sampleRate)
    print("windowSeconds=",windowSeconds)
    print("windowSamples=",windowSamples)
    print("windowOverlap=",windowOverlap)
    print("windowSampleOverlap=",windowSampleOverlap)
    print("fftSize=",fftSize)
    #print "numMelBands=",numMelBands
    ###

    #housekeeping - where everything is and where it goes
    inFileTrainManifest = "training-data/xccoverbl/xcmeta.csv" #manifest file for all training data sound files
    inDirData = "training-data/xccoverbl/xccoverbl_2014_269" #location of the wave files
    outDirSpectrogram = "spectrograms" #where spectrogram plots can get put so they're all together - basically debug/testing
    outDirTrainingVectors = "training-vectors" #location where data for training is stored


    ###

    #build the training vectors from the Xeno Canto dataset into files that we can load easily
    #buildXCTrainingVectors(inDirData,outDirSpectrogram,outDirTrainingVectors)

    #trainingset = loadXCTrainingVectors(outDirTrainingVectors)

    #learning algorithm...
    #hello = tf.constant('Hello, TensorFlow!')
    #sess = tf.Session()
    #print(sess.run(hello))
    #a=tf.constant(10)
    #b=tf.constant(32)
    #print(sess.run(a+b))
    #sess.close()

    #test1
    #x=tf.constant([1,1,2,2,3,4,5,5,5,6,6,7,7,7,7,7,8,9,10],name="points")
    #y=tf.constant([1,1,2,2,3,4,5,5,5,6,6,7,7,7,7,7,8,9,10],name="points2")
    #dx = tf.expand_dims(x)
    W = tf.Variable([0.3],tf.float32)
    b = tf.Variable([-0.3],tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W*x+b
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model-y)
    loss = tf.reduce_sum(squared_deltas)
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        #centroids=tf.Variable([1,5,7],name="centroids")
        #dist = tf.subtract(x,centroids)
        #distNode=tf.add(x,y)
        #print(session.run(distNode))
        session.run(init)
        #print(session.run(linear_model,{x:[1,2,3,4]}))
        print(session.run(squared_deltas,{x:[1,2,3,4], y:[0,-1,-2,-3]}))
        #result = session.run(loss,{x:[1,2,3,4]},{y:[0,-1,-2,-3]})
        #dir(result)
        

    #some testing
    #need RNN cell
    #tf.nn.dynamic_rnn(
    #    cell,
    #    inputs,
    #    sequence_length=None,
    #    initial_state=None,
    #    dtype=None,
    #    parallel_iterations=None,
    #    swap_memory=False,
    #    time_major=False,
    #    scope=None
    #)

    #https://www.tensorflow.org/get_started/tflearn
    #feature_columns = [tf.contrib.layers.real_valued_column("",dimension=4)]
    #classifier = tf.contrib.learn.DNNClassifier(
    #    feature_columns=feature_columns,
    #    hidden_units=[10,20,10],
    #    n_classes=3,
    #    model_dir="tmp-dnnmodel")
    

    #def get_train_inputs():
    #    x=tf.constant(training_set.data)
    #    y=tf.constant(training_set.target)
    #    return x, y
    

    #classifier.fit(input_fn=get_train_inputs, steps=2000)



###############################################################################
if __name__ == "__main__":
    main()
