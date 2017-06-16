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
from math import cos, pi, floor, sqrt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
#import cPickle
import pickle

import tensorflow as tf

from spectrogram import Spectrogram
from kmeans import KMeans1D


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

def calcRMS(x):
    """Compute RMS value using time domain signal in x"""
    rms = np.sqrt(np.mean([x[i]**2 for i in range(len(x))]))
    return rms


###############################################################################

def buildXCSpectrograms(inFileTrainManifest,inDirData,outDirSpectrogram,outDirTrainingVectors):
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

def buildXCTrainingVectors(frameWidth,frameOverlap,dirTrainingVectors,outTrainingVectorsFilename):
    """
    Takes the dumps of the spectrogram files for each sample and turns them into a set of training vectors
    :param frameWidth: each training frame is "frameWidth" (~=10?) samples (~20ms*10)
    :param frameOverlap: overlap by this many samples (~=5? which would be 50% overlap)
    :param dirTrainingVectors: input directory containing the xeno canto files (spectrograms) to process. This is the
    output from buildXCSpectrograms.
    :return:
    """
    # filenames are 0_xc41428_[dbfs|mag|freq].pkl
    # The zero is the species classification
    # The xc... is the xeno canto identifier
    # dbfs is a bd relative full scale spectrogram, while mag is the magnitude (linear)
    # freq is the list of frequencies in the spectrogram

    outCSV = open(outTrainingVectorsFilename,"w")

    km = KMeans1D()
    km._numK = 3
    for filename in os.listdir(dirTrainingVectors):
        fields = filename.split('_')
        target = fields[0]
        xcid = fields[1]
        obtype = fields[2] # remember that this contains the suffix
        #print("filename: ", filename, target, xcid, obtype)
        if obtype=='dbfs.pkl':
            print("Training vector file: ",filename)
            #print("full name: ",os.path.join(dirTrainingVectors,filename))
            with open(os.path.join(dirTrainingVectors,filename),"rb") as f:
                #NOTE: the latin1 encoding is due to using Python 2 to save the data and Python 3 to load it
                spec = pickle.load(f,encoding="latin1") #it's a list of spectrogram frames
                #do a 3 way rms cluster
                rms = []
                for spec_frame in spec:
                    #calculate power for the spectral frame TODO: what about negative dbs?
                    #pwr = sqrt(np.mean(np.square(spec_frame)))
                    #pwr2 = calcRMS(spec_frame) # you can use this function too: pwr==pwr2
                    pwr = np.sum(spec_frame) # alternative power calculation - todo: need to lookup theory
                    rms.append(pwr)
                km.setData(rms)
                cluster = km.cluster()
                #now go back and see how many of the frames are over the noise threshold we just calculated
                noiseThreshold = (cluster[0]+cluster[1])/2.0
                count = 0
                for p in rms:
                    if p>noiseThreshold:
                        count = count+1
                print(filename," power cluster: ",cluster," data frames: ",count,len(rms))
                #now make some data, each vector is (257 fft)*frameWidth wide, plus the target of course
                i = 0
                while i+frameWidth<len(spec):
                    if rms[i]>noiseThreshold:
                        csv = target+","+xcid+","+str(i)
                        for i2 in range(i,i+frameWidth):
                            spec_frame = spec[i2]
                            for pwr in spec_frame:
                                csv = csv + "," + str(pwr)
                        #print(csv)
                        outCSV.write(csv+"\n")
                    i = i + frameWidth - frameOverlap
        outCSV.flush()
    outCSV.close()

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
    # define constants which determine how the learning works
    # assume sample rate of 44100
    sampleRate = 44100  # this should come from the sample itself really
    windowSeconds = 20.0/1000.0  # window time in seconds
    windowSamples = int(floor(windowSeconds*sampleRate))  # number of samples in a window
    windowOverlap = 0.5  # degree of overlap between sample windows 0.5 means 50% overlap
    windowSampleOverlap = int(floor(windowOverlap*windowSamples))  # how many samples the overlap contains
    fftSize = 512  # number of frequency bins in the ftt spectrogram
    # numMelBands = 16 #number of mel frequency bands for the mel spectrogram
    specFrameWidth = 10  # number of spectrogram frames to batch into a training vector
    specFrameOverlap = 5  # overlap in spectrogram frames for training vector batch =specFrameWidth/2 is 50% overlap
    # todo: you might want to print this lot out at the start of each run?
    print("sampleRate=",sampleRate)
    print("windowSeconds=",windowSeconds)
    print("windowSamples=",windowSamples)
    print("windowOverlap=",windowOverlap)
    print("windowSampleOverlap=",windowSampleOverlap)
    print("fftSize=",fftSize)
    # print "numMelBands=",numMelBands
    print("specFrameWidth=",specFrameWidth)
    print("specFrameOverlap=",specFrameOverlap)
    ###

    # housekeeping - where everything is and where it goes
    inFileTrainManifest = "training-data/xccoverbl/xcmeta.csv"  # manifest file for all training data sound files
    inDirData = "training-data/xccoverbl/xccoverbl_2014_269"  # location of the wave files
    outDirSpectrogram = "spectrograms"  # where spectrogram plots can get put so they're all together - basically debug/testing
    outDirTrainingVectors = "training-vectors"  # location where data for training is stored
    trainingVectorsFilename = os.path.join(outDirTrainingVectors, "vectors.csv")


    ###

    # build the training vectors from the Xeno Canto dataset into files that we can load easily
    # buildXCSpectrograms(inFileTrainManifest,inDirData,outDirSpectrogram,outDirTrainingVectors)
    # then build a csv file of the spectrograms cut into frame sized pieces and labelled with the target
    buildXCTrainingVectors(specFrameWidth, specFrameOverlap, outDirTrainingVectors, trainingVectorsFilename)

    #learning algorithm...
    #hello = tf.constant('Hello, TensorFlow!')
    #sess = tf.Session()
    #print(sess.run(hello))
    #a=tf.constant(10)
    #b=tf.constant(32)
    #print(sess.run(a+b))
    #sess.close()

    #test1
    #numK=3 #number of clusters
    #x = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0], name="x-points")
    #y=tf.constant([1,1,2,2,3,4,5,5,5,6,6,7,7,7,7,7,8,9,10],name="points2")
    #dx = tf.expand_dims(x)
    #c = tf.placeholder(tf.float32, name="c-centroids")
    #c_new = tf.placeholder(tf.float32, name="new-centroids")
    #W = tf.Variable([0.3],tf.float32)
    #b = tf.Variable([-0.3],tf.float32)
    #x = tf.placeholder(tf.float32)
    #linear_model = W*x+b
    #y = tf.placeholder(tf.float32)
    #squared_deltas = tf.square(linear_model-y)
    #loss = tf.reduce_sum(squared_deltas)
    #
    #deltas = tf.squared_difference(x,tf.transpose(c))
    #expanded_vectors = tf.expand_dims(x, 0)
    #debug_expanded_vectors = tf.Print(expanded_vectors,[tf.shape(expanded_vectors)],summarize=100,message="This is me: ")
    #expanded_centroids = tf.expand_dims(c, 1)
    #debug_expanded_centroids = tf.Print(expanded_centroids,[tf.shape(expanded_centroids)],summarize=100,message="This is me: ")
    #deltas = tf.subtract(expanded_vectors, expanded_centroids)
    # debug_deltas = tf.Print(deltas,[tf.shape(deltas)],summarize=100,message="debug_deltas: ")
    #distances = tf.square(deltas)
    #nearest_indices = tf.argmin(distances, 0)
    # debug_mins = tf.Print(mins,[tf.shape(mins)],summarize=100,message="debug_mins: ")
    ##
    #inearest_indices = tf.to_int32(nearest_indices)
    #partitions = tf.dynamic_partition(x, inearest_indices, numK)
    # new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition,0),0) for partition in partitions],0)
    #compute_new_centroids = tf.reshape([tf.reduce_mean(partition, 0) for partition in partitions], (3,))
    ##
    #compute_change = tf.reduce_mean(tf.square(tf.subtract(c, c_new)))
    #
    #centroids = [2,4,7]
    #init = tf.global_variables_initializer()
    #with tf.Session() as session:
    #    writer = tf.summary.FileWriter("output", session.graph)
    #    session.run(init)
    #    delta = sys.float_info.max
    #    while delta > 0.01:
    #        result_1 = session.run(compute_new_centroids, {c: centroids})
    #        # result_1 contains the new centroids, we now need to compare the difference to see if the clusters are stable
    #        delta = session.run(compute_change, {c: centroids, c_new: result_1})
    #        centroids = result_1
    #        print("delta:", delta, " c:", centroids)
    #    writer.close()

    #kmeans class test
    #km = KMeans1D()
    #km._numK = 3
    #km.setData([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
    #km.cluster()



    # some testing
    # need RNN cell
    # tf.nn.dynamic_rnn(
    #    cell,
    #    inputs,
    #    sequence_length=None,
    #    initial_state=None,
    #    dtype=None,
    #    parallel_iterations=None,
    #    swap_memory=False,
    #    time_major=False,
    #    scope=None
    # )

    # https://www.tensorflow.org/get_started/tflearn
    # feature_columns = [tf.contrib.layers.real_valued_column("",dimension=4)]
    # classifier = tf.contrib.learn.DNNClassifier(
    #    feature_columns=feature_columns,
    #    hidden_units=[10,20,10],
    #    n_classes=3,
    #    model_dir="tmp-dnnmodel")

    # def get_train_inputs():
    #    x=tf.constant(training_set.data)
    #    y=tf.constant(training_set.target)
    #    return x, y

    # classifier.fit(input_fn=get_train_inputs, steps=2000)


###############################################################################
if __name__ == "__main__":
    main()
