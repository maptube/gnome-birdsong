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

def buildAllTrainingVectors(\
        inFileTrainManifest,inDirData,outDirSpectrogram,outDirTrainingVectors,\
        outTrainingVectorsFilename,frameWidth,frameOverlap\
):
    """
    Build spectrogram files and training vectors for every training file in the manifest
    :param inFileTrainManifest: file containing the list of seno canto flac files with species data (tab separated)
    :param inDirData: directory containing the original xeno canto flac files
    :param outDirSpectrogram: directory where the spectrogram images will be stored
    :param outDirTrainingVectors: directory where the intermediate training vectors are stored (pkl spectrogram files)
    :param outTrainingVectorsFilename: csv file which holds the final training vectors
    :param frameWidth: number of spectrogram frames to bundle into a single training vector i.e. time slice to match
    :param frameOverlap: number of samples to overlap spectrogram frames (probably =frameWidth/2 for 50% overlap)
    :return:
    """
    birdNameToId = loadSpeciesClassification()

    # go through the training manifest and generate spectrograms for all the files
    # id	gen	sp	en	rec	cnt	lat	lng	type	lic
    # 132608	Acanthis	flammea	Common Redpoll	Jarek Matusiak	Poland	50.7932	15.4995	female, male, song	http://creativecommons.org/licenses/by-nc-sa/3.0/
    # Results in [ id="132608", gen="Acanthis", sp="flammea", en="Common Redpoll", rec="Jarek Matusiak",
    #                   cnt="Poland", lat="50.7932", lng="15.4995", type="female, male, song",
    #                   lic="http://creativecommons.org/licenses/by-nc-sa/3.0/" ]
    outCSV = open(outTrainingVectorsFilename, "w")
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
            spectrogramDBFS, spectrogramMag, freq = \
            buildXCSpectrogram(filename,fileid,speciesid,outDirSpectrogram,outDirTrainingVectors)
            #then the vectors
            c1, c2 = writeXCTrainingVectors(frameWidth,frameOverlap,spectrogramDBFS,speciesid,fileid,outCSV)
            # c1=num spectrograms greater than noise, c2=num lines written to csv

            count=count+1

    outCSV.close()


###############################################################################


def buildXCSpectrogram(filename,fileid,speciesid,outDirSpectrogram,outDirTrainingVectors):
    """
    Build spectrogram data for given file
    :param filename:
    :param fileid
    :param speciesid:
    :param outDirSpectrogram:
    :param outDirTrainingVectors:
    :return:
    """
    
    genSpec = Spectrogram()
    #pick up audio file, make the spectrogram from it and save spec, plus serialise the data for training
    genSpec.load(filename)
    #spectrogram computation on entire waveform file
    spectrogramDBFS = genSpec.spectrogramDBFS #this is the db relative full scale power spectra
    spectrogramMag = genSpec.spectrogramMag #and this is the raw linear power spectra
    freq=genSpec.freq #these are the frequency bands that go with the above
    print("spec check 1: ",np.min(spectrogramDBFS[0]),np.max(spectrogramDBFS[0]))
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

    #serialise the data from the spectrogram here so we can go back to it quickly later if needed
            
    #Save vector data needed for learning
    output = open(os.path.join(outDirTrainingVectors,speciesid+'_xc'+fileid+'_dbfs.pkl'), 'wb')
    pickle.dump(spectrogramDBFS, output)
    output.close()
    output = open(os.path.join(outDirTrainingVectors,speciesid+'_xc'+fileid+'_mag.pkl'), 'wb')
    pickle.dump(spectrogramMag, output)
    output.close()
    output = open(os.path.join(outDirTrainingVectors,speciesid+'_xc'+fileid+'_freq.pkl'), 'wb')
    pickle.dump(freq, output)
    output.close()

    print("spec check 2: ", np.min(spectrogramDBFS[0]), np.max(spectrogramDBFS[0]))

    return spectrogramDBFS, spectrogramMag, freq

###############################################################################

def writeXCTrainingVectors(frameWidth,frameOverlap,spectrogram,target,xcid,outCSV):
    """
    Takes the dumps of the spectrogram files for each sample and turns them into a set of training vectors by
    writing the output to a csv file.
    Pattern is to go through all spectrogram frames and calculate a power value. Then go through the frames again
    and only generate vectors for frames that start with a non-silence frame (from the power). This method could be
    improved upon. Multiple CSV lines are written out for each spectrogram, comprising blocks of frames (frameWidth)
    overlapping by (frameOverlap) sample with high enough power. Output is to outCSV.
    :param frameWidth: each training frame is "frameWidth" (~=10?) samples (~20ms*10)
    :param frameOverlap: overlap by this many samples (~=5? which would be 50% overlap)
    :param spectogram:
    :param target: target value for the vector i.e. the bird species number relating to this spectrogram
    :param: xcid: xenocanto identifier (string)
    :param outCSV: the file handle to write the training vector output to as csv lines
    :return: number of frames over the power threshold and number of target vector blocks written to outCSV
    """
    # filenames are 0_xc41428_[dbfs|mag|freq].pkl
    # The zero is the species classification
    # The xc... is the xeno canto identifier
    # dbfs is a bd relative full scale spectrogram, while mag is the magnitude (linear)
    # freq is the list of frequencies in the spectrogram
    # NOTE the switching around of the power threshold check for dbfs compared to the linear mag scale. It might be
    # better to simply add 120dB to the dbfs one to get it the right way around?

    km = KMeans1D()
    km._numK = 3
    #do a 3 way rms cluster
    rms = []
    for spec_frame in spectrogram:
        #calculate power for the spectral frame TODO: what about negative dbs?
        #pwr = sqrt(np.mean(np.square(spec_frame)))
        #pwr2 = calcRMS(spec_frame) # you can use this function too: pwr==pwr2
        #print(np.min(spec_frame),np.max(spec_frame))
        pwr = np.sum(spec_frame) # alternative power calculation - todo: need to lookup theory
        rms.append(pwr)
    km.setData(rms)
    cluster = km.cluster()
    #now go back and see how many of the frames are over the noise threshold we just calculated
    #NOTE: if using dbfs, the raw values will have had 120dB added to them so that the higher ones have
    #more power where dbfs would normally have its max at zero and everything negative which would mean
    #a reversed threshold test
    noiseThreshold = (cluster[0]+cluster[1])/2.0
    count = 0
    for p in rms:
        if p>noiseThreshold:
            count = count+1
            #print(filename," power cluster: ",cluster," data frames: ",count,len(rms))
    #now make some data, each vector is (257 fft)*frameWidth wide, plus the target of course
    count2=0
    i = 0
    while i+frameWidth<len(spectrogram):
        if rms[i]>noiseThreshold:
            csv = target+","+xcid+","+str(i)
            for i2 in range(i,i+frameWidth):
                spec_frame = spectrogram[i2]
                for pwr in spec_frame:
                    csv = csv + "," + str(pwr)
            #print(csv)
            outCSV.write(csv+"\n")
            count2=count2+1
        i = i + frameWidth - frameOverlap
    outCSV.flush()
    return count, count2

###############################################################################

def loadPickleTrainingVectors(filename,target):
    """
    TODO:
    Load spectrogram data from the pickle file for training
    :param filename:
    :return:
    """
    spec = pickle.load(open(filename, "rb"))
    #todo: here you need the bit from writexctrainingvectors to do the frames and noise level checks
    #then build the data into a training structure and return it
    return spec


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
    #buildXCSpectrograms(inFileTrainManifest,inDirData,outDirSpectrogram,outDirTrainingVectors)
    # then build a csv file of the spectrograms cut into frame sized pieces and labelled with the target
    #buildXCTrainingVectors(specFrameWidth, specFrameOverlap, outDirTrainingVectors, trainingVectorsFilename)
    #buildAllTrainingVectors(\
    #    inFileTrainManifest,inDirData,outDirSpectrogram,outDirTrainingVectors,\
    #    trainingVectorsFilename,specFrameWidth,specFrameOverlap\
    #)

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

    #finally, the learning network
    #as input, 257*10 spectrogram block
    x = tf.placeholder(tf.float32, shape=[None, 2570]) #input 257*10 "None" means no shape size, which is our batch dimension
    y_ = tf.placeholder(tf.float32, shape=[None, 88]) #88 bird classes
    #weights and biases
    W = tf.Variable(tf.zeros([2570, 88]))
    b = tf.Variable(tf.zeros([88]))

    conv1 = tf.nn.conv1d(value,filters,stride,padding,use_cudnn_on_gpu=None,data_format=None,name='conv1')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')


###############################################################################
if __name__ == "__main__":
    main()
