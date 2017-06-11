#!/usr/bin/env python3

#k-means clustering using tensorflow.
#1 dimensional
#http://learningtensorflow.com/lesson6/
#https://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/

#tf.constant
#tf.variable
#tf.placeholder
#tf.fill
#tf.argmin
#tf.expand_dims
#tf.reduce_sum
#tf.square
#tf.subtract
#tf.div
#tf.to_int32
#tf.dynamic_partition
#tf.concat
#tf.reduce_mean
#tf.gather
#tf.slice
#tf.shape
#tf.random_shuffle
#tf.random_normal


import tensorflow as tf
import numpy as np

class kmeans1d:
    def __init__(self):
        #stuff here
        self._data = np.Array()
        self._centroids = []

    #def add(value):
    #    """Add a data value to the current list"""
    #    self._data.append(value)

    def setData(values):
        self._data = np.Array(values)
        
    def chooseRandomCentroids(nClusters):
        """Return a starting set of clusters drawn randomly from the data"""
        dmin = np.min(self._data)
        dmax = np.max(self._data)
        #pick nClusters between these two limits
        self._centroids=[]
        for n in range(nClusters):
            self._centroids.append(np.random.random((dmin,dmax)))

    #assignToNearest
    #updateCentroids

    def cluster(self):
        model = tf.global_variables_initializer()
        with tf.session() as session:
            
            sample_values = session.run(samples)
            updated_centroid_value = session.run(updated_centroids)
            print(updated_centroid_value)
