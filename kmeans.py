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
import sys

#x: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
#centroids: [2,4,7]


class KMeans1D:
    """
    K-Means cluster in 1 dimension. This was basically a means of learning how to use TensorFlow to do something
    simple.
    """
    def __init__(self):
        self._numK = 3
        self._data = [] #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        #self._centroids = [2, 4, 7]

    def setData(self, values):
        """
        :param values: list of the data values
        :return: none
        """
        self._data = values
        
    def chooseRandomCentroids(self, nClusters):
        """
        Return a starting set of clusters drawn randomly from the data.
        NOTE: not using TensorFlow
        NOTE 2: also, how you pick the starting centroids has a large effect on the final clusters, so you can easily
        get the situation where all the numbers are at the low end or at the high end (principally 2 clusters). If
        you then pick starting centroids at 0, 1/2 , 1 (between min...max), then you will probably find that one of the
        three centroids returned is NaN. In other words, nothing was near enough to be allocated to it.
        You could just pick nClusters from the actual data values, but I don't think that would be any better?
        :param nClusters: number of clusters
        :returns: list of cluster centroids containing nClusters elements
        """
        dmin = np.min(self._data)
        dmax = np.max(self._data)
        #pick nClusters between these two limits
        centroids = []
        for n in range(nClusters):
            #self._centroids.append(np.random.random((dmin,dmax)))
            centroids.append(n*(dmax-dmin)/nClusters+dmin)  # space centroids evenly between min and max

        return centroids


    def graph_compute_nearest_indices(self, x, c):
        """
        :param x: the data tensor (constant)
        :param c: the centroids tensor (placeholder)
        :return: the computation graph which works out the nearest clusters to points
        """
        expanded_vectors = tf.expand_dims(x, 0)
        expanded_centroids = tf.expand_dims(c, 1)
        deltas = tf.subtract(expanded_vectors, expanded_centroids)
        distances = tf.square(deltas)
        nearest_indices = tf.argmin(distances, 0)
        return nearest_indices

    def graph_compute_new_centroids(self, k, x, nearest_indices):
        """
        :param k: number of clusters
        :param x: tensor containing data
        :param nearest_indices: operation to compute the index of the nearest centroid to each x
        :return:
        """
        inearest_indices = tf.to_int32(nearest_indices)
        partitions = tf.dynamic_partition(x, inearest_indices, k)
        compute_new_centroids = tf.reshape([tf.reduce_mean(partition, 0) for partition in partitions], (3,))
        return compute_new_centroids

    def graph_compute_change(self, c_existing, c_new):
        """
        compute difference between last centroids and new centroids to see if there has been a change
        :param c_existing: previous centroids
        :param c_new: new centroids
        :return: the mean square error
        """
        compute_change = tf.reduce_mean(tf.square(tf.subtract(c_existing, c_new)))
        return compute_change

    def cluster(self):
        """
        Perform clustering using self._numK (number of clusters) and self._data
        :return: list of k cluster centroids
        """
        x = tf.constant(self._data, name="x-points")
        c = tf.placeholder(tf.float32, name="c-centroids")
        c_new = tf.placeholder(tf.float32, name="new-centroids")
        g_compute_nearest_indices = self.graph_compute_nearest_indices(x,c)
        g_compute_new_centroids = self.graph_compute_new_centroids(self._numK, x, g_compute_nearest_indices)
        g_compute_change = self.graph_compute_change(c, c_new)

        centroids = self.chooseRandomCentroids(self._numK)  # picks starting centroids
        print("starting centroids: ",centroids)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            writer = tf.summary.FileWriter("output", session.graph)
            session.run(init)
            delta = sys.float_info.max
            while delta > 0.01:
                result_1 = session.run(g_compute_new_centroids, {c: centroids})
                # result_1 contains the new centroids, we now need to compare the difference to see if the clusters are stable
                delta = session.run(g_compute_change, {c: centroids, c_new: result_1})
                centroids = result_1
                print("delta:", delta, " c:", centroids)
            writer.close()

        return centroids
