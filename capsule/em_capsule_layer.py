

import math
import numpy as np
import tensorflow as tf
from capsule.utils import squash

layers = tf.keras.layers
models = tf.keras.models


class EMCapsule(tf.keras.Model):

    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, stdev=0.2, routing_iterations=2, name='', use_bias=True):
        super(EMCapsule, self).__init__(name=name)
        self.use_bias = use_bias
        self.in_capsules = in_capsules
        self.in_dim = in_dim
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.routing_iterations = routing_iterations

        with tf.name_scope(self.name):
            w_init = tf.random_normal_initializer(stddev=stdev)
            self.W = tf.Variable(name="W", initial_value=w_init(shape=(1, out_capsules, in_capsules, out_dim, in_dim),
                                                    dtype='float32'), trainable=True)
            
            beta_a_init = tf.random_normal_initializer(stddev=stdev)
            self.beta_a = tf.Variable(name="beta_a", initial_value=beta_a_init(shape=(1, out_capsules),
                                                    dtype='float32'), trainable=True)

            beta_u_init = tf.random_normal_initializer(stddev=stdev)
            self.beta_u = tf.Variable(name="beta_u", initial_value=beta_u_init(shape=(1, out_capsules, 1),
                                                    dtype='float32'), trainable=True)

            if self.use_bias:
                bias_init = tf.constant_initializer(0.1)
                self.bias = tf.Variable(name="bias", initial_value=bias_init(shape=(1, out_capsules, out_dim),
                                                    dtype='float32'), trainable=True)
    
    def call(self, u):
        epsilon = 1e-8
        batch_size = tf.shape(u)[0]

        a_i, V_ij = self.extract_a_v(u)
        rr = tf.ones(shape=[batch_size, self.out_capsules, self.in_capsules]) * 1 / self.out_capsules
        final_lambda = 0.01

        for i in range(self.routing_iterations):
            inv_temp = (final_lambda * 
                       (1 - tf.pow(0.95, tf.cast(i + 1, tf.float32))))
            mean_j, stdv_j, a_j = self.m_step(rr, V_ij, a_i, inv_temp)
            if(i < self.routing_iterations-1):
                rr = self.e_step(mean_j, stdv_j, a_j, V_ij)

        a_tiled = tf.expand_dims(a_j, axis=2)
        a_tiled = tf.tile(a_tiled, [1, 1, self.out_dim])
        mean_j = a_tiled * mean_j / (tf.norm(mean_j + epsilon, axis=-1, keepdims=True) + epsilon)
        return mean_j
  

    def m_step(self, R, V_ij, a_i, inv_temp):
        epsilon = 1e-8
        rr = tf.identity(R)
        batch_size = tf.shape(V_ij)[0]

        a_i_tiled = tf.expand_dims(a_i, axis=1)
        a_i_tiled = tf.tile(a_i_tiled, [1, self.out_capsules, 1])
        rr = rr * a_i_tiled
        
        rr_h = tf.expand_dims(rr, axis=-1)
        rr_h = tf.tile(rr_h, [1, 1, 1, self.out_dim])
        rr_sum = tf.reduce_sum(rr, axis=-1)
        rr_sum = tf.expand_dims(rr_sum, axis=-1)
        rr_sum = tf.tile(rr_sum, [1, 1, self.out_dim])
        mean_j = tf.reduce_sum(rr_h * V_ij, axis=2) / (rr_sum + epsilon)

        mean_ij = tf.expand_dims(mean_j, axis=2)
        mean_ij = tf.tile(mean_ij, [1, 1, self.in_capsules, 1])
        var_j = tf.reduce_sum(rr_h * tf.square(V_ij - mean_ij), axis=2)
        var_j = var_j / (rr_sum + epsilon) + 1e-4

        # Use variance instead of standard deviation, because the sqrt seems to 
        # cause NaN gradients during backprop as proposed by (Gritzman, 2019).
        stdv_j = var_j #tf.sqrt(var_j)

        beta_a_tiled = tf.tile(self.beta_a, [batch_size, 1])
        beta_u_tiled = tf.tile(self.beta_u, [batch_size, 1, self.out_dim])
        
        cost_j_h = (beta_u_tiled + tf.math.log(stdv_j)) * rr_sum
        cost_j = tf.reduce_sum(cost_j_h, axis=-1)
        a_j = tf.sigmoid(inv_temp * (beta_a_tiled - cost_j))

        return mean_j, stdv_j, a_j


    def e_step(self, mean_j, stdv_j, a_j, V_ij):
        epsilon = 1e-8

        mean_ij = tf.expand_dims(mean_j, axis=2)
        mean_ij = tf.tile(mean_ij, [1, 1, self.in_capsules, 1])

        stdv_ij = tf.expand_dims(stdv_j, axis=2)
        stdv_ij = tf.tile(stdv_ij, [1, 1, self.in_capsules, 1])
    
        p_ij_tmp_1 = tf.sqrt(2 * math.pi * tf.reduce_prod(stdv_ij, axis=-1) + epsilon)
        p_ij_tmp_2 = -tf.reduce_sum(tf.square(V_ij - mean_ij) / (2 * stdv_ij + epsilon), axis=-1)
        p_ij = 1 / (p_ij_tmp_1 + epsilon) * tf.exp(p_ij_tmp_2)

        a_ij = tf.expand_dims(a_j, axis=2)
        a_ij = tf.tile(a_ij, [1, 1, self.in_capsules])

        r_ij = (a_ij * p_ij) / (tf.reduce_sum(a_ij * p_ij, axis=1, keepdims=True) + epsilon)

        return r_ij


    def extract_a_v(self, u):
        epsilon = 1e-8
        batch_size = tf.shape(u)[0]

        a_i = tf.norm(u + epsilon, axis=-1)

        uT = tf.expand_dims(u, 1) 
        uT = tf.expand_dims(uT, 3)  
        uT = tf.tile(uT, [1, self.out_capsules, 1, 1, 1])
        uT = tf.tile(uT, [1, 1, 1, self.out_dim, 1])
 
        w = tf.tile(self.W, [batch_size, 1, 1, 1, 1])

        V_ij = tf.reduce_sum(uT * w, axis=-1) + epsilon
        if(self.use_bias):
            bias = tf.expand_dims(self.bias, axis=2)
            bias = tf.tile(bias, [1, 1, self.in_capsules, 1])
            V_ij += bias

        return a_i, V_ij