try:
    import cluster_setup
except ImportError:
    pass

import os
import sys
import time
import math
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, optimizers, datasets
import sklearn.metrics

import utils
from capsule.capsule_network import CapsNet
from capsule.sign_capsule_network import SignCapsNet
from capsule.utils import margin_loss
from data.sign import create_sign_data


#
# Hyperparameters and cmd args
#
# Learning hyperparameters
argparser = argparse.ArgumentParser(description="Show limitations of capsule networks")
argparser.add_argument("--batch_size", default=64, type=int, 
  help="Batch size")
argparser.add_argument("--epochs", default=50, type=int, 
  help="Defines the number of epochs to train the network") 
argparser.add_argument("--enable_tf_function", default=True, type=bool, 
  help="Enable tf.function for faster execution")
argparser.add_argument("--use_bias", default=False, type=bool, 
  help="Add a bias term to the preactivation")
argparser.add_argument("--logging", default=False, type=bool, 
  help="Detailed logging")
argparser.add_argument("--learning_rate", default=0.0001, type=float, 
  help="Learning rate of adam")

# Routing properties
argparser.add_argument("--routing", default="rba",
  help="rba, em")

# Dataset properties
argparser.add_argument("--dataset_size", default=4096, type=int, 
  help="Size of training set")

args = argparser.parse_args()


def compute_loss(logits, y):
  """ The loss is the sum of the margin loss and the reconstruction loss 
      as defined in [2], no reconstruciton loss for lines
  """ 
  # Calculate margin loss
  loss = margin_loss(logits, tf.one_hot(y, 2), down_weighting=1.0)
  return tf.reduce_mean(loss)


def compute_accuracy(logits, labels):
  predictions = tf.cast(tf.argmax(tf.nn.softmax(logits), axis=1), tf.int32)
  return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def train(train_ds, test_ds, layers, use_bias):
  """ Train capsule networks mirrored on multiple gpu
  """

  # Initialize
  model = SignCapsNet(routing=args.routing, layers=layers, use_bias=use_bias)
  optimizer = optimizers.Adam(learning_rate=args.learning_rate)
  
  # Function for a single training step
  def train_step(inputs):
    x, y = inputs
    with tf.GradientTape() as tape:
      logits = model(x, y)
      loss = compute_loss(logits, y)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    acc = compute_accuracy(logits, y)

    return loss, acc
  
  def test_step(inputs):
    x, y = inputs
    logits = model(x, y)
    acc = compute_accuracy(logits, y)

    # Ensure correctness
    tf.debugging.check_numerics(logits, message="Found nan in logits.")
    tf.debugging.check_numerics(acc, message="Found nan in acc.")
    tf.debugging.check_numerics(x, message="Found nan in x.")

    return acc

  if args.enable_tf_function:
    train_step = tf.function(train_step)
    test_step = tf.function(test_step)

  ########################################
  # Training
  ########################################
  step = 0
  for epoch in range(args.epochs):
    for data in train_ds:
      loss, acc = train_step(data) 

    # Logging
    if args.logging:
      print("TRAIN | epoch %d: acc=%.2f, loss=%.6f" % 
        (epoch, acc, loss), 
        flush=True)   

  ########################################
  # Evaluate accuracy for all datapoints
  ########################################
  acc = [test_step(data) for data in test_ds]
  return np.mean(acc)


#
# M A I N
#
def main():


  executions = []

  for num_hidden_layers in [4,3,2,1]:
    for num_caps in [30,25,20,15,10]:
      for caps_dim in [18,16,14,12,10]:

        num_runs = 3
        acc_runs = []
        for run in range(num_runs):
          # Load data
          train_ds = create_sign_data(
            batch_size = args.batch_size,
            dataset_size = args.dataset_size)
            
          # Create architecture
          layers = [(num_caps, caps_dim) for _ in range(num_hidden_layers)]

          # Train network
          acc = train(
            train_ds, 
            train_ds,
            layers=layers, 
            use_bias=args.use_bias)
          acc_runs.append(acc)

        # Evaluate solution
        acc_mean=np.mean(acc_runs)
        acc_std=np.std(acc_runs)
        executions.append(acc_mean)
        solved = bool(acc_mean > 0.6)
        print("num_layers=%d, num_caps=%d, caps_dim=%d | acc=%.3f(std=%.3f) | solved = %s" %  
          (num_hidden_layers+1, num_caps, caps_dim, acc_mean, acc_std, solved), flush=True)
  
  #
  # Log results
  #  
  print("\n==========================", flush=True)
  print("Accuracy | Num solved", flush=True)
  print("==========================", flush=True)
  for b in [0.0, 0.6, 0.7, 0.8, 0.9]:
    num_solved = np.sum([1 if e > b else 0 for e in executions])
    log = "> %.2f | %d" % (b, num_solved)
    print(log, flush=True)

    file_name = "experiments/routing_%s_bias_%s.txt" % (args.routing, args.use_bias)
    with open(file_name, 'a') as f:
      f.write("%s\n" % log)

  print("==========================")


       
if __name__ == '__main__':
    main()