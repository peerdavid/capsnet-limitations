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
from capsule.utils import margin_loss
from data.mnist import create_mnist
from data.fashion_mnist import create_fashion_mnist
from data.norb import create_norb
from data.svhn import create_svhn


#
# Hyperparameters and cmd args
#

# Optimizer
argparser = argparse.ArgumentParser(description="Show limitations of capsule networks")
argparser.add_argument("--learning_rate", default=0.0001, type=float, 
  help="Learning rate of adam")
argparser.add_argument("--reconstruction_weight", default=0.00001, type=float, 
  help="Loss of reconstructions")
argparser.add_argument("--log_dir", default="experiments/tmp", 
  help="Log dir for tensorbaord")    
argparser.add_argument("--batch_size", default=128, type=int, 
  help="Batch size of training data")
argparser.add_argument("--enable_tf_function", default=True, type=bool, 
  help="Enable tf.function for faster execution")
argparser.add_argument("--epochs", default=30, type=int, 
  help="Defines the number of epochs to train the network")

# Data
argparser.add_argument("--test", default=True, type=bool, 
  help="Run tests after each epoch?")
argparser.add_argument("--dataset", default="mnist",
  help="mnist, fashion_mnist, svhn, norb")

# Architecture
argparser.add_argument("--use_bias", default=False, type=bool, 
  help="Add a bias term to the preactivation")
argparser.add_argument("--use_reconstruction", default=True, type=bool, 
  help="Use the reconstruction network as regularization loss")
argparser.add_argument("--routing", default="rba",
  help="rba, em")
argparser.add_argument("--layers", default="64,32,32,32,32,10",
  help=", spereated list of layers. Each number represents the number of hidden units except for the first layer the number of channels.")
argparser.add_argument("--dimensions", default="8,12,12,12,12,16",
  help=", spereated list of layers. Each number represents the dimension of the layer.")

# Load hyperparameters from cmd args and update with json file
args = argparser.parse_args()


def compute_loss(logits, y, reconstruction, x):
  """ The loss is the sum of the margin loss and the reconstruction loss
  """ 
  num_classes = tf.shape(logits)[1]

  loss = margin_loss(logits, tf.one_hot(y, num_classes))
  loss = tf.reduce_mean(loss)

  # Calculate reconstruction loss
  if args.use_reconstruction:
    x_1d = tf.keras.layers.Flatten()(x)
    distance = tf.square(reconstruction - x_1d)
    reconstruction_loss = tf.reduce_sum(distance, axis=-1)
    reconstruction_loss = args.reconstruction_weight * tf.reduce_mean(reconstruction_loss)
  else:
    reconstruction_loss = 0

  loss = loss + reconstruction_loss

  return loss, reconstruction_loss


def compute_accuracy(logits, labels):
  predictions = tf.cast(tf.argmax(tf.nn.softmax(logits), axis=1), tf.int32)
  return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))



def train(train_ds, test_ds, class_names):
  """ Train capsule networks mirrored on multiple gpu's
  """

  # Run training for multiple epochs mirrored on multiple gpus
  strategy = tf.distribute.MirroredStrategy()
  num_replicas = strategy.num_replicas_in_sync

  train_ds = strategy.experimental_distribute_dataset(train_ds)
  test_ds = strategy.experimental_distribute_dataset(test_ds)

  # Create a checkpoint directory to store the checkpoints.
  ckpt_dir = os.path.join(args.log_dir, "ckpt/", "ckpt")

  train_writer = tf.summary.create_file_writer("%s/log/train" % args.log_dir)
  test_writer = tf.summary.create_file_writer("%s/log/test" % args.log_dir)

  with strategy.scope():
    model = CapsNet(args)
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # Define metrics 
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    
    # Function for a single training step
    def train_step(inputs):
      x, y = inputs
      with tf.GradientTape() as tape:
        logits, reconstruction, layers = model(x, y)
        loss, _ = compute_loss(logits, y, reconstruction, x)
      
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      acc = compute_accuracy(logits, y)

      return loss, acc, (x, reconstruction)

    # Function for a single test step
    def test_step(inputs):
      x, y = inputs
      logits, reconstruction, _ = model(x, y)
      loss, _ = compute_loss(logits, y, reconstruction, x)
      
      test_loss.update_state(loss)
      acc = compute_accuracy(logits, y)

      pred = tf.math.argmax(logits, axis=1)
      cm = tf.math.confusion_matrix(y, pred, num_classes=10)
      return acc, cm

    # Define functions for distributed training
    def distributed_train_step(dataset_inputs):
      return strategy.run(train_step, args=(dataset_inputs,))

    def distributed_test_step(dataset_inputs):
      return strategy.run(test_step, args=(dataset_inputs, ))
    
    if args.enable_tf_function:
      distributed_train_step = tf.function(distributed_train_step)
      distributed_test_step = tf.function(distributed_test_step)

    # Loop for multiple epochs
    step = 0
    max_acc = 0.0
    for epoch in range(args.epochs):
      ########################################
      # Test
      ########################################
      if args.test:
        cm = np.zeros((10, 10))
        test_acc = []
        for data in test_ds:
          distr_acc, distr_cm = distributed_test_step(data)
          for r in range(num_replicas):
            if num_replicas > 1:
              cm += distr_cm.values[r]
              test_acc.append(distr_acc.values[r].numpy())
            else:
              cm += distr_cm
              test_acc.append(distr_acc)

        # Log test results (for replica 0 only for activation map and reconstruction)
        test_acc = np.mean(test_acc)
        max_acc = test_acc if test_acc > max_acc else max_acc
        figure = utils.plot_confusion_matrix(cm.numpy(), class_names)
        cm_image = utils.plot_to_image(figure)
        print("TEST | epoch %d (%d): acc=%.4f, loss=%.4f" % 
              (epoch, step, test_acc, test_loss.result()), flush=True)  

        with test_writer.as_default(): 
          tf.summary.image("Confusion Matrix", cm_image, step=step)
          tf.summary.scalar("General/Accuracy", test_acc, step=step)
          tf.summary.scalar("General/Loss", test_loss.result(), step=step)
        test_loss.reset_states()
        test_writer.flush()


      ########################################
      # Train
      ########################################
      for data in train_ds:
        start = time.time()
        distr_loss, distr_acc, distr_imgs = distributed_train_step(data)
        train_loss = tf.reduce_mean(distr_loss.values) if num_replicas > 1 else distr_loss
        acc = tf.reduce_mean(distr_acc.values) if num_replicas > 1 else distr_acc

        # Logging
        if step % 100 == 0:
          time_per_step = (time.time()-start) * 1000 / 100
          print("TRAIN | epoch %d (%d): acc=%.4f, loss=%.4f | Time per step[ms]: %.2f" % 
              (epoch, step, acc, train_loss.numpy(), time_per_step), flush=True)     

          # Create some recon tensorboard images (only GPU 0)
          if args.use_reconstruction:
            x = distr_imgs[0].values[0] if num_replicas > 1 else distr_imgs[0]
            recon_x = distr_imgs[1].values[0] if num_replicas > 1 else distr_imgs[1]
            recon_x = tf.reshape(recon_x, [-1, tf.shape(x)[1], tf.shape(x)[2], args.img_depth])  
            x = tf.reshape(x, [-1, tf.shape(x)[1], tf.shape(x)[2], args.img_depth])  
            img = tf.concat([x, recon_x], axis=1)
            with train_writer.as_default():
              tf.summary.image(
                "X & Recon",
                img,
                step=step,
                max_outputs=3,)

          with train_writer.as_default(): 
            # Write scalars
            tf.summary.scalar("General/Accuracy", acc, step=step)
            tf.summary.scalar("General/Loss", train_loss.numpy(), step=step)

          start = time.time()
          train_writer.flush()
        
        step += 1


      ####################
      # Checkpointing
      if epoch % 15 == 0:
        checkpoint.save(ckpt_dir)

    return max_acc


#
# M A I N
#
def main():
  print("\n\n###############################################", flush=True)
  print(args.log_dir, flush=True)
  print("###############################################\n", flush=True)

  # Configurations for cluster
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  for r in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[r], True)

  # Write log folder and arguments
  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

  with open("%s/args.txt" % args.log_dir, "w") as file:
     file.write(json.dumps(vars(args)))

  # Load data
  if args.dataset=="mnist":
    train_ds, test_ds, class_names = create_mnist(args)
  elif args.dataset=="fashion_mnist":
    train_ds, test_ds, class_names = create_fashion_mnist(args)
  elif args.dataset=="norb":
    train_ds, test_ds, class_names = create_norb(args)
  elif args.dataset=="svhn":
    train_ds, test_ds, class_names = create_svhn(args)
  else:
    raise Exception("Unknown datastet %s." % args.dataset)

  # Train capsule network
  acc = train(train_ds, test_ds, class_names)
  with open("experiments/results.txt", 'a') as f:
    f.write("%s;%.5f\n" % (args.log_dir, acc))

       
if __name__ == '__main__':
    main()