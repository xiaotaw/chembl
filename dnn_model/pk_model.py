# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Aug 2016
# Time Last Updated: Nov 2016
# Addr: Shenzhen, China
# Description: dnn model for pk

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def fcnn_layer(input_tensor, input_dim, output_dim, layer_name,
               wd=False, wd_collection=False,
               keep_prob=0.8, variable_collection=False):
  with tf.name_scope(layer_name):
    weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1.0 / np.sqrt(float(input_dim))), name="weights")
    if wd is not None:
      weight_decay = tf.mul(tf.nn.l2_loss(weights), wd, name="weight_loss")
      tf.add_to_collection(wd_collection, weight_decay)
    biases  = tf.Variable(tf.zeros([output_dim]), name="biases")
    if variable_collection:
      tf.add_to_collection(variable_collection, weights)
      tf.add_to_collection(variable_collection, biases)
    relu = tf.nn.relu(tf.matmul(input_tensor, weights) + biases, name="relu")
    if keep_prob:
      dropout = tf.nn.dropout(relu, keep_prob, name="dropout")
      return dropout
    else:
      return relu

def term_reg(in_layer, in_units=4852, th1_units=4096, th2_units=3072, th3_units=2048, 
                  wd=0.004, keep_prob=0.8):
  th1 = fcnn_layer(in_layer, in_units, th1_units, "term_layer1", wd=wd, wd_collection="term_wd_loss", keep_prob=keep_prob, variable_collection="term")
  th2 = fcnn_layer(th1, th1_units, th2_units, "term_layer2", wd=wd, wd_collection="term_wd_loss", keep_prob=keep_prob, variable_collection="term")
  th3 = fcnn_layer(th2, th2_units, th3_units, "term_layer3", wd=wd, wd_collection="term_wd_loss", keep_prob=keep_prob, variable_collection="term")
  return th3

def branch_reg(branch_name, base_layer, wd=0.004, keep_prob=0.8,
               base_units=2048, bh1_units=2048, bh2_units=1024, out_units = 1):
  var_collection="branch_"+branch_name
  with tf.name_scope(branch_name):
    bh1 = fcnn_layer(base_layer, base_units, bh1_units, "branch_layer1", wd=wd, wd_collection=branch_name+"_wd_loss", keep_prob=keep_prob, variable_collection=var_collection)
    bh2 = fcnn_layer(bh1, bh1_units, bh2_units, "branch_layer2", wd=wd, wd_collection=branch_name+"_wd_loss", keep_prob=keep_prob, variable_collection=var_collection)
    with tf.name_scope("out_relu"):
      weights = tf.Variable(tf.truncated_normal([bh2_units, out_units], stddev=1.0 / np.sqrt(float(bh2_units))), name="weights")
      biases  = tf.Variable(tf.zeros([out_units]), name="biases")
      tf.add_to_collection(var_collection, weights)
      tf.add_to_collection(var_collection, biases)
      out_relu = tf.nn.relu(tf.matmul(bh2, weights) + biases, name="out_relu")
    return out_relu


def term(in_layer, in_units = 9561, th1_units = 8192, th2_units = 6144, th3_units = 4096, 
         wd=0.004, keep_prob=0.8):
  th1 = fcnn_layer(in_layer, in_units, th1_units, "term_layer1", wd=wd, wd_collection="term_wd_loss", keep_prob=keep_prob, variable_collection="term")
  th2 = fcnn_layer(th1, th1_units, th2_units, "term_layer2", wd=wd, wd_collection="term_wd_loss", keep_prob=keep_prob, variable_collection="term")
  th3 = fcnn_layer(th2, th2_units, th3_units, "term_layer3", wd=wd, wd_collection="term_wd_loss", keep_prob=keep_prob, variable_collection="term")
  #th4 = fcnn_layer(th3, th3_units, th4_units, "term_layer4", wd=wd, wd_collection="term_wd_loss", keep_prob=keep_prob, variable_collection="term")
  return th3

def branch(branch_name, base_layer, wd=0.004, keep_prob=0.8,
           base_units = 4096, bh1_units = 4096, bh2_units = 2048, bh3_units = 1024, out_units = 2):
  var_collection="branch_"+branch_name
  with tf.name_scope(branch_name):
    bh1 = fcnn_layer(base_layer, base_units, bh1_units, "branch_layer1", wd=wd, wd_collection=branch_name+"_wd_loss", keep_prob=keep_prob, variable_collection=var_collection)
    bh2 = fcnn_layer(bh1, bh1_units, bh2_units, "branch_layer2", wd=wd, wd_collection=branch_name+"_wd_loss", keep_prob=keep_prob, variable_collection=var_collection)
    bh3 = fcnn_layer(bh2, bh2_units, bh3_units, "branch_layer3", wd=wd, wd_collection=branch_name+"_wd_loss", keep_prob=keep_prob, variable_collection=var_collection)
    with tf.name_scope("softmax_linear"):
      weights = tf.Variable(tf.truncated_normal([bh3_units, out_units], stddev=1.0 / np.sqrt(float(bh3_units))), name="weights")
      biases  = tf.Variable(tf.zeros([out_units]), name="biases")
      tf.add_to_collection(var_collection, weights)
      tf.add_to_collection(var_collection, biases)
      softmax = tf.nn.softmax(tf.matmul(bh3, weights) + biases, name="softmax")
    return softmax

def x_entropy(softmax, labels, loss_name, neg_weight=1):
  with tf.name_scope(loss_name):
    weight = np.array([neg_weight, 1]).astype(np.float32)
    cross_entropy = -tf.reduce_sum(tf.reduce_mean(labels * tf.log(softmax) * weight, reduction_indices=[0]), name="x_entropy")
    return cross_entropy







