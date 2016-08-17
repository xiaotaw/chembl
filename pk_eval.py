# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: evaluate pk model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import numpy
import tensorflow as tf
import pk_model

def evaluate(dataset_dict, neg_dataset, target_list, out_filename,
             ckpt_dir = "/tmp/train"):
  """ evaluate the model """

  outfile = open(out_filename,'w')
  
  with tf.Graph().as_default():
    
    input_placeholder = tf.placeholder(tf.float32, shape = (None, 2048))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))

    global_step = tf.Variable(0, trainable=False)

    """
    start_learning_rate = 0.1
    decay_step = 16000
    decay_rate = 0.7
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_step, decay_rate)
    """

    # build a Graph that computes the softmax predictions from the
    # inference model.
    base = pk_model.term(input_placeholder, keep_prob=1.0)

    softmax_dict = dict()
    loss_dict = dict()
    accuracy_dict = dict()

    for target in target_list:
      # compute softmax
      softmax_dict[target] = pk_model.branch(target, base, keep_prob=1.0)
      # compute accuracy
      accuracy_dict[target] = pk_model.accuracy(softmax_dict[target], label_placeholder, target)


    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables())

    # Start running operations on the Graph.
    with tf.Session() as sess:
      """
      # Restores variables from checkpoint
      ckpt = tf.train.get_checkpoint_state(ckpt_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/tmp/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return
      """
      file_list = glob.glob(ckpt_dir + '/model.ckpt-*')
      ckpt_list = list()
      for f in file_list:
        global_step = int(f.split('/')[-1].split('-')[-1])
        ckpt_list.append((global_step, f))
      ckpt_list.sort(key = lambda ckpt: ckpt[0])


      print("g_step    TP    FN    TN    FP    SEN    SPE    ACC    MCC")
      # for each checkpoint file, restore the variables, and evaluate the model 
      for i in range(0,len(ckpt_list)):
        saver.restore(sess, ckpt_list[i][1])
        global_step = ckpt_list[i][0]
        for j in range(len(target_list)):
          compds = numpy.vstack([dataset_dict[target_list[j]].compds, neg_dataset.compds])
          labels = numpy.vstack([dataset_dict[target_list[j]].labels, neg_dataset.labels])
          # evaluate the model
          ACC, prediction, label_dense = sess.run(
            [accuracy_dict[target_list[j]], tf.argmax(softmax_dict[target_list[j]], 1), tf.argmax(labels, 1)], 
            feed_dict = {
              input_placeholder: compds,
              label_placeholder: labels})

          # compute performance
          TP,TN,FP,FN,SEN,SPE,MCC = pk_model.compute_performance(label_dense, prediction)

          # print
          format_str = "%6d %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %s"
          outfile.write(format_str % (int(global_step), TP, FN, TN, FP, SEN, SPE, ACC, MCC, target_list[j]))
          outfile.write('\n')
          print(format_str % (int(global_step), TP, FN, TN, FP, SEN, SPE, ACC, MCC, target_list[j]))  


        # neg_dataset





        




  outfile.close()

