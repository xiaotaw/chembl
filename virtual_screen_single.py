#!/usr/bin/python
# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: apply pk model to pubchem dataset, to screen potential active substrate(drugs)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import glob
import time
import numpy
import cPickle
import datetime
import tensorflow as tf
from multiprocessing import Pool
from multiprocessing import Manager
from scipy import sparse

import pk_model


def pre_fetch_data(queue, lock, file_list):
  """ pre load compds data from pkl file into queue
  Args: 
    queue: <class 'multiprocessing.managers.AutoProxy[Queue]'> a queue for data
    lock: <class 'multiprocessing.managers.AcquirerProxy'>
    file_list: <type 'list'> a list of filename, the files are input files(.pkl)
  """
  # pre-define parameters
  pkl_dir = "/raid/xiaotaw/pubchem/pkl_files"

  for name in file_list:
    t0 = time.time()
    with open(os.path.join(pkl_dir, name), "rb") as infile:
      # read and pre-treat data
      data = cPickle.load(infile)
      numpy.clip(data, 0, 1, out=data)
      data = data.astype(numpy.float32)
      infile.close()

      t1 = time.time()
      # put data(and it's filename) into queue and it's filename
      lock.acquire()
      queue.put((name, data))
      lock.release()

      t2 = time.time()
      print("pre_fetch %s\t%.3f\t%.3f" % (name, t1-t0, t2-t1))


def predict(target, queue, lock, result_list):
#def predict(queue, lock, result_list):
  """ restore pk dnn model, calculate softmax and prediction for the compds in the queue 
  Args:
    queue: <class 'multiprocessing.managers.AutoProxy[Queue]'> a queue for compds
    lock: <class 'multiprocessing.managers.AcquirerProxy'>
    result_list: <class 'multiprocessing.managers.ListProxy'>
  """

  # pre-define parameters
  ckpt_dir = "ckpt_files/" + target
  mgfp_dir = "/raid/xiaotaw/pubchem/morgan_fp"
  result_dir = "/raid/xiaotaw/pubchem/result_files/" + target
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  # Restores dnn model from checkpoint
  with tf.Graph().as_default(), tf.device("/gpu: 0"):
    input_placeholder = tf.placeholder(tf.float32, shape = (None, 8192))
    base = pk_model.term(input_placeholder, keep_prob=1.0)
    softmax = pk_model.branch(target, base, keep_prob=1.0)
    saver = tf.train.Saver(tf.trainable_variables())
    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(config=config)
    saver.restore(sess, ckpt_dir + "/model.ckpt-10000")

    # screen compds in queue, and save the cid and result of screened out compds
    while True:

      t0 = time.time()
      # dequeue max wait time: 10 sec 
      name, compds = queue.get(True, timeout=10)
      t1 = time.time()
      # screen
      pred = sess.run(tf.argmax(softmax, 1), feed_dict = {input_placeholder: compds})

      t2 = time.time()
      # save if necessary
      if pred.sum() > 0:

        # get picked out compds' index
        index = list()
        for i, p in enumerate(pred):
          if p.any():
            index.append(i)

        # get cid from mgfp files
        mgfp_file = open(os.path.join(mgfp_dir, name.replace("pkl", "mgfp")), "r")
        lines = mgfp_file.readlines()
        cid_list = [lines[x].split("\t")[0] for x in index]
        cid = numpy.array(cid_list, dtype=pred.dtype)
        pred = pred[index]
        # stack cid and prediction, and save into file
        result = numpy.vstack([cid, pred]).T
        #result_file = open(os.path.join(result_dir, name.replace("pkl", "result")), "w")
        #for r in result:
        #  r.tofile(result_file, sep="\t", format="%d")
        #  result_file.write("\n")
        #result_file.close()
        # append into result_list
        result_list.append(result)

      t3 = time.time()
      print("screen %s\t%s\t%d\t%.3f\t%.3f\t%.3f" % (name, pred.sum(axis=0), 0, t1-t0, t2-t1, t3-t2))  



def virtual_screening_single(target, part_num):

  t_0 = time.time()

  # virtual screen log file
  log_dir = "log_files/" + target
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  logpath = os.path.join(log_dir, "virtual_screen_pubchem_single_%d.log" % part_num)
  logfile = open(logpath, "w")
  logfile.write("virtual screen %d starts at: %s\n" % (part_num, datetime.datetime.now()))

  ckpt_dir = os.path.join("ckpt_files", target)

  # input and output dir
  pkl_dir = "/raid/xiaotaw/pubchem/pkl_files"
  prediction_dir = "/raid/xiaotaw/pubchem/prediction_files/"+target
  if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)

  # screening
  with tf.Graph().as_default(), tf.device("/gpu:%d" % (part_num // 3)):
    # the input
    input_placeholder = tf.placeholder(tf.float32, shape = (None, 8192))

    # the term
    base = pk_model.term(input_placeholder, keep_prob=1.0)

    # the branches
    softmax = pk_model.branch(target, base, keep_prob=1.0)

    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables())

    # Start screen
    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    with tf.Session(config=config) as sess:
      # Restores variables from checkpoint
      saver.restore(sess, ckpt_dir + "/model.ckpt-10000")

      
      #for i in xrange(1, 121225001, 25000):
      begin_num = part_num * 10000000 + 1
      if part_num == 11:
        end_num = 121225001
      else:
        end_num = (part_num + 1) * 10000000 + 1  

      #for i in xrange(begin_num, end_num, 25000):
      for i in xrange(10000001, 12000001, 25000):
        # get input compounds
        in_file = "Compound_" + "{:0>9}".format(i) + "_" + "{:0>9}".format(i + 24999) + ".pkl"
        if not os.path.exists(os.path.join(pkl_dir, in_file)):
          logfile.write("%s\t0\tnot exists" % in_file)
          continue

        t0 = time.time()
        infile = open(os.path.join(pkl_dir, in_file), "rb")
        data = cPickle.load(infile)
        numpy.clip(data, 0, 1, out=data)
        compds = data.astype(numpy.float32)
        infile.close()

        t1 = time.time()
        prediction = sess.run(tf.argmax(softmax, 1), feed_dict = {input_placeholder: compds})

        t2 = time.time()
        logfile.write("%s\t%s\t%d\n" % (in_file, prediction.sum(axis=0), compds.shape[0]))
        # convert into sparse matrix
        if not prediction.sum()==0:
          sparse_prediction = sparse.csr_matrix(prediction)
          # save result into file
          out_file = in_file.replace("pkl", "prediction")
          outfile = open(os.path.join(prediction_dir, out_file), "wb")
          cPickle.dump(sparse_prediction, outfile, protocol=2)
          outfile.close()
          #logfile.write(str(sparse_prediction)+"\n")

        t3 = time.time()
        print("%s\t%s\t%d\t%.3f\t%.3f\t%.3f" % (in_file, prediction.sum(axis=0), compds.shape[0], t1-t0, t2-t1, t3-t2))  
  logfile.write("virtual screen %d ends at: %s\n" % (part_num, datetime.datetime.now()))
  logfile.close()

  print("duration: %.3f" % (time.time() - t_0))


# analyse vs result
def analyse_vs_result(target):
  prediction_dir = "/raid/xiaotaw/pubchem/prediction_files/" + target
  mgfp_dir = "/raid/xiaotaw/pubchem/morgan_fp"

  sum_ = 0

  cid_list = []
  result_list = []
 
  for i in xrange(1, 121225001, 25000):

  #for i in xrange(1, 125001, 25000):

    # load data from prediction file
    pre_file = "Compound_" + "{:0>9}".format(i) + "_" + "{:0>9}".format(i + 24999) + ".prediction"
    pre_filepath = os.path.join(prediction_dir, pre_file)
    if not os.path.exists(pre_filepath):
      continue
    prefile = open(pre_filepath, "rb")
    sp = cPickle.load(prefile)
    prefile.close()

    # get potential hit compounds' index
    _, index = sp.nonzero()
    index = sorted(list(set(index)))
    # get potential hit compounds' prediction result
    result = sp.toarray()[0, index]

    # get potential hit compounds' cids from mgfp file
    mgfp_file = pre_file.replace("prediction", "mgfp") 
    mgfp_filepath = os.path.join(mgfp_dir, mgfp_file)
    mgfpfile = open(mgfp_filepath, "r")
    lines = mgfpfile.readlines()
    mgfpfile.close()
    cid = [lines[x].split("\t")[0] for x in index]
    
    # append each file to 
    cid_list.extend(cid)
    result_list.append(result)
   
    sum_ += len(cid)

    print("%s\t%d" % (pre_file, len(index)))

  results_pre = numpy.hstack(result_list)
  results_cid = numpy.array(cid_list, dtype=numpy.int)
  results = numpy.hstack([results_cid.reshape(len(cid_list), 1), results_pre.reshape(results_pre.shape[0], 1)])

  outfile = open(target + "_vs_pubchem.result", "wb")
  cPickle.dump(results, outfile, protocol=2)
  outfile.close()

  print(sum_)

  return results


  
def get_chembl_pos(target_list):
  mgfp_dir = "data_files/mgfp_files/"
  cid_dir = "data_files/id_files/"
  
  def get_cids(target):
    tmp_list = list()
    infile = open(mgfp_dir + target + ".mgfp6", "r")
    lines = infile.readlines()
    infile.close()
    lines = [x.split("\t") for x in lines]
    infile = open(cid_dir + target + ".cids", "r")
    cids = [x.split("\t")[1] for x in infile.readlines()]
 
    for i in range(len(lines)):
      line = lines[i]
      if line[1] == "1":
        tmp_list.append(cids[i])
    return tmp_list


  pos_cid_dict = dict()
  for target in target_list:
    pos_cid_dict[target] = set(get_cids(target))

  return pos_cid_dict




if __name__ == "__main__":


  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  virtual_screening_single("cdk2", int(sys.argv[1]))
  #analyse_vs_result("cdk2")
  """

  t0 = time.time()

  target = "cdk2"
  process_num = 2
  
  # define multi-process manager and pool
  manager = Manager()
  queue = manager.Queue()
  lock = manager.Lock()
  result_list = manager.list()
  pool = Pool(6)

  # allocate pkl files for each pre-fetch process line
  pkl_list = ["Compound_" + "{:0>9}".format(i) + "_" + "{:0>9}".format(i + 24999) + ".pkl" for i in range(10000001, 12000001, 25000)]
  m = int(math.ceil(len(pkl_list) / float(process_num)))

  for i in range(process_num):
    b = i * m
    e = (i + 1) * m
    if e > len(pkl_list):
      e = len(pkl_list)
    _ = pool.apply_async(pre_fetch_data, args=(queue, lock, pkl_list[b:e])) 

  _ = pool.apply_async(predict, args=(target, queue, lock, result_list))
  #_ = pool.apply_async(predict, args=(queue, lock, result_list))
  pool.close()
  pool.join()

  results = numpy.vstack(result_list)
  outfile = open(target + "_vs_pubchem.result", "wb")
  cPickle.dump(results, outfile, protocol=2)
  outfile.close()

  print("duration: %.3f" % (time.time() - t0))

"""
"""
import virtual_screen_pubchem as vsp
import cPickle

target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

f = open("vs_pubchem.result", "r")
results = cPickle.load(f)
f.close()

pos_cid_dict = vsp.get_chembl_pos(target_list)

# test cdk2
cdk2_vs = [results[i, 0] for i in range(results.shape[0]) if results[i, 1]==1]
vs = set(cdk2_vs)
cdk2_re = [int(x) for x in pos_cid_dict["cdk2"]]
re = set(cdk2_re)
len(list(vs | re))

"""


