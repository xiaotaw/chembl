# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Aug 2016
# Time Last Updated: Oct 2016
# Addr: Shenzhen, China
# Description: apply pk model to pubchem dataset, to screen potential active substrate(drugs)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import time
import numpy
import cPickle
import datetime
import tensorflow as tf
from scipy import sparse

import dnn_model


def virtual_screening(target_list, part_num):

  # virtual screen log file
  log_dir = "log_files"
  logpath = os.path.join(log_dir, "virtual_screen_pubchem_%d.log" % part_num)
  logfile = open(logpath, "w")
  logfile.write("virtual screen %d starts at: %s\n" % (part_num, datetime.datetime.now()))

  # input and output dir
  pkl_dir = "/raid/xiaotaw/pubchem/pkl_files"
  prediction_dir = "/raid/xiaotaw/pubchem/prediction_files"
  if not os.path.exists(prediction_dir):
    os.mkdir(prediction_dir)

  # screening
  with tf.Graph().as_default(), tf.device("/gpu:%d" % (part_num // 3)):
    # the input
    input_placeholder = tf.placeholder(tf.float32, shape = (None, 8192))

    # the term
    base = dnn_model.term(input_placeholder, keep_prob=1.0)

    # the branches
    softmax_dict = dict()
    for target in target_list:
      softmax_dict[target] = dnn_model.branch(target, base, keep_prob=1.0)

    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables())

    # Start screen
    prediction_dict = dict()
    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    with tf.Session(config=config) as sess:
      # Restores variables from checkpoint
      saver.restore(sess, "ckpt_files/model.ckpt-40000")

      
      #for i in xrange(1, 121225001, 25000):
      begin_num = part_num * 10000000 + 1
      if part_num == 11:
        end_num = 121225001
      else:
        end_num = (part_num + 1) * 10000000 + 1  

      for i in xrange(begin_num, end_num, 25000):
        start_time = float(time.time())
        # get input compounds
        in_file = "Compound_" + "{:0>9}".format(i) + "_" + "{:0>9}".format(i + 24999) + ".pkl"
        if not os.path.exists(os.path.join(pkl_dir, in_file)):
          logfile.write("%s\t0\tnot exists" % in_file)
          continue
        infile = open(os.path.join(pkl_dir, in_file), "rb")
        data = cPickle.load(infile)
        numpy.clip(data, 0, 1, out=data)
        compds = data.astype(numpy.float32)
        infile.close()
        for target in target_list:
          prediction_dict[target] = sess.run(tf.argmax(softmax_dict[target], 1), feed_dict = {input_placeholder: compds})

        # stack prediction result into a matrix with shape = (num_compds, num_targets)
        prediction = numpy.vstack([prediction_dict[k] for k in target_list]).T
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
        print("%s\t%s\t%d\t%.3f" % (in_file, prediction.sum(axis=0), compds.shape[0], time.time()-start_time))  
  logfile.write("virtual screen %d ends at: %s\n" % (part_num, datetime.datetime.now()))
  logfile.close()


# analyse vs result
def analyse_vs_result():
  prediction_dir = "/raid/xiaotaw/pubchem/prediction_files"
  mgfp_dir = "/raid/xiaotaw/pubchem/morgan_fp"

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
    index, _ = sp.nonzero()
    index = sorted(list(set(index)))
    # get potential hit compounds' prediction result
    result = sp.toarray()[index]

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

    print("%s\t%d" % (pre_file, len(index)))

  results_pre = numpy.vstack(result_list)
  results_cid = numpy.array(cid_list, dtype=numpy.int)
  results = numpy.hstack([results_cid.reshape(len(cid_list), 1), results_pre])

  outfile = open("vs_pubchem.result", "wb")
  cPickle.dump(results, outfile, protocol=2)
  outfile.close()

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

  #virtual_screening(target_list, int(sys.argv[1]))

  




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




























