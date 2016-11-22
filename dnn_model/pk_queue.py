# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Oct 2016
# Time Last Updated: Oct 2016
# Addr: Shenzhen, China
# Description: using multi-thread to load input data and generate batch.

import time

import Queue
import threading

import pk_input as pki



target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr", "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]
target = target_list[0]
d = pki.Datasets(target_list)


# using queue

# producer thread
class Producer(threading.Thread):
  def __init__(self, t_name, d, queue):
    threading.Thread.__init__(self, name=t_name)
    self.queue = queue
    self.d = d
  def run(self):
    for i in range(10):
      t0 = time.time()
      batch = self.d.next_train_batch(target, 100, 200)
      t1 = time.time()
      #print("%s: %s generate batch with neg_begin=%d %5.3f" % (time.ctime(), self.getName(), self.d.neg.train_begin, t1-t0))
      self.queue.put(batch, block=True, timeout=None)
      time.sleep(0.5)
    #print("%s: %s finished!" % (time.ctime(), self.getName()))

# consumer thread
class Consumer(threading.Thread):
  def __init__(self, t_name, queue):
    threading.Thread.__init__(self, name=t_name)
    self.queue = queue
  def run(self):
    while True:
      try:
        t0 = time.time()
        batch = self.queue.get(block=True, timeout=5)
        time.sleep(0.5)
        t1 = time.time()
        #print("%s: %s generate batch %5.3f" % (time.ctime(), self.getName(), t1-t0))
      except:
        #print("%s: %s finished!" % (time.ctime(), self.getName()))
        break


if __name__ == "__main__":
  queue = Queue.Queue(50)
  pro_list = []
  for i in range(10):
    pro_list.append(Producer("Pro%d" % i, d, queue))

  con = Consumer("Con", queue)

  for pro in pro_list:
    pro.start()

  con.start()

  for pro in pro_list:
    pro.join()

  con.join()
  


