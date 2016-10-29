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
      batch = self.d.next_train_batch(target, 256, 256*25)
      t1 = time.time()
      #print("%s: %s generate batch with neg_begin=%d %5.3f" % (time.ctime(), self.getName(), self.d.neg.train_begin, t1-t0))
      self.queue.put(batch, block=True, timeout=None)
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
  




"""
# normal
for i in range(0, 10000):
  t0 = time.time()
  compds_batch, labels_batch = d.next_train_batch(target, 256, 256*25)
  t1 = float(time.time())
  if i % 10 == 0 or (i+1) == 10000:
    print("%d %5.3f" % (i, t1-t0))

0 0.851
10 0.815
20 0.762
30 0.787
40 0.811
50 0.805
60 0.795
70 0.812
80 0.766
90 0.755
100 0.842
110 0.831
120 0.819
130 0.765
140 0.817
150 0.746
160 0.810
170 0.790
180 0.775
190 0.812
200 0.648
210 0.685
220 0.794
230 0.686
240 0.573
250 0.691
260 0.698
270 0.778
280 0.767
290 0.841
"""
