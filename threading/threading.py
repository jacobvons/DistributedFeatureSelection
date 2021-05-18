import threading
import time
import logging
from scikit_feature.skfeature.function.similarity_based import SPEC

THREAD_COUNT = 10

def thread_ping(name):
    logging.info("Thread %s starts", name)
    time.sleep(2)
    logging.info("Thread %s returns", name)

def range_sum(index, arr):
    logging.info("Thread %s starts", index)
    my_sum = 0
    for j in arr:
        my_sum += j
    sums[index] = my_sum
    logging.info("Thread %s returns a sum of %d", index, my_sum)

def feature_select(id, data, h):
    scores = SPEC.spec(data)
    ranked_features = SPEC.feature_ranking(scores)
    selected = ranked_features[0:h]
    for feature in selected:
        logging.info("Thread %s selects" + str(feature), index)
    return selected

if __name__ == '__main__':
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(filename='threading.log', filemode='w', format=format, level=logging.INFO, datefmt="%H:%M:%S")
    threads = []
    sums = [0] * THREAD_COUNT
    for i in range(THREAD_COUNT):
        arr = range(i, 100)
        threads.append(threading.Thread(target=range_sum, args=(i,arr,)))
    for thread in threads:
        thread.start()
    logging.info("Main: all threads created, waiting on joins")
    for (index, thread) in enumerate(threads):
        thread.join()
        logging.info("Main: Thread %d joins, with a sum of %d in its array slot", index, sums[index])
    logging.info("Main: all done")
    