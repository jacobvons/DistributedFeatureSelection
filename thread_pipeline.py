import threading
import time
import logging
from scikit_feature.skfeature.function.similarity_based import SPEC
import numpy as np
import random

THREAD_COUNT = 10

recv_buffers = [[]] * THREAD_COUNT
local_data_stores = [np.array()] * THREAD_COUNT

def deploy_sends(id, instructions):
    # Structure of an instruction is as follows
    # (receiver_id, [columns])
    for (receiver_id, columns) in instructions:
        recv_buffers[receiver_id].append(local_data_stores[id][:, [columns]])

def feature_select(id, data):
    print(str(id) + ": " + str(data))
    scores = SPEC.spec(data)
    ft_dict = {}
    for (index, score) in enumerate(scores):
        print(str(id) + ": Score for feature " + str(index) + ": " + str(score))
        ft_dict[index] = score
    selected[id] = ft_dict

if __name__ == '__main__':
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(filename='threading.log', filemode='w', format=format, level=logging.INFO, datefmt="%H:%M:%S")
    threads = []
    selected = [{}] * THREAD_COUNT
    print(len(selected))
    entire_dataset = np.array(100,)
    for i in range(THREAD_COUNT):
        # arr = range(i, 100)
        # threads.append(threading.Thread(target=range_sum, args=(i,arr,)))
        # For now, generate some random data to test the pipeline is working - at future stage should use actual data split

        # Need it in the form of a 2d numpy array
        whole_data = []
        for j in range(100):
            col = []
            for k in range(20):
                col.append(random.randint(0, 100))
            whole_data.append(col)
        npArray = np.array(whole_data)
        if i == 0:
            entire_dataset = npArray
        else:
            entire_dataset = np.concatenate((npArray, entire_dataset), axis=0)
        print(str(i) + " generates data: " + str(npArray) + " with shape " + str(npArray.shape))
        threads.append(threading.Thread(target=feature_select, args=(i,npArray,5)))
    for thread in threads:
        thread.start()
    logging.info("Main: all threads created, waiting on joins")
    for (index, thread) in enumerate(threads):
        thread.join()
        logging.info("Main: Thread %d joins", index)
    # Now check that these feature selections match up with what we expect
    print(entire_dataset.shape)
    actual_scores = SPEC.spec(entire_dataset)
    for (index, score) in enumerate(actual_scores):
        print("Actual Score for feature " + str(index) + ": " + str(score))
    logging.info("Main: all done")
    
