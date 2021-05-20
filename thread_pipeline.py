import threading
import time
import logging
from scikit_feature.skfeature.function.similarity_based import SPEC
import numpy as np
import random

THREAD_COUNT = 10


def thread_ping(name):
    logging.info("Thread %s starts", name)
    time.sleep(2)
    logging.info("Thread %s returns", name)


def feature_select(id, data, h):
    print(str(id) + ": " + str(data))
    scores = SPEC.spec(data)
    ft_dict = {}
    for (index, score) in enumerate(scores):
        print(str(id) + ": Score for feature " + str(index) + ": " + str(score))
        ft_dict[index] = score
    selected[id] = ft_dict
    # ranked_features = SPEC.feature_ranking(scores)
    # selected_inner = ranked_features[0:h]
    # for feature in selected_inner:
        # logging.info("Thread %s selects" + str(feature), id)
        # selected[id].append(feature)
    # return selected_inner


if __name__ == '__main__':
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(filename='threading.log', filemode='w', format=format, level=logging.INFO, datefmt="%H:%M:%S")
    threads = []
    selected = [{}] * THREAD_COUNT
    print(len(selected))



    entire_dataset = np.array(20,)





    for i in range(THREAD_COUNT):
        # arr = range(i, 100)
        # threads.append(threading.Thread(target=range_sum, args=(i,arr,)))
        # For now, generate some random data to test the pipeline is working -
        # at future stage should use actual data split

        # Need it in the form of a 2d numpy array
        whole_data = []
        for j in range(20):
            col = []
            for k in range(100):
                col.append(random.randint(0, 100))
            whole_data.append(col)
        npArray = np.array(whole_data)
        if i == 0:
            entire_dataset = npArray
        else:
            entire_dataset = np.concatenate((npArray, entire_dataset), axis=1)
        print(str(i) + " generates data: " + str(npArray) + " with shape " + str(npArray.shape))
        threads.append(threading.Thread(target=feature_select, args=(i, npArray, 5)))
    for thread in threads:
        thread.start()
    logging.info("Main: all threads created, waiting on joins")
    for (index, thread) in enumerate(threads):
        thread.join()
        logging.info("Main: Thread %d joins", index)
    logging.info("Main: all done")
