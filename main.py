from threading import Thread
import os
import pandas as pd
import random
import numpy as np
from scikit_feature.skfeature.function.information_theoretical_based import CIFE
from scikit_feature.skfeature.function.similarity_based import fisher_score
from conv_iter import *
from functools import reduce

# Hyper parameters
N = 500  # Number of rows grouped together
M = 50  # Number of columns grouped together
F = None  # Feature Selection algorithm
H = 3  # Number of features to be selected each time
MAT_ITER = 50  # Maximum number of iterations


# Data Split
def split_data():
    DATA_DIR = "./Swarm_Behavior_Data"
    file_names = [os.path.join(DATA_DIR, n) for n in os.listdir(DATA_DIR) if n.endswith(".csv")]

    # Initialisation
    original_features = None
    split_folder = "./split"
    split_name = "Aligned"
    random_state = None

    # Creating new folder for split data
    if not os.path.exists(split_folder):
        os.mkdir(split_folder)

    with open(file_names[0], "r") as f:
        for i, line in enumerate(f):
            line = line.strip()  # Remove trailing \n
            h_group = i // (N + 1)  # Every N rows belong to a group

            if i == 0:  # Get the header information
                original_header = line.split(",")
                original_features = original_header[:-1]
                original_y = original_header[-1]

            elif i % (N + 1) == 0 or i == 1:  # i = 1, N+1, 2N+2, ...
                # Random permutation and split vertically
                line = line.split(",")
                features = line[:-1]
                y = line[-1]
                features = pd.Series(features, index=original_features)
                random_state = random.randint(0, 10 * M)  # Generate new random state for shuffling
                np.random.seed(random_state)  # Reset the random state for consistency
                features = features.reindex(np.random.permutation(features.index), axis=1)
                cols = features.index

                # Vertically split this row
                for m in range(len(cols) // M):
                    ind_start, ind_end = m * M, (m + 1) * M
                    v_group = features[cols[ind_start: ind_end]]
                    with open(os.path.join(split_folder, f"{split_name}_{h_group}_{m}.csv"), "w") as outf:  # Write new
                        outf.write(",".join(list(v_group.index)) + ",Class\n")
                        outf.write(",".join(list(v_group)) + "," + y + "\n")

            else:  # i = 2, 3, 4, ... , N
                line = line.split(",")
                features = line[:-1]
                y = line[-1]
                features = pd.Series(features, index=original_features)
                np.random.seed(random_state)  # Reset the random state for consistency
                features = features.reindex(np.random.permutation(features.index), axis=1)
                cols = features.index

                # Vertically split this row
                for m in range(len(cols) // M):
                    ind_start, ind_end = m * M, (m + 1) * M
                    v_group = features[cols[ind_start: ind_end]]
                    with open(os.path.join(split_folder, f"{split_name}_{h_group}_{m}.csv"),
                              "a") as outf:  # Append to old
                        outf.write(",".join(list(v_group)) + "," + y + "\n")


SCORES = []
FEATURES = []
ROW_FEATURES = []


def fs_method(arr, headers, n_features):
    X, y = arr[:, :-1], arr[:, -1]
    F, J_CMI, MIfy = CIFE.cife(X, y, n_selected_features=n_features)
    fs = {}
    for ind, feature_ind in enumerate(F):
        feature = headers[feature_ind]
        fs[feature] = J_CMI[ind]  # feature name (str) as key and importance value (float) as value
    FEATURES.append(fs)


file_pref = "./split/Aligned_0_"

file_dir = "./split"
split_files = [f for f in os.listdir(file_dir) if f.endswith(".csv")]
row_group_num = max([int(f.split("_")[1]) for f in split_files])
# for i in range(0, row_group_num+1):  # Production
for i in range(0, 1):  # Testing first row group
    global ROW_FEATURES
    global FEATURES
    for iter_num in range(0, MAT_ITER):
        FEATURES = []
        threads = []
        row_group_i_files = [os.path.join(file_dir, f) for f in split_files if f.startswith(f"Aligned_{i}")]
        for index, file_name in enumerate(row_group_i_files):
            sub_dataset_arr = pd.read_csv(file_name)
            headers = sub_dataset_arr.columns
            sub_dataset_arr = pd.read_csv(file_name).to_numpy()
            thread = Thread(target=fs_method, args=(sub_dataset_arr, headers, H))
            threads.append(thread)

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        top_h = get_top_h(FEATURES, H)
        top_h_values = {}
        all_headers = []

        for feature_name in top_h:
            for file_name in row_group_i_files:
                file_df = pd.read_csv(file_name)
                headers = file_df.columns
                all_headers.append(headers)
                if feature_name in headers:
                    top_h_values[feature_name] = file_df[feature_name].values
                    break

        # Test if converged (or reached iteration maximum)
        # If converged, save the result and continue to next
        if reduce(lambda a, b: a and b, [node_converged(node_fs, top_h) for node_fs in all_headers]) \
                or iter_num == MAT_ITER-1:
            # Save result features
            ROW_FEATURES.append(top_h)
            break
        else:
            for file_name in row_group_i_files:
                sub_dataset_arr = pd.read_csv(file_name)
                for feature_name in top_h:
                    if feature_name not in sub_dataset_arr.columns:
                        sub_dataset_arr = pd.concat([pd.DataFrame(top_h_values[feature_name], columns=[feature_name]),
                                                     sub_dataset_arr], axis=1)
                # sub_dataset_arr.to_csv(file_name, index=False)  # Production
                sub_dataset_arr.to_csv("test.csv", index=False)  # For testing purpose
                break  # For testing only
        break  # For testing only
    break  # For testing only

# ROW_FEATURES now contains all top_h features from all row groups
# Perform voting aggregation (by number of appearance?)
