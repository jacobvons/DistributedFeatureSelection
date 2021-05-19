import numpy as np
import pandas as pd
import random
import os


DATA_DIR = "../Swarm_Behavior_Data"
file_names = [os.path.join(DATA_DIR, n) for n in os.listdir("../Swarm_Behavior_Data") if n.endswith(".csv")]

# Hyper parameters
N = 500  # Number of rows grouped together
M = 50  # Number of columns grouped together
F = None  # Feature Selection algorithm
H = 3  # Number of features to be selected each time
MAT_ITER = 50  # Maximum number of iterations

# Initialisation
original_header = None
original_features = None
original_y = None
split_folder = "../split"
split_name = "Aligned"
random_state = None

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
                    outf.write(",".join(list(v_group)) + y)

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
                with open(os.path.join(split_folder, f"{split_name}_{h_group}_{m}.csv"), "a") as outf:  # Append to old
                    outf.write(",".join(list(v_group)) + y)
