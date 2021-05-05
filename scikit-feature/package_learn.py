import skfeature.function.information_theoretical_based.CIFE as CIFE
import numpy as np
import pandas as pd


if __name__ == "__main__":
    path = "~/Downloads/Y4S1/COMP4540/dataset/Ag-quantum.csv"
    data = np.array(pd.read_csv(path, header=0))
    X = data[:, :-1]
    y = data[:, -1]
    features = CIFE.cife(X, y)
    print(features)
