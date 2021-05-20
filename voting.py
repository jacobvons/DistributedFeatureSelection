"""
The below codes construct a function which aggregate the features using a voting method
In data partition step, we will partition the data into n subsets and evaluate each subset via some FS algorithm.
Suppose we use CIFE from skfeature, it will return a list of m features for a given data set using CIFE.
We combine the ranking for each subset by voting
"""
import numpy as np


def voting(feature_ranking, total_n):
    """
    Inputs:
    - An n*m array which contains the ranking for all subsets
    where m is the number of the feature we want as the result
    n is the number of subset from data partition
    - Total number of feature (i.e The possible feature index)
    Outputs:
    - An 1*m array which contains the final feature indexs after voting
    Assume the index start from zero
    """

    # Get the number of target features and the number of subsets
    m = feature_ranking.shape[1]
    n = feature_ranking.shape[0]

    # Array to store the vote
    voting_count = np.zeros((1,total_n))

    # Raise the vote for each time one feature appears in the array
    for i in range(n):
        for j in range(m):

            # Feature index
            index = feature_ranking[i,j]

            voting_count[index] += 1

    # Generate the final
    return voting_count.argsort()[::-1][:m+1]
