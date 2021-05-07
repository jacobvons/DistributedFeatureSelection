"""
This file contains core functions for the iterative converge process
"""


def get_top_h(feature_dicts: list, h: int) -> list:
    """
    From all features from all nodes, choose top h features according to their importance value
    and output the features
    :param feature_dicts: list, contains feature dicts from all nodes with feature name being the key and
    feature "importance value" being the value
    :param h: int, number of features we want to select
    :return: list, a list of features with highest importance value among all
    """
    all_features_dict = {}
    for fd in feature_dicts:
        for f_name, v in fd.items():
            if f_name in all_features_dict.keys():
                all_features_dict[f_name] += v
            else:
                all_features_dict[f_name] = v

