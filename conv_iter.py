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
        for f_name, value in fd.items():
            if f_name in all_features_dict.keys():
                all_features_dict[f_name] += value
            else:
                all_features_dict[f_name] = value

    all_features_list = [k for k, v in sorted(all_features_dict.items(), key=lambda item: item[1], reverse=True)]
    return all_features_list[0:h]


def node_converged(node_fs: list, new_fs: list) -> bool:
    """
    To check if a node has got all the features in the newly generated feature set

    :param node_fs: list, existing features on a node
    :param new_fs: list, newly aggregated features by all nodes
    :return: bool, True if a node has all features in the new feature list, otherwise False
    """
    return len(node_fs) == len(set(node_fs + new_fs))
