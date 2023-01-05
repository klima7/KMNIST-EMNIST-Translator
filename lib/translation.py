import numpy as np


def match_clusters(sizes_from, sizes_to):
    indices_from = np.flip(np.argsort(sizes_from))
    indices_to = np.flip(np.argsort(sizes_to))

    mapping = {}

    for i in range(len(sizes_from)):
        index_from = int(indices_from[i])

        if i < len(indices_to):
            index_to = indices_to[i]
            mapping[index_from] = int(indices_to[index_to])
        else:
            mapping[index_from] = int(indices_to[-1])

    return mapping
