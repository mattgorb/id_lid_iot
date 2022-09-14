import numpy as np
import matplotlib.pyplot as plt


def calculate_id(sorted_distances, k_=20,train_set=True, ):
    exact_matches = 0
    k_base = k_
    lids = []
    for i in sorted_distances:
        if i[0] != 0:
            unique, counts = np.unique(i, return_counts=True)
            argsort = np.argsort(unique)

            if unique.shape[0] <= k_base:
                k = unique.shape[0] - 1
            else:
                k = k_base

            if k==0:
                lids.append(unique[0])
                continue

            argsort_topk = argsort[:k]
            argsort_denom = argsort[k]

            topk = unique[argsort_topk]
            numer = unique[argsort_denom]

            if k==1:
                lid = 1 / ((1 / (len(topk))) * np.sum(np.log(numer/topk[i]) for i in range(len(topk))))
            else:
                lid = 1 / ((1 / (len(topk)-1)) * np.sum(np.log(numer/topk[i]) for i in range(len(topk))))
            # print(lid)
            if lid < 100:
                lids.append(lid)
            else:
                lids.append(0)
        else:
            lids.append(0)

    exact_matches += np.count_nonzero(sorted_distances[:, 0] == 0)
    #print('Exact Matches in {} samples: {}'.format(sorted_distances.shape[0], exact_matches))


    return lids

