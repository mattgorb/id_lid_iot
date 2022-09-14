import numpy as np
import matplotlib.pyplot as plt


def calculate_knn(sorted_distances,k_, train_set=True, unique_=True):
    exact_matches = 0
    k_base = k_
    knns = []
    for i in sorted_distances:
        if i[0] != 0:
            '''topk = i[ 1:k_base]
            denom = i[k_base]
            lid = -1 / ((1 / len(topk)) * np.sum(np.log(topk[i] / denom) for i in range(len(topk))))'''

            if unique_:
                unique, counts = np.unique(i, return_counts=True)
                argsort = np.argsort(unique)

                if unique.shape[0] <= k_base:
                    k = unique.shape[0]
                else:
                    k = k_base

                argsort_topk = argsort[:k]
                # argsort_denom = argsort[k]

                topk = unique[argsort_topk]
                # denom = unique[argsort_denom]
            else:
                topk=i[:k_base]

            knn = np.mean(topk)  # -1 / ((1 / len(topk)) * np.sum(np.log(topk[i] / denom) for i in range(len(topk))))

            knns.append(knn)

        else:
            knns.append(0)
    # print(total_distance[:15])
    exact_matches += np.count_nonzero(sorted_distances[:, 0] == 0)
    print('Exact Matches in {} samples: {}'.format(sorted_distances.shape[0], exact_matches))


    return knns


def all_knn(sorted_distances,k_, train_set=True, save=False):
    exact_matches = 0
    k_base = k_
    knns = []
    for i in sorted_distances:
        if i[0] != 0:
            '''topk = i[ 1:k_base]
            denom = i[k_base]
            lid = -1 / ((1 / len(topk)) * np.sum(np.log(topk[i] / denom) for i in range(len(topk))))'''

            unique, counts = np.unique(i, return_counts=True)
            argsort = np.argsort(unique)

            if unique.shape[0] <= k_base:
                continue
                #k = unique.shape[0] - 1
            else:
                k = k_base

            argsort_topk = argsort[:k]
            # argsort_denom = argsort[k]

            topk = unique[argsort_topk]
            # denom = unique[argsort_denom]

            #knn = topk#$np.mean(topk)  # -1 / ((1 / len(topk)) * np.sum(np.log(topk[i] / denom) for i in range(len(topk))))

            knns.append(np.array(topk))

    return np.array(knns)