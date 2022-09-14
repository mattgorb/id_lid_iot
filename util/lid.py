import numpy as np
import matplotlib.pyplot as plt

def calculate_lid(sorted_distances, k_,train_set=True, save=False):
    exact_matches=0
    k_base=k_
    lids=[]
    for i in sorted_distances:
        if i[0] != 0:
            '''topk = i[ 1:k_base]
            denom = i[k_base]
            lid = -1 / ((1 / len(topk)) * np.sum(np.log(topk[i] / denom) for i in range(len(topk))))'''

            unique, counts = np.unique(i, return_counts=True)
            argsort = np.argsort(unique)

            if unique.shape[0]<=k_base:
                k=unique.shape[0]-1
            else:
                k=k_base
            

            argsort_topk = argsort[:k]
            argsort_denom =  argsort[k]

            topk = unique[argsort_topk]
            denom = unique[argsort_denom]

            lid = -1 / ((1 / len(topk)) * np.sum(np.log(topk[i] / denom) for i in range(len(topk))))
            #print(lid)
            if lid < 100:
                lids.append(lid)
            else:
                lids.append(1000)
        else:
            lids.append(0)

    exact_matches += np.count_nonzero(sorted_distances[:, 0] == 0)
    print('Exact Matches in {} samples: {}'.format( sorted_distances.shape[0], exact_matches))

    if save:
        np.savetxt("results/benign_" + str(train_set) + ".csv", np.array(lids), delimiter=",")
    return lids




def calculate_exactmatch(sorted_distances, train_set=True, save=False):
    exact_matches = 0
    k_base = 20
    lids = []
    for i in sorted_distances:
        if i[0] != 0:
            lids.append(1)
        else:
            lids.append(0)

    exact_matches += np.count_nonzero(sorted_distances[:, 0] == 0)
    print('Exact Matches in {} samples: {}'.format(sorted_distances.shape[0], exact_matches))

    if save:
        np.savetxt("results/benign_" + str(train_set) + ".csv", np.array(lids), delimiter=",")
    return lids



def calculate_weighted_lid(sorted_distances, k_, train_set=True, save=False):
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

            argsort_topk = argsort[:k]
            argsort_denom = argsort[k]

            topk = unique[argsort_topk]
            denom = unique[argsort_denom]

            #lid = -1 / ((1 / len(topk)) * np.sum(np.log(topk[i] / denom) for i in range(len(topk))))

            lid_parts=[]
            for i in topk:
                lid_parts.append(np.log(i/denom))

            lid_parts*=counts[:len(topk)]
            lid_parts=np.sum(lid_parts)

            lid=-1/(1/len(topk)*lid_parts)

            if lid < 100:
                lids.append(lid)
            else:
                lids.append(0)
        else:
            lids.append(0)

    exact_matches += np.count_nonzero(sorted_distances[:, 0] == 0)
    print('Exact Matches in {} samples: {}'.format(sorted_distances.shape[0], exact_matches))

    #print(lids)
    if save:
        np.savetxt("results/benign_" + str(train_set) + ".csv", np.array(lids), delimiter=",")
    return lids