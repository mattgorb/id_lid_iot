from scipy.spatial.distance import pdist
import numpy as np
import sys
from scipy.spatial.distance import cdist
import sys
import numpy as np
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=sys.maxsize)

'''def categorial_sum(X):
    m = X.shape[0]
    n = X.shape[1]
    D = np.zeros(int(m * (m - 1) / 2), dtype=np.float64)  # corrected dtype
    ind = 0

    for i in range(m):
        for j in range(i+1, m):
            d = 0.0
            for k in range(n):
                if X[i, k] != X[j, k]:
                    d += 1.
            D[ind] = d
            ind += 1

    return D


def pdist_with_categorial(continuous, categorical):
    # calculate the squared distance of the float values
    #distances_squared = pdist(vectors[:, where_float_type], metric='sqeuclidean')
    distances_squared = pdist(continuous, metric='sqeuclidean')
    # sum the number of mismatched categorials and add that to the distances
    # and then take the square root
    return np.sqrt(distances_squared + categorial_sum(categorical))


def batchwise_categorial_sum(X_batch, X):
    matching_distances = cdist(X_batch, X, metric='matching')#*2*X_batch.shape[1]
    return matching_distances'''





def batch_distances(test_data, train_data, batch_size=1000,  weights=None, train_set=True, sort_=True):

    all_distances=[]
    for a in range(0, test_data.shape[0], batch_size):
        sample= test_data[a:a + batch_size, :]
        print(sample.shape)
        print(train_data.shape)
        sys.exit()
        total_distance = cdist(sample, train_data, metric='hamming', w=weights)
        if sort_==False:
            return total_distance

        total_distance = np.sort(total_distance, axis=1)

        if train_set:
            total_distance=total_distance[:,1:]

        all_distances.extend(total_distance)


    return np.array(all_distances)