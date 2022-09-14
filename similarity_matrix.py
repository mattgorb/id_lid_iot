from pairwise_distances import *
from data_setup import df_to_np, calculate_weights
from util.knn import calculate_knn, all_knn
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

save_knns=False
if save_knns:
    dataset='ton_iot'
    if dataset=='ton_iot':
        from data_preprocess.drop_columns import ton_iot
        benign_np=df_to_np('csv/ton_iot/Train_Test_Network.csv',ton_iot.datatypes, train_set=True)
        mal_np=df_to_np('csv/ton_iot/Train_Test_Network.csv', ton_iot.datatypes,train_set=False)
        #X_train, X_test = train_test_split(benign_np, test_size = 0.01, random_state = 42)
        X_train, X_test =benign_np, benign_np
        feature_weights=calculate_weights(X_train)
    elif dataset=='iot23':
        from data_preprocess.drop_columns import iot23
        benign_np=df_to_np('csv/iot23/CTU-IoT-Malware-Capture-1-1.csv',iot23.datatypes, train_set=True)
        mal_np=df_to_np('csv/iot23/CTU-IoT-Malware-Capture-1-1.csv',iot23.datatypes, train_set=False)
        X_train, X_test =benign_np, benign_np
        feature_weights=calculate_weights(X_train)
        print(benign_np.shape)
        print(mal_np.shape)
    elif dataset=='unsw_nb15':
        from data_preprocess.drop_columns import unsw_n15
        benign_np =df_to_np('csv/unsw-nb15/UNSW_NB15_training-set.csv', unsw_n15.datatypes,train_set=True)
        mal_np=df_to_np('csv/unsw-nb15/UNSW_NB15_training-set.csv',  unsw_n15.datatypes,train_set=False)
        X_train, X_test =benign_np, benign_np

        feature_weights=calculate_weights(X_train)
        #feature_weights=None

        print(X_train.shape)

    print('benign')
    pairwise_distances=batch_distances(benign_np, benign_np, weights=None)
    print(pairwise_distances.shape)

    print('malicious')
    pairwise_distances_malicious=batch_distances(mal_np,benign_np, train_set=False,weights=None)

    knns=all_knn(pairwise_distances)
    knns_mal=all_knn(pairwise_distances_malicious, )




    #np.savetxt("results/knns.csv", np.array(knns), delimiter=",")
    #np.savetxt("results/knns_mal.csv", np.array(knns_mal), delimiter=",")
    np.save("results/knns.npy", np.array(knns))
    np.save("results/knns_mal.npy", np.array(knns_mal))
else:
    #knns=np.loadtxt("results/knns.npy",  delimiter=",")
    #knns_mal=np.loadtxt("results/knns_mal.npy",  delimiter=",")

    knns=np.load("results/knns.npy")
    knns_mal=np.load("results/knns_mal.npy")

    print(knns)
    print(knns_mal.shape)
    print(knns_mal)
    plt.clf()
    sns.heatmap(knns[-20:], cmap="Greens", annot=False, yticklabels=False, xticklabels=False)
    plt.savefig('results/knnMatrix.pdf')

    plt.clf()
    sns.heatmap(knns_mal[-20:], cmap="Greens", annot=False, yticklabels=False, xticklabels=False)
    plt.savefig('results/knnMatrix_mal.pdf')