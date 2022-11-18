from pairwise_distances import *
from data_setup import df_to_np, calculate_weights
from util.knn import calculate_knn
from sklearn import metrics

dataset='kaggle_nid'
weights=True
if dataset=='ton_iot':
    from data_preprocess.drop_columns import ton_iot
    benign_np=df_to_np('/s/luffy/b/nobackup/mgorb/iot/ton_iot/Train_Test_Network.csv',ton_iot.datatypes, train_set=True)
    mal_np=df_to_np('/s/luffy/b/nobackup/mgorb/iot/ton_iot/Train_Test_Network.csv', ton_iot.datatypes,train_set=False)
    #X_train, X_test = train_test_split(benign_np, test_size = 0.01, random_state = 42)
    X_train, X_test =benign_np, benign_np
    if weights:
        feature_weights=calculate_weights(X_train)
    else:
        feature_weights=None
elif dataset=='iot23':
    from data_preprocess.drop_columns import iot23

    benign_np=df_to_np('csv/iot23/iot23_sample_with_real.csv',iot23.datatypes,  train_set=True)

    mal_np=df_to_np('csv/iot23/iot23_sample_with_real.csv', iot23.datatypes,train_set=False)
    #X_train, X_test = train_test_split(benign_np, test_size = 0.01, random_state = 42)
    X_train, X_test =benign_np, benign_np
    feature_weights=calculate_weights(X_train)



    print(X_train.shape)
elif dataset=='nf_bot_iot':
    from data_preprocess.drop_columns import nf_bot_iot
    benign_np =df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes,train_set=True)
    mal_np=df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv',  nf_bot_iot.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np

    if weights:
        feature_weights=calculate_weights(X_train)
    else:
        feature_weights=None

    print(X_train.shape)

elif dataset=='unsw_nb15':
    from data_preprocess.drop_columns import unsw_n15
    benign_np , preprocess, float_cols, categorical_cols=df_to_np('/s/luffy/b/nobackup/mgorb/iot/unsw-nb15/UNSW_NB15_training-set.csv', unsw_n15.datatypes,train_set=True, return_preprocess=True)
    benign_np_test , _, _, _=df_to_np('/s/luffy/b/nobackup/mgorb/iot/unsw-nb15/UNSW_NB15_testing-set.csv', unsw_n15.datatypes,train_set=True, return_preprocess=True)

    mal_np=df_to_np('/s/luffy/b/nobackup/mgorb/iot/unsw-nb15/UNSW_NB15_testing-set.csv',  unsw_n15.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np_test

    if weights:
        feature_weights=calculate_weights(X_train)
    else:
        feature_weights=None


elif dataset=='kaggle_nid':
    from data_preprocess.drop_columns import kaggle_nid
    benign_np =df_to_np('/s/luffy/b/nobackup/mgorb/iot/kaggle_nid/Train_data.csv', kaggle_nid.datatypes,train_set=True)
    mal_np=df_to_np('/s/luffy/b/nobackup/mgorb/iot/kaggle_nid/Train_data.csv',  kaggle_nid.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np

    if weights:
        feature_weights=calculate_weights(X_train)
    else:
        feature_weights=None

def save_lids(pairwise_distances,k, file_name):
    knns = calculate_knn(pairwise_distances, k_=k)
    if a==0:
        np.savetxt('results/'+file_name+str(k)+'.txt', knns)
    else:
        with open('results/'+file_name+str(k)+'.txt', "ab") as f:
            np.savetxt(f, knns)


batch_size=1000
#print('total batches dataset/{}={}'.format(batch_size, X_test.shape[0]/batch_size))
for a in range(0, X_test.shape[0], batch_size):
    sample= X_test[a:a + batch_size, :]




    pairwise_distances=batch_distances(sample, X_train, weights=None, batch_size=batch_size)
    save_lids(pairwise_distances,3, str(dataset)+'_benign_knns_weighted_')
    save_lids(pairwise_distances,5, str(dataset)+'_benign_knns_weighted_')
    save_lids(pairwise_distances,10, str(dataset)+'_benign_knns_weighted_')
    save_lids(pairwise_distances,20, str(dataset)+'_benign_knns_weighted_')
    print('{}/{}'.format(a+batch_size, X_test.shape[0]))


print('total batches dataset/{}={}'.format(batch_size, X_test.shape[0]/batch_size))
for a in range(0, mal_np.shape[0], batch_size):

    sample= mal_np[a:a + batch_size, :]
    #pairwise_distances=batch_distances(sample, X_train, weights=feature_weights, batch_size=batch_size, train_set=False)
    pairwise_distances = batch_distances(sample, X_train, weights=None, batch_size=batch_size,
                                         train_set=False)
    save_lids(pairwise_distances,3, str(dataset)+'_mal_knns_weighted_')
    save_lids(pairwise_distances,5, str(dataset)+'_mal_knns_weighted_')
    save_lids(pairwise_distances,10, str(dataset)+'_mal_knns_weighted_')
    save_lids(pairwise_distances,20, str(dataset)+'_mal_knns_weighted_')
    print('{}/{}'.format(a+batch_size, X_test.shape[0]))

