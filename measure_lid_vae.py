
from pairwise_distances import *
from data_setup import df_to_np, calculate_weights
from util.lid import calculate_lid, calculate_exactmatch
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='VAE')

parser.add_argument('--dataset', type=str, default=None, metavar='N',
                    help='prior')

parser.add_argument('--run_benign', type=bool, default=True, metavar='N',
                    help='prior')
args = parser.parse_args()

dataset=args.dataset

directory='/s/luffy/b/nobackup/mgorb/iot/'

#directory='csv/'
if dataset=='ton_iot':
    from data_preprocess.drop_columns import ton_iot
    benign_np=df_to_np(directory+'/ton_iot/Train_Test_Network.csv',ton_iot.datatypes, train_set=True)


    benign_gen=np.load(f"{directory}/vae/syn_benign_True_ds_{dataset}.npy")
    mal_gen = np.load(f"{directory}/vae/syn_benign_False_ds_{dataset}.npy")

    X_train, X_test =benign_np, benign_gen


    print(X_train[0])
    print(X_test[0])
    sys.exit()

    feature_weights=calculate_weights(X_train)




elif dataset=='iot23':
    from data_preprocess.drop_columns import iot23

    benign_np=df_to_np(directory+'iot23/iot23_sample_with_real.csv',iot23.datatypes,  train_set=True)

    benign_gen=np.load(f"{directory}/vae/syn_benign_True_ds_{dataset}.npy")
    mal_gen = np.load(f"{directory}/vae/syn_benign_False_ds_{dataset}.npy")

    X_train, X_test =benign_np, benign_gen
    feature_weights=calculate_weights(X_train)




elif dataset=='nf_bot_iot':
    from data_preprocess.drop_columns import nf_bot_iot
    benign_np =df_to_np(directory+'nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes,train_set=True)
    benign_gen=np.load(f"{directory}/vae/syn_benign_True_ds_{dataset}.npy")
    mal_gen = np.load(f"{directory}/vae/syn_benign_False_ds_{dataset}.npy")



    X_train, X_test =benign_np, benign_gen

    feature_weights=calculate_weights(X_train)




elif dataset=='unsw_nb15':
    from data_preprocess.drop_columns import unsw_n15
    benign_np , preprocess, float_cols, categorical_cols=df_to_np('/s/luffy/b/nobackup/mgorb/iot/unsw-nb15/UNSW_NB15_training-set.csv', unsw_n15.datatypes,train_set=True, return_preprocess=True)
    benign_gen=np.load(f"{directory}/vae/syn_benign_True_ds_{dataset}.npy")
    mal_gen = np.load(f"{directory}/vae/syn_benign_False_ds_{dataset}.npy")

    X_train, X_test =benign_np, benign_gen


    feature_weights=calculate_weights(X_train)




elif dataset=='kaggle_nid':
    from data_preprocess.drop_columns import kaggle_nid
    benign_np =df_to_np('/s/luffy/b/nobackup/mgorb/iot/kaggle_nid/Train_data.csv', kaggle_nid.datatypes,train_set=True)

    benign_gen=np.load(f"{directory}/vae/syn_benign_True_ds_{dataset}.npy")
    mal_gen = np.load(f"{directory}/vae/syn_benign_False_ds_{dataset}.npy")

    X_train, X_test =benign_np, benign_gen

    feature_weights=calculate_weights(X_train)




def save_lids(pairwise_distances,k, sample_details,file_name):
    lids = np.expand_dims(np.array(calculate_lid(pairwise_distances, k_=k)), axis=1)

    if a==0:
        np.savetxt('results/'+file_name+str(k)+'.txt', lids)
    else:
        with open('results/'+file_name+str(k)+'.txt', "ab") as f:
            np.savetxt(f, lids)


batch_size=1000
print('total batches dataset/{}={}'.format(batch_size, X_test.shape[0]/batch_size))
for a in range(0, X_test.shape[0], batch_size):
    sample= X_test[a:a + batch_size, :]
    sample_details = benign_gen[a:a + batch_size, :]

    pairwise_distances=batch_distances(sample, X_train, weights=feature_weights, batch_size=batch_size)
    save_lids(pairwise_distances,3,sample_details, str(dataset)+'_benign_lids_syn_')
    save_lids(pairwise_distances,5,sample_details, str(dataset)+'_benign_lids_syn_')
    save_lids(pairwise_distances,10,sample_details, str(dataset)+'_benign_lids_syn_')
    save_lids(pairwise_distances,20,sample_details, str(dataset)+'_benign_lids_syn_')
    print('{}/{}'.format(a+batch_size, X_test.shape[0]))


print('total batches dataset/{}={}'.format(batch_size, X_test.shape[0]/batch_size))
for a in range(0, mal_gen.shape[0], batch_size):
    sample= mal_gen[a:a + batch_size, :]
    sample_details = mal_gen[a:a + batch_size, :]

    pairwise_distances=batch_distances(sample, X_train, weights=feature_weights, batch_size=batch_size, train_set=False)
    save_lids(pairwise_distances,3, sample_details,str(dataset)+'_mal_lids_syn_')
    save_lids(pairwise_distances,5,sample_details, str(dataset)+'_mal_lids_syn_')
    save_lids(pairwise_distances,10,sample_details, str(dataset)+'_mal_lids_syn_')
    save_lids(pairwise_distances,20,sample_details, str(dataset)+'_mal_lids_syn_')
    print('{}/{}'.format(a+batch_size, X_test.shape[0]))


