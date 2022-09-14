from pairwise_distances import *
from data_setup import df_to_np, calculate_weights
from util.id import calculate_id
from sklearn import metrics

from sklearn.model_selection import train_test_split
import pandas as pd

dataset='csu'
directory='/s/luffy/b/nobackup/mgorb/iot/'
directory='csv/'
if dataset=='ton_iot':
    from data_preprocess.drop_columns import ton_iot
    benign_np=df_to_np(directory+'ton_iot/Train_Test_Network.csv',ton_iot.datatypes, train_set=True)
    mal_np=df_to_np(directory+'ton_iot/Train_Test_Network.csv', ton_iot.datatypes,train_set=False)
    #X_train, X_test = train_test_split(benign_np, test_size = 0.01, random_state = 42)
    X_train, X_test =benign_np, benign_np
    feature_weights=calculate_weights(X_train)

    df = pd.read_csv(directory+'ton_iot/Train_Test_Network.csv')
    df_benign = df[df['label'] == 0]
    idxs=['src_ip', 'dst_ip', 'type']
    benign_ips_attacks = df_benign[idxs].to_numpy()
elif dataset=='iot23':
    from data_preprocess.drop_columns import iot23
    benign_np=df_to_np(directory+'iot23/iot23_sample_with_real.csv',iot23.datatypes, train_set=True)

    mal_np=df_to_np(directory+'iot23/iot23_sample_with_real.csv', iot23.datatypes,train_set=False)
    #X_train, X_test = train_test_split(benign_np, test_size = 0.01, random_state = 42)
    X_train, X_test =benign_np, benign_np
    feature_weights=calculate_weights(X_train)

    df = pd.read_csv(directory +'iot23/iot23_sample_with_real.csv')
    df_benign = df[df['label string'] == 'Benign']
    idxs=['id.orig_h addr', 'id.resp_h addr', 'detailed-label string']
    benign_ips_attacks = df_benign[idxs].to_numpy()

elif dataset=='unsw_nb15':
    from data_preprocess.drop_columns import unsw_n15
    benign_np =df_to_np('csv/unsw-nb15/UNSW_NB15_training-set.csv', unsw_n15.datatypes,train_set=True)
    mal_np=df_to_np('csv/unsw-nb15/UNSW_NB15_training-set.csv',  unsw_n15.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np

    feature_weights=calculate_weights(X_train)
    #feature_weights=None

    print(X_train.shape)
elif dataset=='kaggle_nid':
    from data_preprocess.drop_columns import kaggle_nid
    benign_np =df_to_np('csv/kaggle_nid/Train_data.csv', kaggle_nid.datatypes,train_set=True)
    mal_np=df_to_np('csv/kaggle_nid/Train_data.csv',  kaggle_nid.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np

    feature_weights=calculate_weights(X_train)


elif dataset=='nf_bot_iot':
    from data_preprocess.drop_columns import nf_bot_iot
    benign_np =df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes,train_set=True)
    mal_np=df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv',  nf_bot_iot.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np

    feature_weights=calculate_weights(X_train)

    df = pd.read_csv(directory + 'nf_bot_iot/NF-BoT-IoT.csv')
    df_benign = df[df['Label'] == 0]
    idxs=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack']
    benign_ips_attacks = df_benign[idxs].to_numpy()

elif dataset=='csu':
    from data_preprocess.drop_columns import csu
    benign_np =df_to_np('csv/csu/features.csv', csu.datatypes,train_set=True)
    #mal_np=df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv',  nf_bot_iot.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np

    feature_weights=calculate_weights(X_train)

    print(benign_np.shape)


    df = pd.read_csv('csv/csu/features.csv')
    df_benign = df
    idxs=['device']
    benign_ips_attacks = df_benign[idxs].to_numpy()


def save_lids(pairwise_distances,k,sample_details, file_name):
    lids = np.expand_dims(np.array(calculate_id(pairwise_distances, k_=k)), axis=1)
    result=np.concatenate([lids, sample_details], axis=1)
    if a==0:
        df=pd.DataFrame( result, columns=['value']+idxs, )
    else:
        df2=pd.DataFrame(result, columns=['value']+idxs)
        df=pd.read_csv('results2/id/'+file_name+str(k)+'.csv')
        df=df.append(df2)
    df.to_csv('results2/id/'+file_name+str(k)+'.csv', index=False)


batch_size=1000
print('total batches dataset/{}={}'.format(batch_size, X_test.shape[0]/batch_size))
for a in range(0, X_test.shape[0], batch_size):
    sample= X_test[a:a + batch_size, :]
    sample_details=benign_ips_attacks[a:a + batch_size, :]
    pairwise_distances=batch_distances(sample, X_train, weights=feature_weights, batch_size=batch_size)
    save_lids(pairwise_distances,3,sample_details, str(dataset)+'_benign_ids_expanded_')
    save_lids(pairwise_distances,5,sample_details, str(dataset)+'_benign_ids_expanded_')
    save_lids(pairwise_distances,10,sample_details, str(dataset)+'_benign_ids_expanded_')
    save_lids(pairwise_distances,20,sample_details, str(dataset)+'_benign_ids_expanded_')
    print('{}/{}'.format(a+batch_size, X_test.shape[0]))


'''
full_dataset=np.concatenate([X_train,mal_np ], axis=0)

print('total batches dataset/{}={}'.format(batch_size, X_test.shape[0]/batch_size))
for a in range(0, mal_np.shape[0], batch_size):
    sample= mal_np[a:a + batch_size, :]
    pairwise_distances=batch_distances(sample, full_dataset, weights=feature_weights, batch_size=batch_size)
    save_lids(pairwise_distances,3, str(dataset)+'_mal_ids_')
    save_lids(pairwise_distances,5, str(dataset)+'_mal_ids_')
    save_lids(pairwise_distances,10, str(dataset)+'_mal_ids_')
    save_lids(pairwise_distances,20, str(dataset)+'_mal_ids_')
    print('{}/{}'.format(a+batch_size, X_test.shape[0]))
'''

