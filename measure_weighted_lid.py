from pairwise_distances import *
from data_setup import df_to_np, calculate_weights
from util.lid import calculate_weighted_lid, calculate_exactmatch
from sklearn import metrics

from sklearn.model_selection import train_test_split

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

    files_group1 = ['CTU-IoT-Malware-Capture-1-1', 'CTU-IoT-Malware-Capture-3-1', 'CTU-IoT-Malware-Capture-7-1',
                    'CTU-IoT-Malware-Capture-8-1', 'CTU-IoT-Malware-Capture-9-1', 'CTU-IoT-Malware-Capture-17-1',
                    'CTU-IoT-Malware-Capture-20-1', 'CTU-IoT-Malware-Capture-21-1']
    files_group2 = [
        'CTU-IoT-Malware-Capture-34-1', 'CTU-IoT-Malware-Capture-33-1', 'CTU-IoT-Malware-Capture-35-1',
        'CTU-IoT-Malware-Capture-36-1', 'CTU-IoT-Malware-Capture-39-1', 'CTU-IoT-Malware-Capture-42-1', ]

    files_group3 = ['CTU-IoT-Malware-Capture-43-1', 'CTU-IoT-Malware-Capture-44-1', 'CTU-IoT-Malware-Capture-48-1',
                    'CTU-IoT-Malware-Capture-49-1', 'CTU-IoT-Malware-Capture-52-1', 'CTU-IoT-Malware-Capture-60-1',
                    ]

    # files=['CTU-Honeypot-Capture-4-1','CTU-Honeypot-Capture-5-1','CTU-Honeypot-Capture-7-1']

    benign_np = None

    files = files_group3

    if files == files_group1:
        dataset = 'Malware-Capture-Group-1'
    elif files == files_group2:
        dataset = 'Malware-Capture-Group-2'
    elif files == files_group3:
        dataset = 'Malware-Capture-Group-3'
    print(dataset)
    for file in files:
        if benign_np is None:
            benign_np = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=True)
            mal_np = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=False)
        else:
            if files == files_group2 and file == 'CTU-IoT-Malware-Capture-35-1':
                benign_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes,
                                   train_set=True)  # [:int(benign_np.shape[0]/2)]
                mal_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes,
                                train_set=False)  # [:int(benign_np.shape[0]/2)]
                benign_ = benign_[:int(benign_.shape[0] / 2)]
                mal_ = mal_[:int(mal_.shape[0] / 2)]
            else:
                benign_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=True)
                mal_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=False)

            benign_np = np.concatenate([benign_np, benign_], axis=0)
            mal_np = np.concatenate([mal_np, mal_], axis=0)

    if files == files_group1:
        mal_np = mal_np[:577000]
    elif files == files_group2:
        mal_np = mal_np[:benign_np.shape[0]]
    elif files == files_group3:
        mal_np = mal_np[:benign_np.shape[0]]

    X_train, X_test = benign_np, benign_np
    feature_weights = calculate_weights(X_train)

    print('dataset shapes')
    print(benign_np.shape)
    print(mal_np.shape)




elif dataset=='unsw_nb15':
    from data_preprocess.drop_columns import unsw_n15
    benign_np =df_to_np('csv/unsw-nb15/UNSW_NB15_training-set.csv', unsw_n15.datatypes,train_set=True)
    mal_np=df_to_np('csv/unsw-nb15/UNSW_NB15_training-set.csv',  unsw_n15.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np

    feature_weights=calculate_weights(X_train)
    #feature_weights=None

elif dataset=='nf_bot_iot':
    from data_preprocess.drop_columns import nf_bot_iot
    benign_np =df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes,train_set=True)
    mal_np=df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv',  nf_bot_iot.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np

    feature_weights=calculate_weights(X_train)

    print(X_train.shape)

def save_lids(pairwise_distances,k, file_name):
    lids = calculate_weighted_lid(pairwise_distances, k_=k)
    if a==0:
        np.savetxt('results/'+file_name+str(k)+'.txt', lids)
    else:
        with open('results/'+file_name+str(k)+'.txt', "ab") as f:
            np.savetxt(f, lids)


batch_size=1000
print('total batches dataset/{}={}'.format(batch_size, X_test.shape[0]/batch_size))
#for a in range(0, X_test.shape[0], batch_size):
for a in range(0, 10000, batch_size):
    sample= X_test[a:a + batch_size, :]

    pairwise_distances=batch_distances(sample, X_train, weights=feature_weights, batch_size=batch_size)
    save_lids(pairwise_distances,3, str(dataset)+'_benign_weighted_lids_')
    save_lids(pairwise_distances,5, str(dataset)+'_benign_weighted_lids_')
    save_lids(pairwise_distances,10, str(dataset)+'_benign_weighted_lids_')
    save_lids(pairwise_distances,20, str(dataset)+'_benign_weighted_lids_')
    print('{}/{}'.format(a+batch_size, X_test.shape[0]))


print('total batches dataset/{}={}'.format(batch_size, X_test.shape[0]/batch_size))
#for a in range(0, mal_np.shape[0], batch_size):
for a in range(0, 10000, batch_size):
    sample= mal_np[a:a + batch_size, :]
    pairwise_distances=batch_distances(sample, X_train, weights=feature_weights, batch_size=batch_size, train_set=False)
    save_lids(pairwise_distances,3, str(dataset)+'_mal_weighted_lids_')
    save_lids(pairwise_distances,5, str(dataset)+'_mal_weighted_lids_')
    save_lids(pairwise_distances,10, str(dataset)+'_mal_weighted_lids_')
    save_lids(pairwise_distances,20, str(dataset)+'_mal_weighted_lids_')
    print('{}/{}'.format(a+batch_size, X_test.shape[0]))


