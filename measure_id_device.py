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
    benign_np=df_to_np(directory+'iot23/iot23_sample.csv',iot23.datatypes, train_set=True)

    mal_np=df_to_np(directory+'iot23/iot23_sample.csv', iot23.datatypes,train_set=False)
    #X_train, X_test = train_test_split(benign_np, test_size = 0.01, random_state = 42)
    X_train, X_test =benign_np, benign_np
    feature_weights=calculate_weights(X_train)

    df = pd.read_csv(directory +'iot23/iot23_sample.csv')
    df_benign = df[df['label string'] == 'Benign']
    idxs=['id.orig_h addr', 'id.resp_h addr', 'detailed-label string']
    benign_ips_attacks = df_benign[idxs].to_numpy()

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

    print(benign_ips_attacks.shape)


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


'''x1=[si for i, si in enumerate(benign_ips_attacks[:,0]) if si.startswith('192.168.')]
x2=[si for i, si in enumerate(benign_ips_attacks[:,1]) if si.startswith('192.168.')]
x=list(x1)+list(x2)

devices=list(set(x))
print(devices)
print(len(devices))
for d in devices:

    mask = (benign_ips_attacks[:, 0] == d )| (benign_ips_attacks[:,1]==d)
    device_details=benign_ips_attacks[mask, :]
    device_features=X_train[mask, :]
    device_features2=X_test[mask, :]
    print(d)
    print(device_details.shape)
    print(device_features.shape)

    batch_size=1000
    print('total batches dataset/{}={}'.format(batch_size, device_features2.shape[0]/batch_size))
    for a in range(0, device_features2.shape[0], batch_size):
        sample= device_features2[a:a + batch_size, :]
        sample_details=device_details[a:a + batch_size, :]

        pairwise_distances=batch_distances(sample, device_features, weights=feature_weights, batch_size=batch_size)
        save_lids(pairwise_distances,3,sample_details, str(dataset)+'_benign_ids_expanded_'+d+'_')
        save_lids(pairwise_distances,5,sample_details, str(dataset)+'_benign_ids_expanded_'+d+'_')
        save_lids(pairwise_distances,10,sample_details, str(dataset)+'_benign_ids_expanded_'+d+'_')
        save_lids(pairwise_distances,20,sample_details, str(dataset)+'_benign_ids_expanded_'+d+'_')
        print('{}/{}'.format(a+batch_size, device_features2.shape[0]))'''

#x1=[si for i, si in enumerate(benign_ips_attacks[:,0]) if si.startswith('192.168.')]
#x2=[si for i, si in enumerate(benign_ips_attacks[:,1]) if si.startswith('192.168.')]
x=list(benign_ips_attacks[:,0])#+list(x2)

devices=list(set(x))

print(devices)
print(len(devices))

for d in devices:

    mask = (benign_ips_attacks[:, 0] == d )#| (benign_ips_attacks[:,1]==d)
    device_details=benign_ips_attacks[mask, :]
    device_features=X_train[mask, :]
    device_features2=X_test[mask, :]
    print(d)
    d=str(d)
    print(device_details.shape)
    print(device_features.shape)

    batch_size=1000
    print('total batches dataset/{}={}'.format(batch_size, device_features2.shape[0]/batch_size))
    for a in range(0, device_features2.shape[0], batch_size):
        sample= device_features2[a:a + batch_size, :]
        sample_details=device_details[a:a + batch_size, :]

        pairwise_distances=batch_distances(sample, device_features, weights=feature_weights, batch_size=batch_size)
        save_lids(pairwise_distances,3,sample_details, str(dataset)+'_benign_ids_expanded_'+d+'_')
        save_lids(pairwise_distances,5,sample_details, str(dataset)+'_benign_ids_expanded_'+d+'_')
        save_lids(pairwise_distances,10,sample_details, str(dataset)+'_benign_ids_expanded_'+d+'_')
        save_lids(pairwise_distances,20,sample_details, str(dataset)+'_benign_ids_expanded_'+d+'_')
        print('{}/{}'.format(a+batch_size, device_features2.shape[0]))