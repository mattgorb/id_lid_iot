from pairwise_distances import *
from data_setup import df_to_np, calculate_weights
from util.id import calculate_id
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter



dataset = 'ton_iot'
if dataset == 'ton_iot':
    from data_preprocess.drop_columns import ton_iot

    benign_np = df_to_np('../csv/ton_iot/Train_Test_Network.csv', ton_iot.datatypes, train_set=True)
    mal_np = df_to_np('../csv/ton_iot/Train_Test_Network.csv', ton_iot.datatypes, train_set=False)
    # X_train, X_test = train_test_split(benign_np, test_size = 0.01, random_state = 42)
    X_train, X_test = benign_np, benign_np
    feature_weights = calculate_weights(X_train)
elif dataset == 'iot23':
    from data_preprocess.drop_columns import iot23

    files_group1 = ['CTU-IoT-Malware-Capture-1-1', 'CTU-IoT-Malware-Capture-3-1', 'CTU-IoT-Malware-Capture-7-1',
                    'CTU-IoT-Malware-Capture-8-1', 'CTU-IoT-Malware-Capture-9-1', 'CTU-IoT-Malware-Capture-17-1',
                    'CTU-IoT-Malware-Capture-20-1', 'CTU-IoT-Malware-Capture-21-1']
    files_group2 = [
        'CTU-IoT-Malware-Capture-34-1', 'CTU-IoT-Malware-Capture-33-1', 'CTU-IoT-Malware-Capture-35-1',
        'CTU-IoT-Malware-Capture-36-1', 'CTU-IoT-Malware-Capture-39-1', 'CTU-IoT-Malware-Capture-42-1', ]

    files_group3 = ['CTU-IoT-Malware-Capture-43-1', 'CTU-IoT-Malware-Capture-44-1', 'CTU-IoT-Malware-Capture-48-1',
                    'CTU-IoT-Malware-Capture-49-1', 'CTU-IoT-Malware-Capture-52-1', 'CTU-IoT-Malware-Capture-60-1',
                    'CTU-IoT-Malware-Capture-35-1'
                    ]

    # files=['CTU-Honeypot-Capture-4-1','CTU-Honeypot-Capture-5-1','CTU-Honeypot-Capture-7-1']

    benign_np = None

    files = files_group2

    if files == files_group1:
        dataset = 'Malware-Capture-Group-1'
    elif files == files_group2:
        dataset = 'Malware-Capture-Group-2'
    elif files == files_group3:
        dataset = 'Malware-Capture-Group-3'
    print(dataset)
    attack_type=[]
    for file in files:
        if benign_np is None:
            benign_np = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=True)
            mal_np = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=False)
            df = pd.read_csv('./csv/iot23/' + file + '.csv')
            df = df[df[df.columns[-2]] == 'Malicious']
            attack_type.extend(df[df.columns[-1]].values)

        else:
            if files == files_group2 and file == 'CTU-IoT-Malware-Capture-35-1':
                benign_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes,
                                   train_set=True)  # [:int(benign_np.shape[0]/2)]
                mal_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes,
                                train_set=False)  # [:int(benign_np.shape[0]/2)]
                benign_ = benign_[:int(benign_.shape[0] / 2)]
                #print(mal_.shape)
                mal_ = mal_[:int(mal_.shape[0] / 2)]
                #print(mal_.shape)
                df=pd.read_csv('./csv/iot23/' + file + '.csv')
                df=df[df[df.columns[-2]]=='Malicious']
                df=df[:mal_.shape[0]]

                attacks=df[df.columns[-1]].values
                attack_type.extend(attacks)

            elif files == files_group3 and file == 'CTU-IoT-Malware-Capture-35-1':
                benign_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=True)
                mal_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=False)
                benign_ = benign_[int(benign_.shape[0] / 2):]
                mal_ = mal_[int(mal_.shape[0] / 2):]
            else:
                benign_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=True)
                mal_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=False)
                df=pd.read_csv('./csv/iot23/' + file + '.csv')
                df=df[df[df.columns[-2]]=='Malicious']
                attack_type.extend(df[df.columns[-1]].values)
                #ys.exit()

            benign_np = np.concatenate([benign_np, benign_], axis=0)
            mal_np = np.concatenate([mal_np, mal_], axis=0)


    files = ['CTU-Honeypot-Capture-4-1','CTU-Honeypot-Capture-5-1','CTU-Honeypot-Capture-7-1']

    print(benign_np.shape)
    print(mal_np.shape)
    print(len(attack_type))
    print(attack_type[:5000])

    #words = ['a', 'b', 'c', 'a']

    print(Counter(attack_type[:5000]).keys() ) # equals to list(set(words))
    print(Counter(attack_type[:5000]).values())  # counts the elements' frequency

    from sklearn.utils import shuffle
    mal_np, attack_type = shuffle(mal_np, np.array(attack_type), random_state=1)
    print(Counter(attack_type[:5000]).keys() ) # equals to list(set(words))
    print(Counter(attack_type[:5000]).values())  # counts the elements' frequency

    #print(dataset)
    honeypot={}

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

            elif files == files_group3 and file == 'CTU-IoT-Malware-Capture-35-1':
                benign_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=True)
                mal_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=False)
                benign_ = benign_[int(benign_.shape[0] / 2):]
                mal_ = mal_[int(mal_.shape[0] / 2):]
            else:
                benign_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=True)
                mal_ = df_to_np('./csv/iot23/' + file + '.csv', iot23.datatypes, train_set=False)
            print(file)
            print(benign_.shape)
            print(mal_.shape)
            honeypot[file]=benign_
            #benign_np = np.concatenate([benign_np, benign_], axis=0)
            #mal_np = np.concatenate([mal_np, mal_], axis=0)



    if files == files_group1:
        mal_np = mal_np[:577000]
    elif files == files_group2:
        mal_np = mal_np[:benign_np.shape[0]]
    elif files == files_group3:
        mal_np = mal_np[:benign_np.shape[0]]

    X_train, X_test = benign_np, benign_np
    feature_weights = calculate_weights(X_train)




elif dataset == 'unsw_nb15':
    from data_preprocess.drop_columns import unsw_n15

    benign_np = df_to_np('../csv/unsw-nb15/UNSW_NB15_training-set.csv', unsw_n15.datatypes, train_set=True)
    mal_np = df_to_np('../csv/unsw-nb15/UNSW_NB15_training-set.csv', unsw_n15.datatypes, train_set=False)
    X_train, X_test = benign_np, benign_np

    feature_weights = calculate_weights(X_train)
    # feature_weights=None

    print(X_train.shape)
elif dataset == 'kaggle_nid':
    from data_preprocess.drop_columns import kaggle_nid

    benign_np = df_to_np('../csv/kaggle_nid/Train_data.csv', kaggle_nid.datatypes, train_set=True)
    mal_np = df_to_np('../csv/kaggle_nid/Train_data.csv', kaggle_nid.datatypes, train_set=False)
    X_train, X_test = benign_np, benign_np

    feature_weights = calculate_weights(X_train)


elif dataset == 'nf_bot_iot':
    from data_preprocess.drop_columns import nf_bot_iot

    benign_np = df_to_np('../csv/nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes, train_set=True)
    mal_np = df_to_np('../csv/nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes, train_set=False)
    X_train, X_test = benign_np, benign_np

    feature_weights = calculate_weights(X_train)

    print(X_train.shape)


batch_size=1000
pairwise_distances = None


benign_size=25000
mal_size=10000
X_test=np.concatenate([X_test[:benign_size],mal_np[:mal_size]], axis=0)
X_train=np.concatenate([X_train[:benign_size],mal_np[:mal_size]], axis=0)

#for key, value in honeypot.items():
#'CTU-Honeypot-Capture-4-1','CTU-Honeypot-Capture-5-1','CTU-Honeypot-Capture-7-1']
'''honeypot['Philips HUE']=honeypot['CTU-Honeypot-Capture-4-1']
del honeypot['CTU-Honeypot-Capture-4-1']

honeypot['Amazon Echo']=honeypot['CTU-Honeypot-Capture-5-1']
del honeypot['CTU-Honeypot-Capture-5-1']

honeypot['Somfy Door Lock']=honeypot['CTU-Honeypot-Capture-7-1']
del honeypot['CTU-Honeypot-Capture-7-1']

for key, value in honeypot.items():
    X_test = np.concatenate([X_test, value], axis=0)
    X_train = np.concatenate([X_train, value], axis=0)
    print(value.shape)'''

print('total batches dataset/{}={}'.format(batch_size, X_test.shape[0]/batch_size))
for a in range(0, X_test.shape[0], batch_size):
    sample= X_test[a:a + batch_size, :]
    if pairwise_distances is None:
        pairwise_distances=batch_distances(sample, X_train, weights=feature_weights, batch_size=batch_size, sort_=False)
    else:
        pairwise_distances=np.concatenate([pairwise_distances,batch_distances(sample, X_train, weights=feature_weights, batch_size=batch_size,sort_=False) ], axis=0)
    print(pairwise_distances.shape)
    #print(X_train.shape)
    #sys.exit()

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2,metric='precomputed' )
embedded=tsne.fit_transform(pairwise_distances)

print(embedded.shape)
plt.clf()
plt.plot(embedded[:benign_size, 0], embedded[:benign_size,1], '.', label='Benign Traffic')
plt.plot(embedded[benign_size:benign_size+mal_size, 0], embedded[benign_size:benign_size+mal_size,1], '.', label='Attack')
'''start=benign_size+mal_size
for key, value in honeypot.items():
    print(start)
    print(value.shape[0])
    print(embedded[start:start+value.shape[0], 0].shape)
    print(embedded[start:start+value.shape[0], 1].shape)
    plt.plot(embedded[start:start+value.shape[0], 0], embedded[start:start+value.shape[0], 1], '.', label=key)
    start+=value.shape[0]'''
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          #fancybox=True, shadow=True, ncol=5)
plt.legend()
plt.title('T-SNE Projection of TON_IoT')
plt.savefig('tsne.pdf', bbox_inches="tight")
sys.exit()

print(embedded.shape)
plt.clf()
plt.plot(embedded[:benign_size, 0], embedded[:benign_size,1], '.', label='Benign (Test Bed Device)')
for attack in set(attack_type[:mal_size]):
    plt.plot([embedded[benign_size+i,0] for i in range(embedded[benign_size:benign_size+mal_size].shape[0]) if attack_type[i]==attack], [embedded[benign_size+i,1] for i in range(embedded[benign_size:benign_size+mal_size].shape[0]) if attack_type[i]==attack], '.', label=attack)
start=benign_size+mal_size
for key, value in honeypot.items():
    plt.plot(embedded[start:start+value.shape[0], 0], embedded[start:start+value.shape[0], 1], '.', label=key)
    start+=value.shape[0]
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.savefig('tsne2.pdf', bbox_inches="tight")
