from data_preprocess.preprocess import Preprocess
from sklearn.manifold import TSNE
from pairwise_distances import *
#from data_setup import df_to_np, calculate_weights
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, QuantileTransformer, KBinsDiscretizer, OneHotEncoder

def preprocess(df,datatypes, all=True):
    new_df={}
    def preprocess_type(col):

        if col == 'label' or col == 'ts' or col == 'type' or col == 'label string':
            return 'default'
        elif (df[col].dtype == int or df[col].dtype == float) and (len(df[col].unique()) >= 60):
            return 'minmax'
        else:
            return 'categorical'

    if not all:
        if 'label' in datatypes:
            temp_df = df[df['label'] == 0]
        else:
            temp_df = df[df['label string'] == 'Benign']
    else:
        temp_df=df

    for col in df.columns:
        if col in datatypes:
            continue
        type_ = preprocess_type(col)
        if type_ == 'default':
            new_df[col] = df[col]

        elif type_ == 'categorical':
            # print("Processing {} as category".format(col))
            enc = OneHotEncoder()
            label_to_num=enc.fit_transform(np.expand_dims(temp_df[col].values, axis=1))
            new_df[col] = label_to_num
        elif type_ == 'minmax':
            # print("Processing {} as minmax".format(col))
            enc = MinMaxScaler()
            enc.fit(np.expand_dims(temp_df[col].values, axis=1))
            min_max_values = enc.transform(np.expand_dims(temp_df[col].values, axis=1))

            new_df[col] = min_max_values#.flatten()


    return new_df

def convert_df(file_path,datatypes, train_set=True):
    df=pd.read_csv(file_path)
    '''
    https://www.researchgate.net/publication/352055999_A_new_distributed_architecture_for_evaluating_AI-based_security_systems_at_the_edge_Network_TON_IoT_datasets
    We recommend that the researchers should remove the source and destination IP addresses and ports
    when they develop new machine learning algorithms.
    '''

    preprocess_dict=preprocess(df,datatypes)
    return preprocess_dict

dataset='ton_iot'
if dataset=='ton_iot':
    from data_preprocess.drop_columns import ton_iot
    preprocess_dict=convert_df('../csv/ton_iot/Train_Test_Network.csv', ton_iot.datatypes, train_set=True, )
    #print(list(preprocess_dict.values())[0])
    #x=np.array([preprocess_dict.values()[0], preprocess_dict.values()[1]])
    #print(x.shape)
    #sys.exit()
    features=[]
    for key,val in preprocess_dict.items():
        print(key)
        print(val.shape)
        features.append(val)

    x=np.array(features)
    sim_matrix=np.corrcoef(x)

    print(sim_matrix)
    print(sim_matrix.shape)
    sys.exit()


elif dataset=='iot23':
    from data_preprocess.drop_columns import iot23
    preprocess=convert_df('../csv/iot23/CTU-IoT-Malware-Capture-1-1.csv', iot23.datatypes, train_set=True, )

    '''print("here")
    print(benign_np.shape)
    print(mal_np.shape)

    benign_np=df_to_np('csv/iot23/CTU-IoT-Malware-Capture-3-1.csv',iot23.datatypes, train_set=True)
    mal_np=df_to_np('csv/iot23/CTU-IoT-Malware-Capture-3-1.csv',iot23.datatypes, train_set=False)

    print("here")
    print(benign_np.shape)
    print(mal_np.shape)

    benign_np=df_to_np('csv/iot23/CTU-IoT-Malware-Capture-20-1.csv',iot23.datatypes, train_set=True)
    mal_np=df_to_np('csv/iot23/CTU-IoT-Malware-Capture-20-1.csv',iot23.datatypes, train_set=False)
    print("here")
    print(benign_np.shape)
    print(mal_np.shape)
    benign_np=df_to_np('csv/iot23/CTU-IoT-Malware-Capture-21-1.csv',iot23.datatypes, train_set=True)
    mal_np=df_to_np('csv/iot23/CTU-IoT-Malware-Capture-21-1.csv',iot23.datatypes, train_set=False)
    print("here")
    print(benign_np.shape)
    print(mal_np.shape)

    benign_np=df_to_np('csv/iot23/CTU-IoT-Malware-Capture-34-1.csv',iot23.datatypes, train_set=True)
    mal_np=df_to_np('csv/iot23/CTU-IoT-Malware-Capture-34-1.csv',iot23.datatypes, train_set=False)
    print("here")
    print(benign_np.shape)
    print(mal_np.shape)
    sys.exit()
    #TOO BIG
    #benign_np=df_to_np('csv/iot23/CTU-IoT-Malware-Capture-35-1.csv',iot23.datatypes, train_set=True)
    #mal_np=df_to_np('csv/iot23/CTU-IoT-Malware-Capture-35-1.csv',iot23.datatypes, train_set=False)'''


elif dataset=='unsw_nb15':
    from data_preprocess.drop_columns import unsw_n15
    benign_np =convert_df('../csv/unsw-nb15/UNSW_NB15_training-set.csv', unsw_n15.datatypes, train_set=True, )
    #X_train, X_test =benign_np, benign_np
    #feature_weights=calculate_weights(X_train)
    #feature_weights=None
    #print(X_train.shape)




#similarity matrix
fig, ax = plt.subplots()
sns.heatmap(preprocess.benign_df.corr(method='pearson'), annot=False, fmt='.2f',
            cmap=plt.get_cmap('Blues'), cbar=False, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
plt.show()
plt.savefig('results/heatmap.pdf', bbox_inches='tight', pad_inches=0.0)



'''
#TSNE
n=20000
bigdata = preprocess.malicious_df.head(n)#.append(preprocess.malicious_df.head(n), ignore_index=True)

bigdata_float=bigdata[float_cols].to_numpy()
bigdata_cat=bigdata[categorical_cols].to_numpy()
dists=batch_distances(bigdata_float,bigdata_cat,bigdata_float,bigdata_cat)

tsne = TSNE(n_components=2,metric='precomputed' )
embedded=tsne.fit_transform(dists)

print(embedded.shape)

plt.clf()
plt.plot(embedded[:, 0], embedded[:,1], '.', label='benign')
#plt.plot(embedded[n:, 0], embedded[n:,1], '.', label='malicious')
plt.legend()
plt.savefig("tsne_malicious.png")
'''