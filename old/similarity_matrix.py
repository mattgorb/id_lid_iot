from data_preprocess.preprocess import Preprocess
from sklearn.manifold import TSNE
from pairwise_distances import *
#from data_setup import df_to_np, calculate_weights
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, QuantileTransformer, KBinsDiscretizer, OneHotEncoder
from scipy.spatial.distance import pdist, squareform

def preprocess(df,datatypes, all=True):
    new_df=pd.DataFrame()
    def preprocess_type(col):

        if col == 'label' or col == 'ts' or col == 'type' or col == 'label string':
            return 'default'
        elif (df[col].dtype == int or df[col].dtype == float) and (len(df[col].unique()) >= 60):
            return 'minmax'
        else:
            return 'categorical'

    if  all:
        temp_df = df

    else:
        if 'label' in datatypes:
            temp_df = df[df['label'] == 0]
        else:
            temp_df = df[df['label string'] == 'Benign']

    for col in df.columns:

        if col in datatypes:
            continue
        type_ = preprocess_type(col)
        if type_ == 'default':
            new_df[col] = df[col]

        elif type_ == 'categorical':
            #print("Processing {} as category".format(col))
            enc = LabelEncoder()
            label_to_num=enc.fit_transform(np.expand_dims(temp_df[col].values, axis=1).ravel())

            new_df[col] = label_to_num.ravel()
        elif type_ == 'minmax':
            #print("Processing {} as minmax".format(col))
            enc = MinMaxScaler()
            min_max_values =enc.fit_transform(np.expand_dims(temp_df[col].values, axis=1))
            new_df[col] = min_max_values.ravel()


    return new_df

def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def convert_df(file_path,datatypes, train_set=True):
    df=pd.read_csv(file_path)
    '''
    https://www.researchgate.net/publication/352055999_A_new_distributed_architecture_for_evaluating_AI-based_security_systems_at_the_edge_Network_TON_IoT_datasets
    We recommend that the researchers should remove the source and destination IP addresses and ports
    when they develop new machine learning algorithms.
    '''


    preprocess_df=preprocess(df,datatypes,all=True)
    print(preprocess_df.shape)
    print(preprocess_df.to_numpy()[:20000].shape)
    #sys.exit()

    distances=squareform(pdist([preprocess_df['label']==0].to_numpy()[:5000], metric='hamming'))

    sns.heatmap(distances, cmap="Greens", annot=False, yticklabels=False, xticklabels=False)
    plt.savefig('results/corrMatrix.pdf')

    x=preprocess_df[preprocess_df['label']==0].to_numpy()[:2500] +preprocess_df[preprocess_df['label']==1].to_numpy()[:2500]
    print(x.shape)
    distances=squareform(pdist(x, metric='hamming'))

    sns.heatmap(distances, cmap="Greens", annot=False, yticklabels=False, xticklabels=False)
    plt.savefig('results/corrMatrix.pdf')

    sys.exit()

    float_cols=[col for col in preprocess.benign_df.columns if preprocess.df[col].dtype=='float64']
    categorical_cols=[col for col in preprocess.benign_df.columns if col not in float_cols]

    cramer_vals=[]
    num=8
    for i in range(len(categorical_cols[:num])):
        rows=[]
        for j in range(len(categorical_cols[:num])):
            confusion_matrix = pd.crosstab(preprocess.df[categorical_cols[i]], preprocess.benign_df[categorical_cols[j]])
            cramer_val=cramers_v(confusion_matrix.values)
            #print(cramer_val)
            rows.append(cramer_val)
        cramer_vals.append(np.array(rows))
    cramer_vals=np.array(cramer_vals)

    #df_matrix = pd.DataFrame(cramer_vals,index=pd.Index(categorical_cols[:num]),columns=pd.Index(categorical_cols[:num]))

    plt.clf()
    sns.heatmap(cramer_vals, cmap="Greens", annot=False, yticklabels=False, xticklabels=False)
    plt.show()

    plt.savefig('results/corrMatrix_anomal_new.png')
    sys.exit()

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
plt.savefig('results/heatmap.png', bbox_inches='tight', pad_inches=0.0)



