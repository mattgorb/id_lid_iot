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
from pairwise_distances import *
from data_setup import df_to_np, calculate_weights
from util.lid import calculate_lid, calculate_exactmatch
from sklearn import metrics

from sklearn.model_selection import train_test_split
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
    plt.savefig('results/corrMatrix.png')

    x=preprocess_df[preprocess_df['label']==0].to_numpy()[:2500] +preprocess_df[preprocess_df['label']==1].to_numpy()[:2500]
    print(x.shape)
    distances=squareform(pdist(x, metric='hamming'))

    sns.heatmap(distances, cmap="Greens", annot=False, yticklabels=False, xticklabels=False)
    plt.savefig('results/corrMatrix.png')

    sys.exit()

    float_cols=[col for col in preprocess_df.columns if preprocess_df[col].dtype=='float64']
    categorical_cols=[col for col in preprocess_df.columns if col not in float_cols]

    cramer_vals=[]
    num=3
    for i in range(len(categorical_cols[:num])):
        rows=[]
        for j in range(len(categorical_cols[:num])):
            confusion_matrix = pd.crosstab(preprocess_df[categorical_cols[i]], preprocess_df[categorical_cols[j]])
            cramer_val=cramers_v(confusion_matrix.values)
            #print(cramer_val)
            rows.append(cramer_val)
        cramer_vals.append(np.array(rows))
    cramer_vals=np.array(cramer_vals)

    df_matrix = pd.DataFrame(cramer_vals,index=pd.Index(categorical_cols[:num]),columns=pd.Index(categorical_cols[:num]))

    sns.heatmap(cramer_vals, cmap="Greens", annot=False, yticklabels=False, xticklabels=False)
    plt.savefig('results/corrMatrix_anomal_new.png')
    sys.exit()

    return preprocess_dict



dataset='ton_iot'
if dataset=='ton_iot':
    from data_preprocess.drop_columns import ton_iot
    benign_np=df_to_np('../csv/ton_iot/Train_Test_Network.csv', ton_iot.datatypes, train_set=True)
    mal_np=df_to_np('../csv/ton_iot/Train_Test_Network.csv', ton_iot.datatypes, train_set=False)
    #X_train, X_test = train_test_split(benign_np, test_size = 0.01, random_state = 42)
    X_train, X_test =benign_np, benign_np
    feature_weights=calculate_weights(X_train)

    #distances=squareform(pdist(benign_np[:5000], metric='hamming'))
    #sns.heatmap(distances, cmap="Greens", annot=False, yticklabels=False, xticklabels=False)
    #plt.savefig('results/corrMatrix.png')

    feature_weights = calculate_weights(X_train)
    pairwise_distances = batch_distances(X_test, X_train, weights=None, train_set=False)
    pairwise_distances_malicious = batch_distances(mal_np, X_train, weights=None, train_set=False)
    pairwise_all=np.concatenate([pairwise_distances, pairwise_distances_malicious], axis=0)

    print(pairwise_all)
    print(np.max(pairwise_all))
    print(np.min(pairwise_all))
    #sys.exit()
    #x=np.concatenate([X_train[:2000], mal_np[:2000]],axis=0)

    #distances=squareform(pdist(x, metric='hamming'))
    #distances=cdist(X_train[:2000],X_train[:2000])
    #distances_mal=cdist(mal_np[:2000],X_train[:2000])
    #distances=np.concatenate([distances, distances_mal], axis=0)


    #distances=np.sort(distances,axis=1)

    #enc = MinMaxScaler()
    #distances=enc.fit_transform(pairwise_all)*255
    #distances=(pairwise_all-np.min(pairwise_all))/(np.max(pairwise_all)-np.min(pairwise_all))
    #distances*=255

    '''print(distances)
    print(distances.shape)
    #sys.exit()

    from matplotlib import pyplot as plt
    plt.clf()
    plt.imshow(distances, interpolation='nearest')
    #plt.show()
    plt.savefig('results/a.png')
    sys.exit()

    sns.heatmap(pairwise_all, cmap="Greens", annot=False, yticklabels=False, xticklabels=False)
    plt.savefig('results/corrMatrix_anomaly.png')
    sys.exit()'''

#similarity matrix
fig, ax = plt.subplots()
sns.heatmap(preprocess.benign_df.corr(method='pearson'), annot=False, fmt='.2f',
            cmap=plt.get_cmap('Blues'), cbar=False, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
plt.show()
plt.savefig('results/heatmap.png', bbox_inches='tight', pad_inches=0.0)



