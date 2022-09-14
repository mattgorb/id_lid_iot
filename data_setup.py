import pandas as pd
from data_preprocess.preprocess import Preprocess
from pairwise_distances import *

from scipy.stats import entropy

def df_to_np(file_path,datatypes,  train_set=True, return_preprocess=False):
    if type(file_path)==list:
        li = []
        for filename in file_path:
            df = pd.read_csv(filename, index_col=None, header=0)
            if 'CTU-IoT-Malware-Capture-35-1' in filename:
                df=df[:int(df.shape[0] / 2)]
            li.append(df)

        df = pd.concat(li, axis=0, ignore_index=True)

    else:
        df=pd.read_csv(file_path)
    '''
    https://www.researchgate.net/publication/352055999_A_new_distributed_architecture_for_evaluating_AI-based_security_systems_at_the_edge_Network_TON_IoT_datasets
    We recommend that the researchers should remove the source and destination IP addresses and ports
    when they develop new machine learning algorithms.
    '''

    preprocess=Preprocess(df,datatypes)
    preprocess.preprocess()

    float_cols=[col for col in preprocess.benign_df.columns if preprocess.benign_df[col].dtype=='float64']
    categorical_cols=[col for col in preprocess.benign_df.columns if col not in float_cols]

    if train_set:
        _float=preprocess.benign_df[float_cols].to_numpy()
        _cat=preprocess.benign_df[categorical_cols].to_numpy()

    else:
        _float=preprocess.malicious_df[float_cols].to_numpy()
        _cat=preprocess.malicious_df[categorical_cols].to_numpy()

    full_data = np.concatenate([_float, _cat], axis=1)


    #full_data = np.concatenate([ _cat], axis=1)
    if return_preprocess:
        return full_data, preprocess, float_cols, categorical_cols
    else:
        return full_data

def calculate_weights(data):
    numerator=data.shape[1]
    feature_weights=[]
    for col_data in data.T:
        unique, counts=np.unique(col_data, return_counts=True)
        #print(entropy(counts, ))
        if entropy(counts, )>0:
            feature_weights.append(numerator/entropy(counts, ))
        else:
            feature_weights.append(.00001)
    feature_weights=np.array(feature_weights)
    return feature_weights