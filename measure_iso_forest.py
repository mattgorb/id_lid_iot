import pandas as pd
from data_preprocess.preprocess import Preprocess
from pairwise_distances import *
from sklearn.metrics import roc_auc_score, f1_score
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from data_setup import df_to_np, calculate_weights

syn=True
dataset='ton_iot'
directory='/s/luffy/b/nobackup/mgorb/iot/'
if dataset=='ton_iot':
    from data_preprocess.drop_columns import ton_iot
    benign_np=df_to_np(f'{directory}/ton_iot/Train_Test_Network.csv',ton_iot.datatypes, train_set=True)

    mal_np=df_to_np(f'{directory}/ton_iot/Train_Test_Network.csv', ton_iot.datatypes,train_set=False)

    if syn:
        print("synthetic")
        benign_np = np.load(f"{directory}/vae/recon_benign_True_ds_{dataset}.npy")
        mal_np = np.load(f"{directory}/vae/recon_benign_False_ds_{dataset}.npy")

    #X_train, X_test = train_test_split(benign_np, test_size = 0.01, random_state = 42)
    X_train, X_test =benign_np, benign_np
    feature_weights=calculate_weights(X_train)
elif dataset=='iot23':
    from data_preprocess.drop_columns import iot23

    benign_np = df_to_np( f'{directory}/iot23/iot23_sample_with_real.csv', iot23.datatypes, train_set=True)
    mal_np = df_to_np( f'{directory}/iot23/iot23_sample_with_real.csv', iot23.datatypes, train_set=False)

    if syn:
        print("synthetic")
        benign_np = np.load(f"{directory}/vae/recon_benign_True_ds_{dataset}.npy")
        mal_np = np.load(f"{directory}/vae/recon_benign_False_ds_{dataset}.npy")

    X_train, X_test = benign_np, benign_np
    feature_weights = calculate_weights(X_train)

    print('dataset shapes')
    print(benign_np.shape)
    print(mal_np.shape)




elif dataset=='nf_bot_iot':
    from data_preprocess.drop_columns import nf_bot_iot
    benign_np =df_to_np(f'{directory}/nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes,train_set=True)
    mal_np=df_to_np(f'{directory}/nf_bot_iot/NF-BoT-IoT.csv',  nf_bot_iot.datatypes,train_set=False)

    if syn:
        print("synthetic")
        benign_np = np.load(f"{directory}/vae/recon_benign_True_ds_{dataset}.npy")
        mal_np = np.load(f"{directory}/vae/recon_benign_False_ds_{dataset}.npy")

    X_train, X_test =benign_np, benign_np

    feature_weights=calculate_weights(X_train)

    print(X_train.shape)

elif dataset=='unsw_nb15':
    from data_preprocess.drop_columns import unsw_n15
    benign_np , preprocess, float_cols, categorical_cols=df_to_np('/s/luffy/b/nobackup/mgorb/iot/unsw-nb15/UNSW_NB15_training-set.csv', unsw_n15.datatypes,train_set=True, return_preprocess=True)
    benign_np_test , _, _, _=df_to_np('/s/luffy/b/nobackup/mgorb/iot/unsw-nb15/UNSW_NB15_testing-set.csv', unsw_n15.datatypes,train_set=True, return_preprocess=True)

    mal_np=df_to_np('/s/luffy/b/nobackup/mgorb/iot/unsw-nb15/UNSW_NB15_testing-set.csv',  unsw_n15.datatypes,train_set=False)

    if syn:
        print("synthetic")
        benign_np = np.load(f"{directory}/vae/recon_benign_True_ds_{dataset}.npy")
        mal_np = np.load(f"{directory}/vae/recon_benign_False_ds_{dataset}.npy")

    X_train, X_test =benign_np, benign_np

    X_train = X_train.astype('float64')

elif dataset=='kaggle_nid':
    from data_preprocess.drop_columns import kaggle_nid
    benign_np =df_to_np('/s/luffy/b/nobackup/mgorb/iot/kaggle_nid/Train_data.csv', kaggle_nid.datatypes,train_set=True)
    mal_np=df_to_np('/s/luffy/b/nobackup/mgorb/iot/kaggle_nid/Train_data.csv',  kaggle_nid.datatypes,train_set=False)
    if syn:
        print("synthetic")
        benign_np = np.load(f"{directory}/vae/recon_benign_True_ds_{dataset}.npy")
        mal_np = np.load(f"{directory}/vae/recon_benign_False_ds_{dataset}.npy")

    X_train, X_test =benign_np, benign_np

    feature_weights=calculate_weights(X_train)
elif dataset=='nf-cse-cic':
    from data_preprocess.drop_columns import nf_cse_cic
    benign_np =df_to_np(f'{directory}/nf-cse-cic/nf-cse-cic-sample.csv', nf_cse_cic.datatypes,train_set=True)
    mal_np=df_to_np(f'{directory}/nf-cse-cic/nf-cse-cic-sample.csv',  nf_cse_cic.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np




full = np.concatenate([X_test, mal_np], axis=0)
contamination_rate=mal_np.shape[0]/full.shape[0]
#clf = IsolationForest(random_state=0, contamination=contamination_rate).fit(full)
clf = IsolationForest(random_state=0, ).fit(full)
full_pred=clf.decision_function(full)
#mal_pred=clf.predict(full_mal)
labels=[-1 for i in X_test]+[1 for i in mal_np]
preds=list(full_pred)
print("ROC")
print(roc_auc_score(labels, preds))
precision, recall, thresholds = metrics.precision_recall_curve(labels, preds)
print("PR")
print(metrics.auc(recall, precision))
