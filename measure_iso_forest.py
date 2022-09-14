import pandas as pd
from data_preprocess.preprocess import Preprocess
from pairwise_distances import *
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from data_setup import df_to_np, calculate_weights


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

    benign_np = df_to_np( 'csv/iot23/iot23_sample_with_real.csv', iot23.datatypes, train_set=True)

    mal_np = df_to_np( 'csv/iot23/iot23_sample_with_real.csv', iot23.datatypes, train_set=False)

    X_train, X_test = benign_np, benign_np
    feature_weights = calculate_weights(X_train)

    print('dataset shapes')
    print(benign_np.shape)
    print(mal_np.shape)




elif dataset=='nf_bot_iot':
    from data_preprocess.drop_columns import nf_bot_iot
    benign_np =df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes,train_set=True)
    mal_np=df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv',  nf_bot_iot.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np

    feature_weights=calculate_weights(X_train)

    print(X_train.shape)


full = np.concatenate([benign_np, mal_np], axis=0)
contamination_rate=mal_np.shape[0]/full.shape[0]
clf = IsolationForest(random_state=0, contamination=contamination_rate).fit(full)

full_pred=clf.decision_function(full)
#mal_pred=clf.predict(full_mal)
labels=[-1 for i in benign_np]+[1 for i in mal_np]
preds=list(full_pred)
print(roc_auc_score(labels, preds))
precision, recall, thresholds = metrics.precision_recall_curve(labels, preds)
print(metrics.auc(recall, precision))


