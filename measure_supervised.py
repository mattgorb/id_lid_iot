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




elif dataset=='nf_bot_iot':
    from data_preprocess.drop_columns import nf_bot_iot
    benign_np =df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes,train_set=True)
    mal_np=df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv',  nf_bot_iot.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np

    feature_weights=calculate_weights(X_train)

    print(X_train.shape)


full = np.concatenate([benign_np, mal_np], axis=0)
contamination_rate=mal_np.shape[0]/full.shape[0]
labels=[0 for i in benign_np]+[1 for i in mal_np]
from sklearn.svm import SVC

clf=SVC(kernel="linear", )
#clf=SVC()

clf.fit(full, np.array(labels))

full_pred=clf.decision_function(full)
#mal_pred=clf.predict(full_mal)

preds=list(full_pred)
print(roc_auc_score(labels, preds))
precision, recall, thresholds = metrics.precision_recall_curve(labels, preds)
print(metrics.auc(recall, precision))


