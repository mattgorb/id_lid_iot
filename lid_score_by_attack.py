from sklearn import metrics
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

k=5

#LID SCORES
print('LID Scores')


in_dist_lids=pd.read_csv('results/results/ton_iot_benign_lids_expanded_'+str(k)+'.csv')['value'].values
#ood_lids=pd.read_csv('results/results/ton_iot_mal_lids_expanded_'+str(k)+'.csv')['value'].values

df_benign=pd.read_csv('old/results/results/ton_iot_benign_lids_expanded_5.csv')
df_mal=pd.read_csv('old/results/results/ton_iot_mal_lids_expanded_5.csv')
#df_mal=pd.read_csv('results/nf_mal_lids_5_with_attack.csv')


ood_lids = [i for i in df_mal['value'].values if i != 0]
#in_dist_lids=[i for i in df_benign['value'].values]
labels=[1 for i in ood_lids]+[0 for i in in_dist_lids]
print(len(in_dist_lids))
print(len(ood_lids))
print(metrics.roc_auc_score(labels, ood_lids+in_dist_lids))
precision, recall, thresholds = metrics.precision_recall_curve(labels, ood_lids+in_dist_lids)
print(metrics.auc(recall, precision))


attacks=df_mal['type'].unique()
for attack in attacks:
    x=df_mal[df_mal['type']==attack]['value'].values
    print(x.shape)
    ood_lids = [i for i in x if i != 0]
    #in_dist_lids = [i for i in df_benign['value'].values]
    labels = [1 for i in ood_lids] + [0 for i in in_dist_lids]
    print(attack)
    print(len(in_dist_lids))
    print(len(ood_lids))
    print(metrics.roc_auc_score(labels, list(ood_lids)+list(in_dist_lids)))
    precision, recall, thresholds = metrics.precision_recall_curve(labels, list(ood_lids)+list(in_dist_lids))
    print(metrics.auc(recall, precision))
















df_benign=pd.read_csv('old/results2/iot23_benign_lids_expanded_3.csv')
df_mal=pd.read_csv('old/results2/iot23_mal_lids_expanded_3.csv')



ood_lids = [i for i in df_mal['value'].values if i != 0]
in_dist_lids=[i for i in df_benign['value'].values]
labels=[1 for i in ood_lids]+[0 for i in in_dist_lids]
print(len(in_dist_lids))
print(len(ood_lids))
print(metrics.roc_auc_score(labels, ood_lids+in_dist_lids))
precision, recall, thresholds = metrics.precision_recall_curve(labels, ood_lids+in_dist_lids)
print(metrics.auc(recall, precision))


attacks=df_mal['detailed-label string'].unique()
for attack in ['PartOfAHorizontalPortScan', 'DDoS', 'Okiru', 'C&C',]:
    x=df_mal[df_mal['detailed-label string'].str.contains(attack)]['value'].values
    print(x.shape)
    ood_lids = [i for i in x if i != 0]
    in_dist_lids = [i for i in df_benign['value'].values]
    labels = [1 for i in ood_lids] + [0 for i in in_dist_lids]
    print(attack)
    print(len(in_dist_lids))
    print(len(ood_lids))
    print(metrics.roc_auc_score(labels, ood_lids + in_dist_lids))
    precision, recall, thresholds = metrics.precision_recall_curve(labels, ood_lids + in_dist_lids)
    print(metrics.auc(recall, precision))