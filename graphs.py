from sklearn import metrics
import numpy as np

import matplotlib.pyplot as plt


file='ton_iot'

in_dist_lids=list(np.loadtxt('results/'+str(file)+'_benign_knns_5.txt'))
ood_lids=list(np.loadtxt('results/'+str(file)+'_mal_knns_5.txt'))
ood_lids = [i for i in ood_lids if i != 0]
knns_list=ood_lids+in_dist_lids
knns_labels=[1 for i in ood_lids]+[0 for i in in_dist_lids]


in_dist_lids=list(np.loadtxt('results/'+str(file)+'_benign_knns_weighted_hamming_5.txt'))
ood_lids=list(np.loadtxt('results/'+str(file)+'_mal_knns_weighted_hamming_5.txt'))
ood_lids = [i for i in ood_lids if i != 0]
knns_weighted_list=ood_lids+in_dist_lids
knns_weighted_labels=[1 for i in ood_lids]+[0 for i in in_dist_lids]

in_dist_lids=list(np.loadtxt('results/'+str(file)+'_benign_lids_5.txt'))
ood_lids=list(np.loadtxt('results/'+str(file)+'_mal_lids_5.txt'))
ood_lids = [i for i in ood_lids if i != 0]
lids_list=ood_lids+in_dist_lids
lids_labels=[1 for i in ood_lids]+[0 for i in in_dist_lids]


#precision, recall, thresholds = metrics.precision_recall_curve(labels, ood_lids+in_dist_lids)


plt.clf()
fpr, tpr, _ = metrics.roc_curve(lids_labels , lids_list )
auc = str(metrics.roc_auc_score(lids_labels,  lids_list))[:5]
plt.plot(fpr,tpr,label="Weighted Hamming LIDs, auc="+str(auc))

fpr, tpr, _ = metrics.roc_curve(knns_weighted_labels , knns_weighted_list )
auc = str(metrics.roc_auc_score(knns_weighted_labels,  knns_weighted_list))[:5]
plt.plot(fpr,tpr,label="Weighted KNN, auc="+str(auc))

fpr, tpr, _ = metrics.roc_curve(knns_labels , knns_list )
auc =str( metrics.roc_auc_score(knns_labels,  knns_list))[:5]
plt.plot(fpr,tpr,label="KNN, auc="+str(auc))

plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under Receiver Operating Characterstic')
plt.savefig('images/roc_ton.pdf')



plt.clf()
#fpr, tpr, _ = metrics.roc_curve(lids_labels , lids_list )
precision, recall, thresholds = metrics.precision_recall_curve(lids_labels, lids_list)
print(metrics.auc(recall, precision))
auc = str(metrics.auc(recall, precision))[:5]
plt.plot(recall,precision,label="Weighted Hamming LIDs, auc="+str(auc))


precision, recall, thresholds = metrics.precision_recall_curve(knns_weighted_labels, knns_weighted_list)
print(metrics.auc(recall, precision))
auc = str(metrics.auc(recall, precision))[:5]
plt.plot(recall,precision,label="Weighted KNN, auc="+str(auc))

precision, recall, thresholds = metrics.precision_recall_curve(knns_labels, knns_list)
print(metrics.auc(recall, precision))
auc = str(metrics.auc(recall, precision))[:5]

plt.plot(recall,precision,label="KNN, auc="+str(auc))

plt.legend(loc=4)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('images/pr_ton.pdf')