from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--dataset', type=str, default=None,)
parser.add_argument('--k', type=int, default=0, )
parser.add_argument('--syn_type', type=str, default=None, )
parser.add_argument('--algorithm', type=str, default=None, )

args = parser.parse_args()



#in_dist_lids=list(np.loadtxt(f'/s/luffy/b/nobackup/mgorb/results/{args.dataset}_benign_{args.algorithm}_{args.syn_type}_'+str(args.k)+'.txt'))
#ood_lids=list(np.loadtxt(f'/s/luffy/b/nobackup/mgorb/results/{args.dataset}_mal_{args.algorithm}_{args.syn_type}_'+str(args.k)+'.txt'))

in_dist_lids=list(np.loadtxt(f'/s/luffy/b/nobackup/mgorb/results/{args.dataset}_benign_{args.algorithm}_real_testset'+str(args.k)+'.txt'))
ood_lids=list(np.loadtxt(f'/s/luffy/b/nobackup/mgorb/results/{args.dataset}_mal_{args.algorithm}_recon_testset_'+str(args.k)+'.txt'))

#in_dist_lids=pd.read_csv('results/unsw_nb15_benign_lids_expanded_'+str(k)+'.csv')['value'].values
#ood_lids=pd.read_csv('results/unsw_nb15_mal_lids_expanded_'+str(k)+'.csv')['value'].values
#in_dist_lids=pd.read_csv('results/results/ton_iot_benign_lids_expanded_'+str(k)+'.csv')['value'].values
#ood_lids=pd.read_csv('results/results/ton_iot_mal_lids_expanded_'+str(k)+'.csv')['value'].values
#ood_lids=list(ood_lids)
#in_dist_lids=list(in_dist_lids)
print(len(ood_lids))
ood_lids = [i for i in ood_lids if i != 0]
print(len(ood_lids))

print(np.mean(np.array(ood_lids)))
print(np.mean(np.array(in_dist_lids)))

#Results
labels=[1 for i in ood_lids]+[0 for i in in_dist_lids]
##print(len(in_dist_lids))
#print(len(ood_lids))
#print(len(ood_lids)+len(in_dist_lids))
#print(len(labels))
print("ROC")
print(metrics.roc_auc_score(labels, list(ood_lids)+list(in_dist_lids)))
precision, recall, thresholds = metrics.precision_recall_curve(labels, list(ood_lids)+list(in_dist_lids))
print("PR")
print(metrics.auc(recall, precision))

#print('f1')
#print(metrics.f1_score(labels, list(ood_lids)+list(in_dist_lids), ))