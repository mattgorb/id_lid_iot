import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

df_benign=list(np.loadtxt('results2/nf_bot_iot_benign_lids_5.txt'))

df_mal=pd.read_csv('results2/nf_bot_5_attacks.csv')

#print(metrics.roc_auc_score(labels, ood_lids+in_dist_lids))
#precision, recall, thresholds = metrics.precision_recall_curve(labels, ood_lids+in_dist_lids)
#print(metrics.auc(recall, precision))

num=500000
plt.clf()
#np.random.shuffle(df_benign)
for attack in list(df_mal['Attack'].unique()):
    print(attack)
    vals=df_mal[df_mal['Attack']==attack]['value'].values
    #vals = [i for i in vals if i < 10]
    print(len(vals))
    #print(vals)
    #vals.sort()
    print(vals)
    #np.random.shuffle(vals)
    vals=vals[:num]
    #plt.plot([i for i in range(len(vals))],vals, label=attack )
    plt.hist( vals, label=attack)

#plt.clf()
#df_benign.sort()
#df_benign=[i for i in df_benign if i>0]
df_benign=df_benign[-num:]
print('benign')
print(df_benign)
#plt.plot([i for i in range(len(df_benign))], df_benign, label='Benign')
plt.hist(df_benign, label='Benign')
plt.legend()
plt.show()