from pairwise_distances import *
from data_setup import df_to_np, calculate_weights
from util.id import calculate_id
from sklearn import metrics

from sklearn.model_selection import train_test_split
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

#def plot():

rootdir = "results2/id/"
def ids(file, k):
    regex = re.compile('{}_192.168.*_{}.csv'.format(file,k))
    avgs=[]
    for file in os.listdir(rootdir):
        #print(file)
        if regex.match(file):
           df=pd.read_csv(rootdir+file)
           vals=[i for i in df['value'].values if i!=0]
           if len(vals)>0:
               avg=np.mean(vals)
               avgs.append(avg)
           else:
               avgs.append(0)
    return avgs

def ids_csu(file, k):
    print('{}_*_{}.csv'.format(file,k))
    regex = re.compile('{}_.*_{}.csv'.format(file,k))
    avgs=[]
    for file in os.listdir(rootdir):
        #print(file)
        if regex.match(file):
           #print(file)
           df=pd.read_csv(rootdir+file)
           vals=[i for i in df['value'].values if i!=0]
           if len(vals)>0:
               avg=np.mean(vals)
               print(file)
               print(avg)
               avgs.append(avg)
           else:
               avgs.append(0)
    return avgs


k=3
file='iot23_benign_ids_expanded'
iot23=ids(file, k)
iot23.extend([.67,.68,1])
file='nf_bot_iot_benign_ids_expanded'
nf_bot=ids(file, k)
file='ton_iot_benign_ids_expanded'
toniot=ids(file, k)

plt.clf()
plt.boxplot([iot23, nf_bot, toniot],labels=['IoT23', 'NF Bot IoT','TON IoT',])
#plt.set_xticklabels(['IoT23', 'NF Bot IoT','TON IoT',])
for i, j in zip([1,2,3],[iot23, nf_bot, toniot]):
    # Add some random "jitter" to the x-axis

    if i==1:
        x = np.random.normal(i, 0.04, size=len(j))
        plt.plot(x[:-3], j[:-3], 'r.', alpha=0.2)
        plt.plot([0.95,1.05,1], j[-3:],'r.', alpha=0.3)
        plt.annotate("Echo", (x[-1], j[-1]), textcoords="offset points", xytext=(10, -5), )
        plt.annotate("HUE", (1.05, j[-2]), textcoords="offset points",xytext=(5,0),)
        plt.annotate("Door Lock", (.95, j[-3]), textcoords="offset points",xytext=(-45,4), )
    else:
        x = np.random.normal(i, 0.04, size=len(j))
        plt.plot(x, j, 'r.', alpha=0.2)
plt.title('Device ID with K=5')
plt.show()
plt.savefig('graphs/boxplot_k5.pdf')

k=5
file='iot23_benign_ids_expanded'
iot23=ids(file, k)
iot23.extend([.81,.73,1.14])
file='nf_bot_iot_benign_ids_expanded'
nf_bot=ids(file, k)
file='ton_iot_benign_ids_expanded'
toniot=ids(file, k)
file='csu_benign_ids_expanded'
csuiot=ids_csu(file, k)

k=10
file='iot23_benign_ids_expanded'
iot23=ids(file, k)
iot23.extend([0.92, .69, 1.35])
file='nf_bot_iot_benign_ids_expanded'
nf_bot=ids(file, k)
file='ton_iot_benign_ids_expanded'
toniot=ids(file, k)

k=20
file='iot23_benign_ids_expanded'
iot23=ids(file, k)
iot23.extend([0.94,.63, 1.4])
file='nf_bot_iot_benign_ids_expanded'
nf_bot=ids(file, k)
file='ton_iot_benign_ids_expanded'
toniot=ids(file, k)




plt.clf()
plt.boxplot([iot23, csuiot],labels=['IoT23', 'CSU',])
#plt.set_xticklabels(['IoT23', 'NF Bot IoT','TON IoT',])
for i, j in zip([1,2,],[iot23, csuiot, ]):
    # Add some random "jitter" to the x-axis

    if i==1:
        x = np.random.normal(i, 0.04, size=len(j))
        plt.plot(x[:-3], j[:-3], 'r.', alpha=0.2)
        plt.plot([0.95,1.05,1], j[-3:],'r.', alpha=0.3)
        plt.annotate("Echo", (x[-1], j[-1]), textcoords="offset points", xytext=(10, -5), )
        plt.annotate("HUE", (1.05, j[-2]), textcoords="offset points",xytext=(5,0),)
        plt.annotate("Door Lock", (.95, j[-3]), textcoords="offset points",xytext=(-45,4), )
    else:
        x = np.random.normal(i, 0.04, size=len(j))
        plt.plot(x, j, 'r.', alpha=0.2)
plt.title('Device ID with K=5')
plt.show()
plt.savefig('graphs/boxplot_k5.pdf')
