import matplotlib.pyplot as plt
import random
from scipy.stats import entropy
import numpy as np
from scipy.spatial.distance import cdist
from util.lid import calculate_lid
from util.knn import calculate_knn


data=np.array([[2, 0],
     [3, 0],
[-1, 3],
[1, 3],[1, 3],[1, 3],[1, 3],
[1, 1],[1, 1],
 ])

data_in=np.array([[0,0]])
data_out=np.array([[1,3]])
data_out=np.array([[1,2]])

numerator=2
entropys=[]
x1_vals, x1_counts=np.unique(data[:,0],return_counts=True)
x2_vals, x2_counts=np.unique(data[:,1],return_counts=True)
entropys.append(numerator/entropy(x1_counts))
entropys.append(numerator/entropy(x2_counts))
entropys=np.array(entropys)

data_in_pairwise=cdist(data_in,data, metric='hamming', w=entropys)
data_in_pairwise=np.sort(data_in_pairwise,axis=1)
data_out_pairwise=cdist(data_out,data, metric='hamming', w=entropys)
data_out_pairwise=np.sort(data_out_pairwise,axis=1)

#data_in_pairwise=cdist(data_in,data, metric='hamming', w=None)
#data_out_pairwise=cdist(data_out,data, metric='hamming', w=None)

k=3
lids_in = calculate_lid(data_in_pairwise, k_=k)
lids_out = calculate_lid(data_out_pairwise, k_=k)

knns_in = calculate_knn(data_in_pairwise, k_=k, unique_=False)
knns_out = calculate_knn(data_out_pairwise, k_=k, unique_=False)


print(lids_in)
print(lids_out)
print(knns_in)
print(knns_out)



data_graph=data=np.array([[2, 0],[3, 0],
[-1, 3],
[.94, 3],[.98, 3],[1.02, 3],[1.06, 3],
[.98, 1],[1.02, 1],
 ])
data_graph=np.array(data_graph)

plt.clf()
fig, ax = plt.subplots()

data_connection0=np.array([list(data_in[0]),list(data_graph[0])])
line=ax.plot(data_connection0[:,0],data_connection0[:,1], '--', c='#1f77b4', linewidth=1 )
ax.annotate('H=0.498', xy=(1.5,0.06),)


data_connection0=np.array([list(data_in[0]),list(data_graph[-1])])
line=ax.plot(data_connection0[:,0],data_connection0[:,1], '--', c='#1f77b4', linewidth=1 )
ax.annotate('H=1', xy=(.68,0.5),)

#data_connection0=np.array([[1.0,1.9],[1,1]])
#line=ax.plot(data_connection0[:,0],data_connection0[:,1], '--', c='red', linewidth=1 )
#ax.annotate('asdf', xy=(1.05,1.5),)

data_connection0=np.array([[1.0,2.1],[1,3]])
line=ax.plot(data_connection0[:,0],data_connection0[:,1], '--', c='red', linewidth=1 )
ax.annotate('H=0.502', xy=(1.05,2.5),)

data_connection0=np.array([[.95, 2.0],[-1,3]])
line=ax.plot(data_connection0[:,0],data_connection0[:,1], '--', c='red', linewidth=1 )
ax.annotate('H=1', xy=(0.1,2.5),)


data_connection1=np.array([list(data_in[0]),list(data_graph[1])])
x=np.linspace(data_connection1[0][0],np.pi,100)
y=np.sin(x)/4
x=3*((x-np.min(x))/(np.max(x)-np.min(x)))
line=ax.plot(data_connection1[:,0],data_connection1[:,1], '.', c='#1f77b4',linewidth=1  )
ax.plot(x,y, '--', c='#1f77b4',linewidth=1  )
ax.annotate('H=0.498', xy=(2.5,0.2),)


ax.plot(data_graph[:,0], data_graph[:,1], '.',  c='#1f77b4')

ax.plot(data_in[:,0],data_in[:,1], '.', color='Black', markersize=10, label='Weighted Hamming KNN=0.665\n'
                                                                             'Weighted Hamming LID=1.435\n')

ax.plot(data_out[:,0],data_out[:,1], 'x', color='red', markersize=10, label='Weighted Hamming KNN=0.502\n'
                                                                             'Weighted Hamming LID=1.451\n')

plt.title('LID and KNN measurements in Simple 2D Example')
lines = plt.gca().get_lines()[-3:]
#include=[-3:]
legend1=plt.legend([lines[i] for i in range(len(lines))],['Training Example', 'Benign (Test)','Malicious (Test)',], loc=1)
plt.xlabel('X1')
plt.ylabel('X2')
lgd=plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.12),
          ncol=2, fancybox=True, shadow=True,)#title='LID and KNN values using Hamming and Weighted Hamming Distances.  '

plt.gca().add_artist(legend1)
#plt.show()
#lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1))
plt.savefig('images/teaser.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

