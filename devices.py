from pairwise_distances import *
from data_setup import df_to_np, calculate_weights
from util.id import calculate_id
from sklearn import metrics

from sklearn.model_selection import train_test_split
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from itertools import groupby




def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .05, ypos],
                      transform=ax.transAxes, color='gray')
    line.set_clip_on(False)
    ax.add_line(line)

def label_len(my_index,level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k,g in groupby(labels)]

def label_group_bar_table(ax, df):
    ypos = -.1
    scale = 1./df.index.size
    for level in range(df.index.nlevels)[::-1]:
        pos = 0
        for label, rpos in label_len(df.index,level):
            lxpos = (pos + .5 * rpos)*scale
            print(lxpos)
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
            add_line(ax, pos*scale, ypos)
            pos += rpos
        add_line(ax, pos*scale , ypos)
        ypos -= .1


df=pd.read_csv('devices.csv')
plt.clf()
fig, ax = plt.subplots()

plt.plot(df.X, df.ID, '^')

#plt.plot(df.X[:6], df.ID[:6], '^', color='green')
#plt.plot(df.X[6:9], df.ID[6:9], 'o', color='blue')
#plt.plot(df.X[9:], df.ID[9:], 'x', color='red')
ax.annotate(df['Device'].values[0], xy=(df.X[0], df.ID[0]), xytext=(5,-6), textcoords='offset pixels')
ax.annotate(df['Device'].values[1], xy=(df.X[1], df.ID[1]), xytext=(5,-4), textcoords='offset pixels')
ax.annotate(df['Device'].values[2], xy=(df.X[2], df.ID[2]), xytext=(-6,-12), textcoords='offset pixels')
ax.annotate(df['Device'].values[3], xy=(df.X[3], df.ID[3]), xytext=(-16,6), textcoords='offset pixels')
ax.annotate(df['Device'].values[4], xy=(df.X[4], df.ID[4]), xytext=(5,-6), textcoords='offset pixels')
ax.annotate(df['Device'].values[5], xy=(df.X[5], df.ID[5]), xytext=(0,5), textcoords='offset pixels')

ax.annotate(df['Device'].values[6], xy=(df.X[6], df.ID[6]), xytext=(-2,-18), textcoords='offset pixels')
ax.annotate(df['Device'].values[7], xy=(df.X[7], df.ID[7]), xytext=(7,4), textcoords='offset pixels')
ax.annotate(df['Device'].values[8], xy=(df.X[8], df.ID[8]), xytext=(-2,11), textcoords='offset pixels')

ax.annotate(df['Device'].values[9], xy=(df.X[9], df.ID[9]), xytext=(-8,-14), textcoords='offset pixels')
ax.annotate(df['Device'].values[10], xy=(df.X[10], df.ID[10]), xytext=(-2,-15), textcoords='offset pixels')
ax.annotate(df['Device'].values[11], xy=(df.X[11], df.ID[11]), xytext=(-45,-12), textcoords='offset pixels')
ax.annotate(df['Device'].values[12], xy=(df.X[12], df.ID[12]), xytext=(-55,-4), textcoords='offset pixels')
ax.annotate(df['Device'].values[13], xy=(df.X[13], df.ID[13]), xytext=(-40,7), textcoords='offset pixels')
ax.annotate(df['Device'].values[14], xy=(df.X[14], df.ID[14]), xytext=(-8,7), textcoords='offset pixels')
#ax.annotate(df['Device'].values[15], xy=(df.X[15], df.ID[15]), xytext=(-70,0), textcoords='offset pixels')
ax.annotate(df['Device'].values[15], xy=(df.X[15], df.ID[15]), xytext=(-55,-5), textcoords='offset pixels')
ax.annotate(df['Device'].values[16], xy=(df.X[16], df.ID[16]), xytext=(-50,-17), textcoords='offset pixels')


'''for i,j, k,l in zip(df.X,df.ID,df.Category,df.Device ):
    #plt.plot(i,j,'', color=)
    for a in [-4, 4]:
    ax.annotate(l, xy=(i,j), xytext=(2,0), textcoords='offset pixels')'''


'''df=df.set_index('Category')
ax.set_xticklabels('')
ax.set_xlabel('')
ax.axes.xaxis.set_visible(False)
ypos = -.05
ax.text(0.1764705882352941, ypos, 'Low', ha='center', transform=ax.transAxes)
add_line(ax, 0.33, ypos)
ax.text(0.5, ypos, 'Medium', ha='center', transform=ax.transAxes)
add_line(ax, 0.66, ypos)
ax.text(0.825, ypos, 'High', ha='center', transform=ax.transAxes)'''
#add_line(ax, 1, ypos)
#
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylabel('Intrinsic Dimensionality')
plt.tight_layout()
#plt.show()
plt.savefig('devices.pdf')

#0.1764705882352941
#0.4411764705882353
#0.7647058823529411