from data_preprocess.drop_columns import iot23
import pandas as pd
files = ['CTU-IoT-Malware-Capture-1-1', 'CTU-IoT-Malware-Capture-3-1', 'CTU-IoT-Malware-Capture-7-1',
                'CTU-IoT-Malware-Capture-8-1', 'CTU-IoT-Malware-Capture-9-1', 'CTU-IoT-Malware-Capture-17-1',
                'CTU-IoT-Malware-Capture-20-1', 'CTU-IoT-Malware-Capture-21-1',
    'CTU-IoT-Malware-Capture-34-1', 'CTU-IoT-Malware-Capture-33-1', 'CTU-IoT-Malware-Capture-35-1',
    'CTU-IoT-Malware-Capture-36-1', 'CTU-IoT-Malware-Capture-39-1', 'CTU-IoT-Malware-Capture-42-1', 'CTU-IoT-Malware-Capture-43-1',
                'CTU-IoT-Malware-Capture-44-1', 'CTU-IoT-Malware-Capture-48-1',
                'CTU-IoT-Malware-Capture-49-1', 'CTU-IoT-Malware-Capture-52-1', 'CTU-IoT-Malware-Capture-60-1',
                ]

df=None
for file in files:
    if df is None:
        df=pd.read_csv('csv/iot23/'+file+'.csv')
    else:
        df2=pd.read_csv('csv/iot23/'+file+'.csv')
        df=df.append(df2)
    print(df.shape)

print('length of full dataset:')
print(df.shape)

df_benign=df[df['label string']=='Benign']
print(df_benign.shape)
df_benign=df_benign[(df_benign['id.orig_h addr'].str.startswith('192.168.'))|(df_benign['id.resp_h addr'].str.startswith('192.168.'))]
print('filtered for 192.168.')
print(df_benign.shape)
x=list(df_benign['id.orig_h addr'].values)+list(df_benign['id.resp_h addr'].values)
x=[i for i in x if i.startswith('192.168.')]
print('ton num devices:')
devices=set(x)
print(len(devices))
for d in devices:
    device_count = df_benign[(df_benign['id.orig_h addr']==str(d)) | (df_benign['id.resp_h addr']==str(d))].shape[0]
    print('device count for {}: {} '.format(d,device_count))
    if device_count<50:
        df_benign=df_benign[(df_benign['id.orig_h addr']!=str(d)) & (df_benign['id.resp_h addr']!=str(d))]

print('filtered for multiple internal ips: ')
print(df_benign.shape)

df_benign=df_benign[df_benign['label string']=='Benign'].sample(n=500000, random_state=1)

df_mal=df[df['label string']!='Benign'].sample(n=500000, random_state=1)
df_benign=df_benign.append(df_mal)
print("HERE")
print(df_benign.shape)

'''files=['CTU-Honeypot-Capture-4-1','CTU-Honeypot-Capture-5-1','CTU-Honeypot-Capture-7-1']
for file in files:
    df2=pd.read_csv('csv/iot23/'+file+'.csv')
    df_benign=df_benign.append(df2)
    print(df_benign.shape)'''


import pandas as pd
df=pd.read_csv('csv/iot23/iot23_sample.csv')
files=['CTU-Honeypot-Capture-4-1','CTU-Honeypot-Capture-5-1','CTU-Honeypot-Capture-7-1']
for file in files:
    df2=pd.read_csv('csv/iot23/'+file+'.csv')
    df=df.append(df2)
    #print(df_benign.shape)

df.to_csv('csv/iot23/iot23_sample_with_real.csv',index=False)



'''df=pd.read_csv('csv/ton_iot/NF-ToN-IoT.csv')

df=df[df['Attack']=='Benign']
df_filtered=df[(df['IPV4_SRC_ADDR'].str.startswith('192.168.'))|(df['IPV4_DST_ADDR'].str.startswith('192.168.'))]
x=list(df_filtered['IPV4_SRC_ADDR'].values)+list(df_filtered['IPV4_DST_ADDR'].values)
x=[i for i in x if i.startswith('192.168.')]
print('ton num devices:')
devices=set(x)
print(len(devices))
for d in devices:
    device_count = df[(df['IPV4_SRC_ADDR']==str(d)) | (df['IPV4_DST_ADDR']==str(d))].shape[0]
    print('device count for {}: {} '.format(d,device_count))'''

df=pd.read_csv('csv/unsw-nb15/UNSW_NB15_training-set.csv')
df=df[df['label']==0]
print(df.shape)
df=pd.read_csv('csv/unsw-nb15/UNSW_NB15_training-set.csv')
df=df[df['label']!=0]
print(df.shape)

df=pd.read_csv('csv/kaggle_nid/Train_data.csv')
df=df[df['class']=='normal']
print(df.shape)
df=pd.read_csv('csv/kaggle_nid/Train_data.csv')
df=df[df['class']!='normal']
print(df.shape)

df=pd.read_csv('csv/nf_bot_iot/NF-BoT-IoT.csv')
print(df.shape)
df=df[df['Attack']=='Benign']
print(df.shape)
df_filtered=df[(df['IPV4_SRC_ADDR'].str.startswith('192.168.'))|(df['IPV4_DST_ADDR'].str.startswith('192.168.'))]
x=list(df_filtered['IPV4_SRC_ADDR'].values)+list(df_filtered['IPV4_DST_ADDR'].values)
x=[i for i in x if i.startswith('192.168.')]

print('ton num devices:')
devices=set(x)
print(len(devices))
for d in devices:
    device_count = df[(df['IPV4_SRC_ADDR']==str(d)) | (df['IPV4_DST_ADDR']==str(d))].shape[0]
    print('device count for {}: {} '.format(d,device_count))

df=pd.read_csv('csv/iot23/iot23_sample.csv')
print(df.shape)
df=df[df['label string']=='Benign']
df_filtered=df[(df['id.orig_h addr'].str.startswith('192.168.'))|(df['id.resp_h addr'].str.startswith('192.168.'))]
x=list(df_filtered['id.orig_h addr'].values)+list(df_filtered['id.resp_h addr'].values)
x=[i for i in x if i.startswith('192.168.')]
print('iot23 num devices:')
devices=set(x)
print(len(devices))
for d in devices:
    device_count = df[(df['id.orig_h addr']==str(d)) | (df['id.resp_h addr']==str(d))].shape[0]
    print('device count for {}: {} '.format(d,device_count))








df=pd.read_csv('csv/ton_iot/Train_Test_Network.csv')

df=df[df['label']==0]
#df=df[df['label']!=0]
print(df.shape)
df_filtered=df[(df['src_ip'].str.startswith('192.168.'))|(df['dst_ip'].str.startswith('192.168.'))]
print(df_filtered.shape)
x=list(df_filtered['src_ip'].values)+list(df_filtered['dst_ip'].values)
x=[i for i in x if i.startswith('192.168.')]
print('ton num devices:')
devices=set(x)
print(len(devices))
for d in devices:
    device_count = df[(df['src_ip']==str(d)) | (df['dst_ip']==str(d))].shape[0]
    print('device count for {}: {} '.format(d,device_count))