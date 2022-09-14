import numpy as np

def get_ids(file_name):
    #print(type(file_name))
    if type(file_name)==list:
        print(str(file_name)+' benign+malicious measurements')
        x=list(np.loadtxt('results/'+str(file_name[0])+'_ids_3.txt'))+list(np.loadtxt('results/'+str(file_name[1])+'_ids_3.txt'))
    else:
        print(str(file_name)+' benign measurements')
        x = list(np.loadtxt('results/'+str(file_name)+'_ids_3.txt'))
    print('size')
    print(len(x))

    nonzeros=np.count_nonzero(x)
    zeros=len(x)-nonzeros
    print('Exact Match')
    print(zeros/len(x))
    x=np.array(x)
    x=x[x!=0]
    print('id 3')
    print(np.mean(x))

    if type(file_name)==list:

        x=list(np.loadtxt('results/'+str(file_name[0])+'_ids_5.txt'))+list(np.loadtxt('results/'+str(file_name[1])+'_ids_5.txt'))
    else:
        x = list(np.loadtxt('results/'+str(file_name)+'_ids_5.txt'))
    x=np.array(x)
    x=x[x!=0]
    print('id 5')
    print(np.mean(x))

    if type(file_name)==list:

        x=list(np.loadtxt('results/'+str(file_name[0])+'_ids_10.txt'))+list(np.loadtxt('results/'+str(file_name[1])+'_ids_10.txt'))
    else:
        x = list(np.loadtxt('results/'+str(file_name)+'_ids_10.txt'))
    x=np.array(x)
    x=x[x!=0]
    print('id 10')
    print(np.mean(x))

    if type(file_name)==list:

        x=list(np.loadtxt('results/'+str(file_name[0])+'_ids_20.txt'))+list(np.loadtxt('results/'+str(file_name[1])+'_ids_20.txt'))
    else:
        x = list(np.loadtxt('results/'+str(file_name)+'_ids_20.txt'))
    x=np.array(x)
    x=x[x!=0]
    print('id 20')
    print(np.mean(x))


get_ids('Malware-Capture-Group-3_benign')
#get_ids('CTU-Honeypot-Capture-7-1_benign')
#get_ids(['unsw_nb15_benign','unsw_nb15_mal'])

