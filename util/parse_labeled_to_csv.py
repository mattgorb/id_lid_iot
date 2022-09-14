import pandas as pd
import csv
from pathlib import Path
import os

files=os.listdir('/Volumes/My Passport/opt/Malware-Project/BigDataset/IoTScenarios')
for file in files:
    for path in Path("/Volumes/My Passport/opt/Malware-Project/BigDataset/IoTScenarios/"+str(file)).rglob('conn.log.labeled'):
        fullpath=path
    if os.path.exists('./csv/iot23/'+str(file)+'.csv'):
        continue
    if file in ['CTU-Honeypot-Capture-4-1','CTU-Honeypot-Capture-5-1','CTU-Honeypot-Capture-7-1','CTU-IoT-Malware-Capture-1-1',
                'CTU-IoT-Malware-Capture-3-1','CTU-IoT-Malware-Capture-20-1','CTU-IoT-Malware-Capture-3-1','CTU-IoT-Malware-Capture-21-1',
                'CTU-IoT-Malware-Capture-34-1']:
        #print(file)
        #print(len(lines))
        continue
    file1 = open(fullpath, 'r')
    lines = file1.readlines()
    print(file)
    print(len(lines))
    #continue

    lines=lines[6:800000]
    lines = [x.replace('   ', '\x09') for x in lines]
    lines = [x.replace('\n', '') for x in lines]
    lines = [x.split('\x09') for x in lines]


    names= [i + str(" ") + j for i, j in zip(lines[0][1:], lines[1][1:])]
    lines=lines[2:]

    new_df = pd.DataFrame(columns=names, data=lines)
    new_df.to_csv('../csv/iot23/'+str(file)+'.csv')





