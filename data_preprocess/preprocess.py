import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, QuantileTransformer, KBinsDiscretizer
import torch
import torch.nn.functional as F
#from data_preprocess.drop_columns.ton_iot import datatypes
from scipy.stats import entropy

class Preprocess():
    def __init__(self, df, datatypes):
        self.df=df
        self.new_df=pd.DataFrame()
        self.encoders={}
        self.datatypes=datatypes


    def df_filter(self):
        if 'label string' in self.datatypes:
            self.benign_df=self.new_df[self.new_df['label string'].str.lower()=='benign']
            self.malicious_df=self.new_df[self.new_df['label string'].str.lower()!='benign']
        elif 'class' in self.datatypes:
            self.benign_df=self.new_df[self.new_df['class'].str.lower()=='normal']
            self.malicious_df=self.new_df[self.new_df['class'].str.lower()!='normal']
        elif 'Attack' in self.datatypes:
            self.benign_df=self.new_df[self.new_df['Label']==0]
            self.malicious_df=self.new_df[self.new_df['Label']!=0]
        elif 'label' in self.datatypes:
            self.benign_df=self.new_df[self.new_df['label']==0]
            self.malicious_df=self.new_df[self.new_df['label']==1]
        else:
            self.benign_df=self.new_df
            self.malicious_df=self.new_df

        self.benign_df=self.benign_df.drop(columns=[col for col in self.new_df if col in self.datatypes], axis=1)
        self.malicious_df=self.malicious_df.drop(columns=[col for col in self.new_df if col in self.datatypes], axis=1)

    def preprocess(self):
        def preprocess_type(col):

            if col=='label' or col=='ts' or col=='type' or col=='label string' or col in self.datatypes:
                return 'default'
            elif (self.df[col].dtype == int or self.df[col].dtype == float) and (len(self.df[col].unique()) >= 60):
                return 'minmax'
            else:
                return 'categorical'


        if 'label' in self.datatypes:
            temp_df=self.df[self.df['label']==0]

        elif 'Label' in self.datatypes:
            temp_df=self.df[self.df['Label']==0]
        elif 'class' in self.datatypes:
            temp_df = self.df[self.df['class'].str.lower() == 'normal']
        elif 'label string' in self.datatypes:
            temp_df=self.df[self.df['label string'].str.lower()=='benign']
        else:
            temp_df=self.df

        print(self.df.shape)
        print(temp_df.shape)
        sys.exit()

        for col in self.df.columns:
            type_=preprocess_type(col)
            if type_=='default':
                self.new_df[col] = self.df[col]
                self.encoders[col]={}
                self.encoders[col]['type']=None
                self.encoders[col]['encoder']=None
            elif type_=='categorical':

                enc = LabelEncoder()
                enc.fit(temp_df[col].astype(str).values)
                le_dict = dict(zip(enc.classes_, enc.transform(enc.classes_)))

                label_to_num=self.df[col].astype(str).apply(lambda x: le_dict.get(x, len(enc.classes_)))



                self.new_df[col]=label_to_num
                self.new_df[col]=self.new_df[col].astype('category')

                self.encoders[col]={}
                self.encoders[col]['type']='categorical'
                self.encoders[col]['encoder']=enc
                #self.encoders[col]['n_classes']=len(enc.classes_)+1
                self.encoders[col]['n_classes']=len(np.unique(label_to_num))
            elif type_=='minmax':
                enc = MinMaxScaler()

                enc.fit(np.expand_dims(temp_df[col].values, axis=1))
                min_max_values =enc.transform(np.expand_dims(self.df[col].values, axis=1))
                self.new_df[col]=min_max_values.flatten()

                self.encoders[col]={}
                self.encoders[col]['type']='continuous'
                self.encoders[col]['encoder']=enc

        self.df_filter()
