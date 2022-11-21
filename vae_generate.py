from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import math

from data_setup import df_to_np, calculate_weights
from util.knn import calculate_knn
from sklearn import metrics

from pairwise_distances import *
from data_setup import df_to_np, calculate_weights
from util.lid import calculate_lid, calculate_exactmatch

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=250, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--prior', type=str, default='standard', metavar='N',
                    help='prior')
parser.add_argument('--dataset', type=str, default=None, metavar='N',
                    help='prior')

parser.add_argument('--run_benign', type=bool, default=False, metavar='N',
                    help='prior')
args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda:5" if args.cuda else "cpu")
print(device)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



def compute_embedding_size(n_categories):
    val = min(600, round(1.6 * n_categories**0.56))
    return int(val)

dataset=args.dataset
run_benign = args.run_benign


weights=False

embeddings=[]
cat_out=[]
cont_dim=0
input_dim=0

base_dir='/s/luffy/b/nobackup/mgorb/iot'



if dataset=='ton_iot':
    from data_preprocess.drop_columns import ton_iot
    benign_np, preprocess, float_cols, categorical_cols=df_to_np(f'{base_dir}/ton_iot/Train_Test_Network.csv',ton_iot.datatypes, train_set=True, return_preprocess=True)
    mal_np=df_to_np(f'{base_dir}/ton_iot/Train_Test_Network.csv', ton_iot.datatypes,train_set=False, return_preprocess=False)


    if run_benign:
        X_train, X_test =benign_np, benign_np
    else:
        X_train, X_test = mal_np, mal_np

    X_train = X_train.astype('float64')
    cont_dim=len(float_cols)



    for col in range(len(categorical_cols)):
        n_cats = preprocess.encoders[categorical_cols[col]]['n_classes']#len(preprocess.encoders[categorical_cols[col]]['encoder'].classes_)

        embed_dim = compute_embedding_size(n_cats)

        embed_layer = torch.nn.Embedding(n_cats, embed_dim).to(device)
        embeddings.append(embed_layer)
        input_dim += embed_dim
        cat_out.append(n_cats)

    input_dim+=cont_dim
    #print(input_dim)
elif dataset=='iot23':
    from data_preprocess.drop_columns import iot23

    benign_np, preprocess, float_cols, categorical_cols = df_to_np( f'{base_dir}/iot23/iot23_sample_with_real.csv', iot23.datatypes, train_set=True, return_preprocess=True)

    mal_np = df_to_np( f'{base_dir}/iot23/iot23_sample_with_real.csv', iot23.datatypes, train_set=False)

    if run_benign:
        X_train, X_test =benign_np, benign_np
    else:
        X_train, X_test = mal_np, mal_np



    X_train = X_train.astype('float64')
    cont_dim=len(float_cols)
    for col in range(len(categorical_cols)):
        n_cats = preprocess.encoders[categorical_cols[col]]['n_classes']#len(preprocess.encoders[categorical_cols[col]]['encoder'].classes_)

        embed_dim = compute_embedding_size(n_cats)
        embed_layer = torch.nn.Embedding(n_cats, embed_dim).to(device)
        embeddings.append(embed_layer)
        input_dim += embed_dim
        cat_out.append(n_cats)


    input_dim+=cont_dim
elif dataset=='nf_bot_iot':
    from data_preprocess.drop_columns import nf_bot_iot
    benign_np , preprocess, float_cols, categorical_cols=df_to_np(f'{base_dir}/nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes,train_set=True, return_preprocess=True)
    mal_np=df_to_np(f'{base_dir}/nf_bot_iot/NF-BoT-IoT.csv',  nf_bot_iot.datatypes,train_set=False)

    if run_benign:
        X_train, X_test =benign_np, benign_np
    else:
        X_train, X_test = mal_np, mal_np

    X_train = X_train.astype('float64')
    cont_dim=len(float_cols)
    for col in range(len(categorical_cols)):
        n_cats = preprocess.encoders[categorical_cols[col]]['n_classes']#len(preprocess.encoders[categorical_cols[col]]['encoder'].classes_)

        embed_dim = compute_embedding_size(n_cats)
        embed_layer = torch.nn.Embedding(n_cats, embed_dim).to(device)
        embeddings.append(embed_layer)
        input_dim += embed_dim
        cat_out.append(n_cats)

    input_dim+=cont_dim

elif dataset=='unsw_nb15':
    from data_preprocess.drop_columns import unsw_n15
    benign_np , preprocess, float_cols, categorical_cols=df_to_np('/s/luffy/b/nobackup/mgorb/iot/unsw-nb15/UNSW_NB15_training-set.csv', unsw_n15.datatypes,train_set=True, return_preprocess=True)
    #benign_np_test , _, _, _=df_to_np('/s/luffy/b/nobackup/mgorb/iot/unsw-nb15/UNSW_NB15_testing-set.csv', unsw_n15.datatypes,train_set=True, return_preprocess=True)

    mal_np=df_to_np('/s/luffy/b/nobackup/mgorb/iot/unsw-nb15/UNSW_NB15_training-set.csv',  unsw_n15.datatypes,train_set=False)

    if run_benign:
        X_train, X_test =benign_np, benign_np
    else:
        X_train, X_test = mal_np, mal_np

    X_train = X_train.astype('float64')
    cont_dim=len(float_cols)
    for col in range(len(categorical_cols)):
        n_cats = preprocess.encoders[categorical_cols[col]]['n_classes']#len(preprocess.encoders[categorical_cols[col]]['encoder'].classes_)
        embed_dim = compute_embedding_size(n_cats)
        embed_layer = torch.nn.Embedding(n_cats, embed_dim).to(device)
        embeddings.append(embed_layer)
        input_dim += embed_dim
        cat_out.append(n_cats)
    input_dim+=cont_dim

elif dataset=='kaggle_nid':
    from data_preprocess.drop_columns import kaggle_nid
    benign_np , preprocess, float_cols, categorical_cols =df_to_np('/s/luffy/b/nobackup/mgorb/iot/kaggle_nid/Train_data.csv', kaggle_nid.datatypes,train_set=True, return_preprocess=True)
    mal_np=df_to_np('/s/luffy/b/nobackup/mgorb/iot/kaggle_nid/Train_data.csv',  kaggle_nid.datatypes,train_set=False)


    if run_benign:

        X_train, X_test =benign_np, benign_np
    else:
        X_train, X_test = mal_np, mal_np

    cont_dim=len(float_cols)
    for col in range(len(categorical_cols)):
        n_cats = preprocess.encoders[categorical_cols[col]]['n_classes']#len(preprocess.encoders[categorical_cols[col]]['encoder'].classes_)

        embed_dim = compute_embedding_size(n_cats)
        embed_layer = torch.nn.Embedding(n_cats, embed_dim).to(device)
        embeddings.append(embed_layer)
        input_dim += embed_dim
        cat_out.append(n_cats)

    input_dim+=cont_dim



class VAE(nn.Module):
    def __init__(self,input_dim, embeddings, cats_out, cont_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)

        #self.fc2 = nn.Linear(128, 64)
        self.enc_mu = torch.nn.Linear(64, 8)
        self.enc_log_sigma = torch.nn.Linear(64, 8)

        self.fc3 = nn.Linear(8, 64)
        self.fc4 = nn.Linear(64, cont_dim)
        self.embeddings=torch.nn.ModuleList(embeddings)
        self.cats_out = []
        for cat in cats_out:
            self.cats_out.append(torch.nn.Linear(64, cat))
        self.cats_out = torch.nn.ModuleList(self.cats_out)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.enc_mu(h1),self.enc_log_sigma(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))

        out_cont=torch.sigmoid(self.fc4(h3))
        cat_outs=[]
        for cat in self.cats_out:
            cat_outs.append(cat(h3))

        return out_cont, cat_outs


    def forward(self, x):
        num_fts=x.size(1)-len(self.embeddings)
        embed_list=[]#torch.nn.ModuleList()
        for embed in range(len(self.embeddings)):
            out=self.embeddings[embed](x[:,num_fts+embed].long())
            embed_list.append(out)

        embedded_cats = torch.cat(embed_list, dim=1)
        inputs=torch.cat([embedded_cats.float(),x[:,:num_fts].float()], dim=1)
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        out_cont, cat_outs=self.decode(z)
        return out_cont, cat_outs, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(out_cont, cat_outs, data, mu, logvar, reduction='sum'):
    if reduction=='none':
        recon_loss = F.mse_loss(out_cont.double(), data[:, :out_cont.size(1)].double(), reduction='none')
        recon_loss=torch.sum(recon_loss, dim=1)

        for cat in range(len(cat_outs)):
            target=data[:,out_cont.size(1)+cat].long()
            recon_loss += F.cross_entropy(cat_outs[cat], target, reduction=reduction)
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    else:
        recon_loss = F.mse_loss(out_cont.double(), data[:, :out_cont.size(1)].double(), reduction=reduction)

        for cat in range(len(cat_outs)):
            target=data[:,out_cont.size(1)+cat].long()
            recon_loss += F.cross_entropy(cat_outs[cat], target, reduction=reduction)


        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (recon_loss+KLD), recon_loss.double(), KLD

def train(epoch, ):
    model.train()
    train_loss = 0
    recon_loss=0
    kld_loss=0
    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.to(device)

        optimizer.zero_grad()
        out_cont, cat_outs , mu, logvar= model(data)
        loss, recon, kld = loss_function(out_cont, cat_outs, data, mu, logvar)

        loss.backward()

        train_loss += loss.item()
        recon_loss+=recon.item()
        kld_loss+=kld.item()


        optimizer.step()


    info = {
        'loss': train_loss/len(train_dataloader.dataset),
        'recon_loss': recon_loss/len(train_dataloader.dataset),
        'kld': kld_loss/len(train_dataloader.dataset)
    }
    print(f"====> Epoch {epoch}: {info}")

def test(best_loss ):
    model.eval()
    train_loss = 0
    losses=[]


    out_cont_list=[]
    out_cat_list=[]

    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.to(device)

        out_cont, cat_outs, mu, logvar = model(data)
        loss , recon, kld= loss_function(out_cont, cat_outs, data, mu, logvar, reduction='none')
        losses.extend(loss.cpu().detach().numpy())



        output=None
        for cat in cat_outs:
            pred = cat.argmax(dim=1, keepdim=False)

            if output is None:
                output=torch.unsqueeze(pred, dim=1)
            else:
                output=torch.cat([output,torch.unsqueeze(pred, dim=1)], dim=1  )

        recon_loss = F.mse_loss(out_cont.double(), data[:, :out_cont.size(1)].double(), reduction='none')


        out_cont_final = torch.where(recon_loss < 0.0025, data[:, :out_cont.size(1)].double(), out_cont.double())


        out_cat_list.extend(output.cpu().detach().numpy())
        out_cont_list.extend(out_cont_final.cpu().detach().numpy())

    if np.mean(losses)<best_loss:
        print("Generating new synthetic examples...")
        best_loss=np.mean(losses)
        recon_syn=np.concatenate([np.array(out_cont_list),np.array(out_cat_list)],axis=1)
        np.save(f"{base_dir}/vae/recon_benign_{run_benign}_ds_{dataset}.npy",recon_syn)

        out_cont_list=[]
        out_cat_list=[]
        for batch_idx, (data, _) in enumerate(train_dataloader):
            data = data.to(device)

            sample = torch.randn(256, 8).to(device)
            out_cont, cat_outs = model.decode(sample)  # .cpu()

            num_fts = data.size(1) - len(cat_outs)
            da = data[:, :num_fts].float()

            print(da[0,:].size())
            print(out_cont.size())
            print(train_dataloader.dataset.size())
            #print(torch.cdist(da[:,:],train_dataloader.dataset[:,:num_fts]))
            sys.exit()
            output=None
            for cat in cat_outs:
                pred = cat.argmax(dim=1, keepdim=False)

                if output is None:
                    output=torch.unsqueeze(pred, dim=1)
                else:
                    output=torch.cat([output,torch.unsqueeze(pred, dim=1)], dim=1  )

            out_cat_list.extend(output.cpu().detach().numpy())
            out_cont_list.extend(out_cont.cpu().detach().numpy())
        np.save(f"{base_dir}/vae/syn_benign_{run_benign}_ds_{dataset}.npy", recon_syn)
        #print(out_cont[0, :].double())
    return best_loss

    #sys.exit()
'''
def test_backup(best_loss ):
    model.eval()
    train_loss = 0
    losses=[]

    out_cont_list=[]
    out_cat_list=[]

    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.to(device)

        out_cont, cat_outs, mu, logvar = model(data)
        loss , recon, kld= loss_function(out_cont, cat_outs, data, mu, logvar, reduction='none')
        losses.extend(loss.cpu().detach().numpy())

        output=None
        for cat in cat_outs:
            pred = cat.argmax(dim=1, keepdim=False)

            if output is None:
                output=torch.unsqueeze(pred, dim=1)
            else:
                output=torch.cat([output,torch.unsqueeze(pred, dim=1)], dim=1  )

        out_cat_list.extend(output.cpu().detach().numpy())
        out_cont_list.extend(out_cont.cpu().detach().numpy())

    loss=np.mean(np.array(losses))
    if loss<best_loss:
        best_loss=loss
        df=pd.DataFrame()
        for col in range(len(float_cols)):
            data_normalizer = preprocess.encoders[float_cols[col]]['encoder']
            transformed_data=data_normalizer.inverse_transform(np.array(out_cont_list)[:,col].reshape(-1,1))

            df[float_cols[col]]=transformed_data[:,0]

        for col in range(len(categorical_cols)):
            data_normalizer=preprocess.encoders[categorical_cols[col]]['encoder']
            transformed_data=data_normalizer.inverse_transform(np.array(out_cat_list)[:,col].astype(int))
            df[categorical_cols[col]]=transformed_data
            #print(preprocess.encoders[categorical_cols[col]]['encoder'].inverse_transform(benign_np[:100,len(float_cols)].astype(int)))
        df.to_csv(f"{base_dir}/vae/benign_{run_benign}.csv")

        #randomly sample gaussian for synthetic examples.
        for i in range(len(train_dataloader)):
            #data = data.to(device)
            sample = torch.randn(256, 64).to(device)
            out_cont, cat_outs = model.decode(sample)#.cpu()

            #out_cont, cat_outs, mu, logvar = model(sample)
            #loss, recon, kld = loss_function(out_cont, cat_outs, data, mu, logvar, reduction='none')
            #losses.extend(loss.cpu().detach().numpy())

            output = None
            for cat in cat_outs:
                pred = cat.argmax(dim=1, keepdim=False)

                if output is None:
                    output = torch.unsqueeze(pred, dim=1)
                else:
                    output = torch.cat([output, torch.unsqueeze(pred, dim=1)], dim=1)

            out_cat_list.extend(output.cpu().detach().numpy())
            out_cont_list.extend(out_cont.cpu().detach().numpy())

        df=pd.DataFrame()
        for col in range(len(float_cols)):
            data_normalizer = preprocess.encoders[float_cols[col]]['encoder']
            transformed_data=data_normalizer.inverse_transform(np.array(out_cont_list)[:,col].reshape(-1,1))

            df[float_cols[col]]=transformed_data[:,0]

        for col in range(len(categorical_cols)):
            data_normalizer=preprocess.encoders[categorical_cols[col]]['encoder']
            transformed_data=data_normalizer.inverse_transform(np.array(out_cat_list)[:,col].astype(int))
            df[categorical_cols[col]]=transformed_data
            #print(preprocess.encoders[categorical_cols[col]]['encoder'].inverse_transform(benign_np[:100,len(float_cols)].astype(int)))
        df.to_csv(f"{base_dir}/vae/benign_{run_benign}_synthetic.csv")

    return best_loss
'''


feature_weights=calculate_weights(X_train)

y=torch.Tensor(np.ones(X_train.shape[0]))
X_train=X_train.astype('float64')
x=torch.from_numpy(X_train)
my_dataset = TensorDataset(x, y)
train_dataloader = DataLoader(my_dataset, batch_size=256)  # create your dataloader

print(train_dataloader.dataset[:])
sys.exit()



print('Train dataset length: {}'.format(len(train_dataloader.dataset)))
#print('Malicious dataset length: {}'.format(len(malicious_dataloader.dataset)))

best_loss=1e6

model = VAE(input_dim, embeddings, cat_out, cont_dim)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    train(epoch, )
    best_loss=test( best_loss)