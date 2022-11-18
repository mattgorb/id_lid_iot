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

import math

from data_setup import df_to_np, calculate_weights
from util.knn import calculate_knn
from sklearn import metrics


parser = argparse.ArgumentParser(description='VAE MNIST Example')
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

args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda:4" if args.cuda else "cpu")
print(device)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



def compute_embedding_size(n_categories):
    val = min(600, round(1.6 * n_categories**0.56))
    return int(val)

dataset='unsw_nb15'
weights=False

embeddings=[]
cat_out=[]
cont_dim=0
input_dim=0

#directory=

if dataset=='ton_iot':
    from data_preprocess.drop_columns import ton_iot
    benign_np, preprocess, float_cols, categorical_cols=df_to_np('csv/ton_iot/Train_Test_Network.csv',ton_iot.datatypes, train_set=True, return_preprocess=True)
    mal_np=df_to_np('csv/ton_iot/Train_Test_Network.csv', ton_iot.datatypes,train_set=False, return_preprocess=False)
    #X_train, X_test = train_test_split(benign_np, test_size = 0.01, random_state = 42)
    X_train, X_test =benign_np, benign_np
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

    benign_np, preprocess, float_cols, categorical_cols = df_to_np( '/s/luffy/b/nobackup/mgorb/iot/iot23/iot23_sample_with_real.csv', iot23.datatypes, train_set=True, return_preprocess=True)

    mal_np = df_to_np( '/s/luffy/b/nobackup/mgorb/iot/iot23/iot23_sample_with_real.csv', iot23.datatypes, train_set=False)

    X_train, X_test = benign_np, benign_np
    feature_weights = calculate_weights(X_train)

    X_train, X_test = benign_np, benign_np

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
    benign_np , preprocess, float_cols, categorical_cols=df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes,train_set=True, return_preprocess=True)
    mal_np=df_to_np('csv/nf_bot_iot/NF-BoT-IoT.csv',  nf_bot_iot.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np

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
    benign_np_test , _, _, _=df_to_np('/s/luffy/b/nobackup/mgorb/iot/unsw-nb15/UNSW_NB15_testing-set.csv', unsw_n15.datatypes,train_set=True, return_preprocess=True)

    mal_np=df_to_np('/s/luffy/b/nobackup/mgorb/iot/unsw-nb15/UNSW_NB15_testing-set.csv',  unsw_n15.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np_test

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
    benign_np =df_to_np('csv/kaggle_nid/Train_data.csv', kaggle_nid.datatypes,train_set=True)
    mal_np=df_to_np('csv/kaggle_nid/Train_data.csv',  kaggle_nid.datatypes,train_set=False)
    X_train, X_test =benign_np, benign_np

    feature_weights=calculate_weights(X_train)




class AE(nn.Module):
    def __init__(self,input_dim, embeddings, cats_out, cont_dim):
        super(AE, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)

        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 64)
        self.fc4 = nn.Linear(64, cont_dim)
        self.embeddings=torch.nn.ModuleList(embeddings)
        self.cats_out = []
        for cat in cats_out:
            self.cats_out.append(torch.nn.Linear(64, cat))
        self.cats_out = torch.nn.ModuleList(self.cats_out)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)


    def decode(self, z):
        h3 = F.relu(self.fc3(z))

        out_cont=torch.sigmoid(self.fc4(h3))
        cat_outs=[]#torch.nn.ModuleList()
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
        z = self.encode(inputs)
        return self.decode(z)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(out_cont, cat_outs, data, reduction='sum'):
    loss=F.mse_loss(out_cont.double(), data[:,:out_cont.size(1)].double(), reduction=reduction)
    if reduction=='none':
        loss=torch.sum(loss, dim=1)
    for cat in range(len(cat_outs)):
        target=data[:,out_cont.size(1)+cat].long()
        loss += F.cross_entropy(cat_outs[cat], target, reduction=reduction)

    return loss.double()


def train(epoch, ):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.to(device)

        optimizer.zero_grad()
        out_cont, cat_outs = model(data)
        loss = loss_function(out_cont, cat_outs, data)

        loss.backward()
        train_loss += loss.item()
        #print(train_loss)
        optimizer.step()
        '''if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader.dataset),
                100. * batch_idx / len(train_dataloader),
                loss.item() / len(data)))'''


    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_dataloader.dataset)))


def test(epoch, best_loss ):
    model.eval()
    train_loss = 0
    losses=[]
    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.to(device)

        out_cont, cat_outs = model(data)
        loss = loss_function(out_cont, cat_outs, data, reduction='none')
        losses.extend(loss.cpu().detach().numpy())

    for batch_idx, (data, _) in enumerate(malicious_dataloader):
        data = data.to(device)

        out_cont, cat_outs = model(data)
        print("HERE")
        loss = loss_function(out_cont, cat_outs, data, reduction='none')
        print("HERE")
        print(loss)

        losses.extend(loss.cpu().detach().numpy())

    labels=[0 for i in range(len(train_dataloader.dataset))]+[1 for i in range(len(malicious_dataloader.dataset))]

    print("AUC: {}".format(metrics.roc_auc_score(labels, losses)))
    precision, recall, thresholds = metrics.precision_recall_curve(labels, losses)
    print("AUPR: {}".format(metrics.auc(recall, precision)))

y=torch.Tensor(np.ones(X_train.shape[0]))
X_train=X_train.astype('float64')


x=torch.from_numpy(X_train)
my_dataset = TensorDataset(x, y)
train_dataloader = DataLoader(my_dataset, batch_size=256)  # create your dataloader

y=torch.Tensor(np.ones(mal_np.shape[0]))
mal_np=mal_np.astype('float64')
x=torch.from_numpy(mal_np)
my_dataset = TensorDataset(x, y)
malicious_dataloader = DataLoader(my_dataset, batch_size=256)  # create your dataloader

print('Train dataset length: {}'.format(len(train_dataloader.dataset)))
print('Malicious dataset length: {}'.format(len(malicious_dataloader.dataset)))

best_loss=1e6

model = AE(input_dim, embeddings, cat_out, cont_dim)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range( args.epochs + 1):
    train(epoch, )
    if epoch%5==0 :
        best_loss=test(epoch, best_loss)