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

device = torch.device("cuda" if args.cuda else "cpu")
print(device)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



def compute_embedding_size(n_categories):
    val = min(600, round(1.6 * n_categories**0.56))
    return int(val)

dataset='ton_iot'
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
    #X_train, X_test = train_test_split(benign_np, test_size = 0.01, random_state = 42)
    X_train, X_test =benign_np, benign_np
    X_train = X_train.astype('float64')
    cont_dim=len(float_cols)

    print(float_cols)
    print(categorical_cols)
    print(benign_np[:2])
    #sys.exit()

    for col in range(len(categorical_cols)):
        n_cats = preprocess.encoders[categorical_cols[col]]['n_classes']#len(preprocess.encoders[categorical_cols[col]]['encoder'].classes_)
        print(col)
        print(preprocess.encoders[categorical_cols[col]])
        print(preprocess.encoders[categorical_cols[col]].inverse_transform(benign_np[:100,len(float_cols)]))
        sys.exit()
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
    benign_np , preprocess, float_cols, categorical_cols=df_to_np(f'{base_dir}/nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes,train_set=True, return_preprocess=True)
    mal_np=df_to_np(f'{base_dir}/nf_bot_iot/NF-BoT-IoT.csv',  nf_bot_iot.datatypes,train_set=False)
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






class VAE(nn.Module):
    def __init__(self,input_dim, embeddings, cats_out, cont_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)

        #self.fc2 = nn.Linear(128, 64)
        self.enc_mu = torch.nn.Linear(128, 64)
        self.enc_log_sigma = torch.nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, cont_dim)
        self.embeddings=torch.nn.ModuleList(embeddings)
        self.cats_out = []
        for cat in cats_out:
            self.cats_out.append(torch.nn.Linear(128, cat))
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
    loss=F.mse_loss(out_cont.double(), data[:,:out_cont.size(1)].double(), reduction=reduction)
    recon_loss=0
    if reduction=='none':
        recon_loss=torch.sum(loss, dim=1)
    for cat in range(len(cat_outs)):
        target=data[:,out_cont.size(1)+cat].long()
        recon_loss += F.cross_entropy(cat_outs[cat], target, reduction=reduction)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss+KLD, recon_loss.double(), KLD


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

        #print(train_loss)
        optimizer.step()



    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_dataloader.dataset)))
    info = {
        'loss': train_loss/len(train_dataloader.dataset),
        'recon_loss': recon_loss/len(train_dataloader.dataset),
        'kld': kld_loss/len(train_dataloader.dataset)
    }
    print(f"====> Epoch {epoch}: \n{info}")


def test(epoch, best_loss ):
    model.eval()
    train_loss = 0
    losses=[]
    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.to(device)

        out_cont, cat_outs, mu, logvar = model(data)
        loss , recon, kld= loss_function(out_cont, cat_outs, data, mu, logvar, reduction='none')
        losses.extend(loss.cpu().detach().numpy())

    for batch_idx, (data, _) in enumerate(malicious_dataloader):
        data = data.to(device)

        out_cont, cat_outs = model(data)
        loss = loss_function(out_cont, cat_outs, data, mu, logvar, reduction='none')
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

model = VAE(input_dim, embeddings, cat_out, cont_dim)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range( 50):
    train(epoch, )
    #if epoch%5==0 :
        #best_loss=test(epoch, best_loss)