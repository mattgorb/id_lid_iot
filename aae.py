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

    mal_np = df_to_np( f'{base_dir}iot23/iot23_sample_with_real.csv', iot23.datatypes, train_set=False)

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
    benign_np , preprocess, float_cols, categorical_cols=df_to_np(f'{base_dir}nf_bot_iot/NF-BoT-IoT.csv', nf_bot_iot.datatypes,train_set=True, return_preprocess=True)
    mal_np=df_to_np(f'{base_dir}nf_bot_iot/NF-BoT-IoT.csv',  nf_bot_iot.datatypes,train_set=False)
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


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


# Encoder
class Q_net(nn.Module):
    def __init__(self, input_dim, N, z_dim, embeddings):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(input_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)

        self.embeddings = torch.nn.ModuleList(embeddings)


    def forward(self, x):
        num_fts=x.size(1)-len(self.embeddings)
        embed_list=[]
        for embed in range(len(self.embeddings)):

            out=self.embeddings[embed](x[:,num_fts+embed].long())
            embed_list.append(out)
        embedded_cats = torch.cat(embed_list, dim=1)
        inputs=torch.cat([embedded_cats.float(),x[:,:num_fts].float()], dim=1)


        x = F.dropout(self.lin1(inputs), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss


# Decoder
class P_net(nn.Module):
    def __init__(self, N, z_dim, cont_dim,cats_out):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, cont_dim)

        self.cats_out = []
        for cat in cats_out:
            self.cats_out.append(torch.nn.Linear(N, cat))

        self.cats_out = torch.nn.ModuleList(self.cats_out)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        out_cont = self.lin3(x)

        cat_outs=[]#torch.nn.ModuleList()
        for cat in self.cats_out:
            cat_outs.append(cat(x))

        return F.sigmoid(out_cont), cat_outs


# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self, N, z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))






def loss_function(out_cont, cat_outs, data, reduction='sum'):
    loss=F.mse_loss(out_cont.double(), data[:,:out_cont.size(1)].double(), reduction=reduction)
    if reduction=='none':
        loss=torch.sum(loss, dim=1)
    for cat in range(len(cat_outs)):
        target=data[:,out_cont.size(1)+cat].long()
        loss += F.cross_entropy(cat_outs[cat], target, reduction=reduction)

    return loss.double()


def train(epoch, ):
    Q.train()
    recon_loss_total = 0
    disc_loss_total=0
    gen_loss_total=0

    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.to(device)

        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

        z_sample = Q(data)  # encode to z
        out_cont, cat_outs = P(z_sample)  # decode to X reconstruction
        #recon_loss = F.binary_cross_entropy(X_sample + EPS, images + EPS)
        recon_loss = loss_function(out_cont, cat_outs, data)

        recon_loss.backward()
        optim_P.step()
        optim_Q_enc.step()

        Q.eval()
        z_real_gauss = (5*torch.randn(data.size()[0], z_red_dims)).cuda()
        D_real_gauss = D_gauss(z_real_gauss)

        z_fake_gauss = Q(data)
        D_fake_gauss = D_gauss(z_fake_gauss)

        D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))

        D_loss.backward()
        optim_D.step()

        # Generator
        Q.train()
        z_fake_gauss = Q(data)
        D_fake_gauss = D_gauss(z_fake_gauss)

        G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))

        G_loss.backward()
        optim_Q_gen.step()

        recon_loss_total+=recon_loss.item()
        disc_loss_total+=D_loss.item()
        gen_loss_total+=G_loss.item()


    info = {
        'recon_loss': recon_loss_total/len(train_dataloader.dataset),
        'discriminator_loss': disc_loss_total/len(train_dataloader.dataset),
        'generator_loss': gen_loss_total/len(train_dataloader.dataset)
    }
    print(f"====> Epoch: \n{info}")


    #print('====> Epoch: {} Average loss: {:.4f}'.format(
          #epoch, train_loss / len(train_dataloader.dataset)))


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
        loss = loss_function(out_cont, cat_outs, data, reduction='none')
        losses.extend(loss.cpu().detach().numpy())

    labels=[0 for i in range(len(train_dataloader.dataset))]+[1 for i in range(len(malicious_dataloader.dataset))]

    print("AUC: {}".format(metrics.roc_auc_score(labels, losses)))
    precision, recall, thresholds = metrics.precision_recall_curve(labels, losses)
    print("AUPR: {}".format(metrics.auc(recall, precision)))



EPS = 1e-15
z_red_dims = 64
Q = Q_net(input_dim, 128, z_red_dims,embeddings).cuda()
P = P_net( 128, z_red_dims,cont_dim,cat_out).cuda()
D_gauss = D_net_gauss(500, z_red_dims).cuda()


y=torch.Tensor(np.ones(X_train.shape[0]))
X_train=X_train.astype('float64')


x=torch.from_numpy(X_train)
my_dataset = TensorDataset(x, y)
train_dataloader = DataLoader(my_dataset, batch_size=256)  # create your dataloader

'''y=torch.Tensor(np.ones(mal_np.shape[0]))
mal_np=mal_np.astype('float64')
x=torch.from_numpy(mal_np)
my_dataset = TensorDataset(x, y)
malicious_dataloader = DataLoader(my_dataset, batch_size=256)  # create your dataloader

print('Train dataset length: {}'.format(len(train_dataloader.dataset)))
print('Malicious dataset length: {}'.format(len(malicious_dataloader.dataset)))'''

best_loss=1e6

#model = AE(input_dim, embeddings, cat_out, cont_dim)
#model = model.to(device)

# Set learning rates
gen_lr = 0.0001
reg_lr = 0.00005
#gen_lr = 0.0001
#reg_lr = 0.0001

#encode/decode optimizers
optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
#regularizing optimizers
optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
optim_D = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)

for epoch in range( 100):

    train(epoch, )
    #if epoch%5==0 :
        #best_loss=test(epoch, best_loss)