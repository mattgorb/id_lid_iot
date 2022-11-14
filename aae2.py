import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from data_setup import df_to_np, calculate_weights

## This code coppied from autoencoders.py:

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

directory='/s/luffy/b/nobackup/mgorb/iot/'
directory='csv/'

if dataset=='ton_iot':
    from data_preprocess.drop_columns import ton_iot
    benign_np, preprocess, float_cols, categorical_cols=df_to_np('/s/luffy/b/nobackup/mgorb/iot/ton_iot/Train_Test_Network.csv',ton_iot.datatypes, train_set=True, return_preprocess=True)
    mal_np=df_to_np('/s/luffy/b/nobackup/mgorb/iot/ton_iot/Train_Test_Network.csv', ton_iot.datatypes,train_set=False, return_preprocess=False)
    #X_train, X_test = train_test_split(benign_np, test_size = 0.01, random_state = 42)
    X_train, X_test =benign_np, benign_np
    X_train = X_train.astype('float64')
    print(X_train.shape)
    cont_dim=len(float_cols)
    for col in range(len(categorical_cols)):
        n_cats = preprocess.encoders[categorical_cols[col]]['n_classes']#len(preprocess.encoders[categorical_cols[col]]['encoder'].classes_)

        embed_dim = compute_embedding_size(n_cats)
        print(n_cats, embed_dim)
        embed_layer = torch.nn.Embedding(n_cats, embed_dim).to(device)
        embeddings.append(embed_layer)
        input_dim += embed_dim
        cat_out.append(n_cats)

    input_dim+=cont_dim

print("HEEERE")
print(input_dim)
print(X_train.shape)
#sys.exit()
y=torch.Tensor(np.ones(X_train.shape[0]))
X_train=X_train.astype('float64')


x=torch.from_numpy(X_train)
my_dataset = TensorDataset(x, y)
train_dataloader = DataLoader(my_dataset, batch_size=16)  # create your dataloader

y=torch.Tensor(np.ones(mal_np.shape[0]))
mal_np=mal_np.astype('float64')
x=torch.from_numpy(mal_np)
my_dataset = TensorDataset(x, y)
malicious_dataloader = DataLoader(my_dataset, batch_size=16)  # create your dataloader

print('Train dataset length: {}'.format(len(train_dataloader.dataset)))
print('Malicious dataset length: {}'.format(len(malicious_dataloader.dataset)))

## Assign variables for code from online:
# Source: https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/

X_dim = X_train.shape[0] #?
print("X.shape:", X_train.shape)
# exit()
N = 10 #?
z_dim = 5 #? #encoded vector
# F = 0 #? torch.nn.functional
# X = X_train #?
X = torch.Tensor(X_train.T)
# if torch.cuda.is_available():
if args.cuda:
    X = X.cuda()
TINY = 1e-05 #?
train_batch_size = 42 #?
# Variable = 0 #? torch.autograd variable
# D_loss = 0 #?

#Encoder
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)
    def forward(self, x):
        # x = F.droppout(self.lin1(x), p=0.25, training=self.training)
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        # x = F.droppout(self.lin2(x), p=0.25, training=self.training)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss

# Decoder
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)
        # return F.sigmoid(x)
        return torch.sigmoid(x)


# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        # return F.sigmoid(self.lin3(x))
        return torch.sigmoid(self.lin3(x))


torch.manual_seed(10)
# Q, P = Q_net(), P_net(0)     # Encoder/Decoder
Q, P = Q_net(), P_net()     # Encoder/Decoder
D_gauss = D_net_gauss()                # Discriminator adversarial
# if torch.cuda.is_available():
if args.cuda:
    Q = Q.cuda()
    P = P.cuda()
    D_cat = D_gauss.cuda()
    D_gauss = D_net_gauss().cuda()

# Set learning rates
gen_lr, reg_lr = 0.0006, 0.0008
# Set optimizators
P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)
Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)

#train:
# def Train():
z_sample = Q(X)
X_sample = P(z_sample)

# print(torch.min(X))
# recon_loss = F.binary_cross_entropy(X_sample + TINY, 
#                                     # X.resize(train_batch_size, X_dim) + TINY)
#                                     X.reshape(train_batch_size, X_dim) + TINY)
recon_loss = F.binary_cross_entropy(X_sample, 
                                    X.reshape(train_batch_size, X_dim))
recon_loss.backward()
P_decoder.step()
Q_encoder.step()


Q.eval()    
# z_real_gauss = Variable(torch.randn(train_batch_size, z_dim) * 5)   # Sample from N(0,5)
z_real_gauss = (torch.randn(train_batch_size, z_dim) * 5).numpy()   # Sample from N(0,5)
# if torch.cuda.is_available():
if args.cuda:
    z_real_gauss = z_real_gauss.cuda()
z_fake_gauss = Q(X)


# Compute discriminator outputs and loss
print(type(z_real_gauss), type(z_fake_gauss))
z_real_gauss = torch.from_numpy(z_real_gauss)
# torch.from_numpy(z_fake_gauss)
D_real_gauss, D_fake_gauss = D_gauss(z_real_gauss), D_gauss(z_fake_gauss)
D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))
# D_loss.backward()       # Backpropagate loss
D_loss_gauss.backward()       # Backpropagate loss
D_gauss_solver.step()   # Apply optimization step


# Generator
Q.train()   # Back to use dropout
z_fake_gauss = Q(X)
D_fake_gauss = D_gauss(z_fake_gauss)

G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))
G_loss.backward()
Q_generator.step()
# return D_loss_gauss, G_loss

# for epoch in range(500):
#     Train()

print("Z FAKE GAUSS")
print(z_fake_gauss)
print("D FAKE GAUSS")
print(D_fake_gauss)

