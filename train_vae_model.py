#!/usr/bin/python
import sys

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader


seq_data = pd.read_csv("tumor_data/data.csv")
seq_data.rename(columns={'Unnamed: 0':'sample'}, inplace=True)


labels = pd.read_csv("tumor_data/labels.csv")
labels.rename(columns={'Unnamed: 0':'sample'}, inplace=True)


tumor_data = seq_data.set_index('sample').join(labels.set_index('sample')).to_numpy()


class TumorDataset(Dataset):
    
    def __init__(self, data, transform=None):
        'Initialization'
        self.labels = data[:, -1]
        self.lookup_table, self.labels = np.unique(data[:, -1], return_inverse=True)
        self.genes = data[:, :-1]
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = torch.from_numpy(self.genes[index, : ].astype(np.float)).float()
        y = torch.from_numpy(np.unique(self.labels[index]).astype(int))

        return X, y
		
		
class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))
		
		
class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))
		
		
class Classifier(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Classifier, self).__init__()
        
        self.linears = torch.nn.ModuleList([torch.nn.Linear(D_in, H[0])])
        
        for i in range(1,len(H)):
            self.linears.append(torch.nn.Linear(H[i-1], H[i]))
            
        self.linears.append(torch.nn.Linear(H[-1], D_out))
  
    def forward(self, x):
        for h in self.linears:
            x = h(x)
       
        return F.softmax(x, dim=1)
		
		
class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder, classifier, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self._enc_mu = torch.nn.Linear(self.encoder.linear2.out_features, latent_dim)
        self._enc_log_sigma = torch.nn.Linear(self.encoder.linear2.out_features, latent_dim)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  
    
    
    def forwardEncoder(self, state):
        h_enc = self.encoder(state)
        return self._sample_latent(h_enc)
    
    def forwardAutoEncoder(self, state):
        z = self.forwardEncoder(state)
        return self.decoder(z)
    
    def forwardClassifier(self, state):
        z = self.forwardEncoder(state)
        return self.classifier(z)
    
    def saveModel(self, fileName="vaeModel"):
        torch.save(self.state_dict(), fileName)
        print("Model Saved Successfully")
        
    def loadModel(self, fileName="vaeModel"):
        self.load_state_dict(torch.load(fileName))
        self.eval()
		
		
def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
	
	
def trainClassifier(modelName, loadModel=False, saveModel=True, input_dim = 20531, latent_dim = 10, encoding_layer = 100, classifier_layers = [20, 20, 20], n_classes = 5, batch_size = 32, epochs = 10, verbose = True):
    
    encoder = Encoder(input_dim, encoding_layer, encoding_layer)
    decoder = Decoder(latent_dim, encoding_layer, input_dim)
    classifier = Classifier(latent_dim, classifier_layers, n_classes)

    vae = VAE(encoder, decoder, classifier, latent_dim)
    
    if loadModel:
        vae.loadModel(modelName)

    train_loader = torch.utils.data.DataLoader( TumorDataset(tumor_data), batch_size=batch_size, shuffle=True, drop_last=True)

    if verbose:
        print('Number of samples: ', len(tumor_data))

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)

    l = None
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, classes = data
            inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes).flatten().long()
            predictions = vae.forwardClassifier(inputs.float())
            optimizer.zero_grad()
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(predictions, classes) + ll
            loss.backward()
            optimizer.step()
            l = loss.item()
        if verbose:
            print(epoch, l)
        
    if saveModel:
        vae.saveModel(modelName + "_loss-" + str(l) + "_CIvec-" + str(latent_dim) + "_class-" + str(classifier_layers).replace("[", "").replace("]", "").replace(", ", "."))
        
    return vae
	
classifiers = [[10]*2, [10]*3, [10]*4, [10]*5, 
               [20]*2, [20]*3, [20]*4, [20]*5, 
               [30]*2, [30]*3, [30]*4, [30]*5,
               [40]*2, [40]*3, [40]*4, [40]*5,
               [50]*2, [50]*3, [50]*4, [50]*5]

for c in classifiers:
	trainClassifier("vaeModel1", epochs=300, latent_dim=int(sys.argv[1]), classifier_layers=c)
	print("Model Complete")			   