#!/usr/bin/env python
#
# file: $ISIP_EXP/so-scope/scripts/model.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#
# This script hold the model architecture
#------------------------------------------------------------------------------

# import pytorch modules
#
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from convlstm import ConvLSTMCell

# import modules
#
import os
import random

# for reproducibility, we seed the rng
#
SEED1 = 1337
NEW_LINE = "\n"
IMG_SIZE = 64 #400
Z_SIZE = 256*2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------------------------------
#
# helper functions are listed here
#
#-----------------------------------------------------------------------------

# function: set_seed
#
# arguments: seed - the seed for all the rng
#
# returns: none
#
# this method seeds all the random number generators and makes
# the results deterministic
#
def set_seed(seed):
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
#
# end of method


# function: get_data
#
# arguments: fp - file pointer
#            num_feats - the number of features in a sample
#
# returns: data - the signals/features
#          labels - the correct labels for them
#
# this method takes in a fp and returns the data and labels
POINTS = 1080
IMG_SIZE = 64
SEQ_LEN = 10
class VaeTestDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, file_name):
        # initialize the data and labels
        # read the names of image data:
        self.scan_file_names = []
        self.pos_file_names = []
        self.vel_file_names = []
        # open train.txt or dev.txt:
        fp_scan = open(img_path+'/scans/'+file_name+'.txt', 'r')
        fp_pos = open(img_path+'/positions/'+file_name+'.txt', 'r')
        fp_vel = open(img_path+'/velocities/'+file_name+'.txt', 'r')
        # for each line of the file:
        for line in fp_scan.read().split(NEW_LINE):
            if('.npy' in line): 
                self.scan_file_names.append(img_path+'/scans/'+line)
        for line in fp_pos.read().split(NEW_LINE):
            if('.npy' in line): 
                self.pos_file_names.append(img_path+'/positions/'+line)
        for line in fp_vel.read().split(NEW_LINE):
            if('.npy' in line): 
                self.vel_file_names.append(img_path+'/velocities/'+line)
        # close txt file:
        fp_scan.close()
        fp_pos.close()
        fp_vel.close()
        self.length = len(self.scan_file_names)

        print("dataset length: ", self.length)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # get the index of start point:
        scans = np.zeros((SEQ_LEN+SEQ_LEN, POINTS))
        positions = np.zeros((SEQ_LEN+SEQ_LEN, 3))
        vels = np.zeros((SEQ_LEN+SEQ_LEN, 2))
        # get the index of start point:
        if(idx+(SEQ_LEN+SEQ_LEN) < self.length): # train1:
            idx_s = idx
        else:
            idx_s = idx - (SEQ_LEN+SEQ_LEN)

        for i in range(SEQ_LEN+SEQ_LEN):
            # get the scan data:
            scan_name = self.scan_file_names[idx_s+i]
            scan = np.load(scan_name)
            scans[i] = scan
            # get the scan_ur data:
            pos_name = self.pos_file_names[idx_s+i]
            pos = np.load(pos_name)
            positions[i] = pos
            # get the velocity data:
            vel_name = self.vel_file_names[idx_s+i]
            vel = np.load(vel_name)
            vels[i] = vel
        
        # initialize:
        scans[np.isnan(scans)] = 20.
        scans[np.isinf(scans)] = 20.
        scans[scans==30] = 20.

        positions[np.isnan(positions)] = 0.
        positions[np.isinf(positions)] = 0.

        vels[np.isnan(vels)] = 0.
        vels[np.isinf(vels)] = 0.

        # transfer to pytorch tensor:
        scan_tensor = torch.FloatTensor(scans)
        pose_tensor = torch.FloatTensor(positions)
        vel_tensor =  torch.FloatTensor(vels)

        data = {
                'scan': scan_tensor,
                'position': pose_tensor,
                'velocity': vel_tensor, 
                }

        return data

#
# end of function


#------------------------------------------------------------------------------
#
# the model is defined here
#
#------------------------------------------------------------------------------

# define the PyTorch VAE model
#
# define a VAE
# Residual blocks: 
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

# Encoder & Decoder Architecture:
# Encoder:
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=in_channels,
                                                  out_channels=num_hiddens//2,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens//2),
                                        nn.ReLU(True)
                                    ])
        self._conv_2 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=num_hiddens//2,
                                                  out_channels=num_hiddens,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens)
                                        #nn.ReLU(True)
                                    ])
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._conv_2(x)
        x = self._residual_stack(x)
        return x

# Decoder:
class Decoder(nn.Module):
    def __init__(self, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_2 = nn.Sequential(*[
                                            nn.ReLU(True),
                                            nn.ConvTranspose2d(in_channels=num_hiddens,
                                                              out_channels=num_hiddens//2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1),
                                            nn.BatchNorm2d(num_hiddens//2),
                                            nn.ReLU(True)
                                        ])

        self._conv_trans_1 = nn.Sequential(*[
                                            nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                              out_channels=num_hiddens//2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1),
                                            nn.BatchNorm2d(num_hiddens//2),
                                            nn.ReLU(True),                  
                                            nn.Conv2d(in_channels=num_hiddens//2,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1),
                                            nn.Sigmoid()
                                        ])

    def forward(self, inputs):
        x = self._residual_stack(inputs)
        x = self._conv_trans_2(x)
        x = self._conv_trans_1(x)
        return x

class VAE_Encoder(nn.Module):
    def __init__(self, input_channel):
        super(VAE_Encoder, self).__init__()
        # parameters:
        self.input_channels = input_channel
        # Constants
        num_hiddens = 128 #128
        num_residual_hiddens = 64 #32
        num_residual_layers = 2
        embedding_dim = 2 #64

        # encoder:
        in_channels = input_channel
        self._encoder = Encoder(in_channels, 
                                num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)

        # z latent variable: 
        self._encoder_z_mu = nn.Conv2d(in_channels=num_hiddens, 
                                    out_channels=embedding_dim,
                                    kernel_size=1, 
                                    stride=1)
        self._encoder_z_log_sd = nn.Conv2d(in_channels=num_hiddens, 
                                    out_channels=embedding_dim,
                                    kernel_size=1, 
                                    stride=1)  
        
    def forward(self, x):
        # input reshape:
        x = x.reshape(-1, self.input_channels, IMG_SIZE, IMG_SIZE)
        # Encoder:
        encoder_out = self._encoder(x)
        # get `mu` and `log_var`:
        z_mu = self._encoder_z_mu(encoder_out)
        z_log_sd = self._encoder_z_log_sd(encoder_out)
        return z_mu, z_log_sd
		
# our proposed model:
class scope_plus_plus(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(scope_plus_plus, self).__init__()
        # parameters:
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_channels = output_channels

        # Constants
        num_hiddens = 128 
        num_residual_hiddens = 64 
        num_residual_layers = 2
        embedding_dim = 2 
    
        # prediction encoder:
        self._convlstm = ConvLSTMCell(input_dim=self.input_channels,
                                    hidden_dim=num_hiddens//4,
                                    kernel_size=(3, 3),
                                    bias=True)
        self._encoder = VAE_Encoder(num_hiddens//4 + self.input_channels,)

        # decoder:
        self._decoder_z_mu = nn.ConvTranspose2d(in_channels=embedding_dim, 
                                    out_channels=num_hiddens,
                                    kernel_size=1, 
                                    stride=1)
        self._decoder = Decoder(self.output_channels,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)
        

    def vae_reparameterize(self, z_mu, z_log_sd):
        """
        :param mu: mean from the encoder's latent space
        :param log_sd: log standard deviation from the encoder's latent space
        :output: reparameterized latent variable z, Monte carlo KL divergence
        """
        # reshape:
        z_mu = z_mu.reshape(-1, self.latent_dim, 1)
        z_log_sd = z_log_sd.reshape(-1, self.latent_dim, 1)
        # define the z probabilities (in this case Normal for both)
        # p(z): N(z|0,I)
        pz = torch.distributions.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_log_sd))
        # q(z|x,phi): N(z|mu, z_var)
        qz_x = torch.distributions.Normal(loc=z_mu, scale=torch.exp(z_log_sd))

        # repameterization trick: z = z_mu + xi (*) z_log_var, xi~N(xi|0,I)
        z = qz_x.rsample()
        # Monte Carlo KL divergence: MCKL(p(z)||q(z|x,phi)) = log(p(z)) - log(q(z|x,phi))
        # sum over weight dim, leaves the batch dim 
        kl_divergence = (pz.log_prob(z) - qz_x.log_prob(z)).sum(dim=1)
        kl_loss = -kl_divergence.mean()

        return z, kl_loss 

    def forward(self, x, x_map):
        """
        Forward pass `input_img` through the network
        """
        # reconstruction: 
        # encode:
        # input reshape:
        x = x.reshape(-1, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE)
        x_map = x_map.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
        # find size of different input dimensions
        b, seq_len, c, h, w = x.size()
        
        # encode: 
        # initialize hidden states
        h_enc, enc_state = self._convlstm.init_hidden(batch_size=b, image_size=(h, w))
        for t in range(seq_len): 
            x_in = x[:,t]
            h_enc, enc_state = self._convlstm(input_tensor=x_in,
                                              cur_state=[h_enc, enc_state])

        enc_in = torch.cat([h_enc, x_map], dim=1)  # concatenate along channel axis
        z_mu, z_log_sd = self._encoder(enc_in)

        # get the latent vector through reparameterization:
        z, kl_loss = self.vae_reparameterize(z_mu, z_log_sd)
    
        # decode:
        # reshape:
        z = z.reshape(-1, 2, 16, 16)
        x_d = self._decoder_z_mu(z)
        prediction = self._decoder(x_d)

        return prediction, kl_loss

# our proposed model:
class scope(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(scope, self).__init__()
        # parameters:
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_channels = output_channels

        # Constants
        num_hiddens = 128 
        num_residual_hiddens = 64 
        num_residual_layers = 2
        embedding_dim = 2 
    
        # prediction encoder:
        self._convlstm = ConvLSTMCell(input_dim=self.input_channels,
                                    hidden_dim=num_hiddens//4,
                                    kernel_size=(3, 3),
                                    bias=True)
        self._encoder = VAE_Encoder(num_hiddens//4)

        # decoder:
        self._decoder_z_mu = nn.ConvTranspose2d(in_channels=embedding_dim, 
                                    out_channels=num_hiddens,
                                    kernel_size=1, 
                                    stride=1)
        self._decoder = Decoder(self.output_channels,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)
        

    def vae_reparameterize(self, z_mu, z_log_sd):
        """
        :param mu: mean from the encoder's latent space
        :param log_sd: log standard deviation from the encoder's latent space
        :output: reparameterized latent variable z, Monte carlo KL divergence
        """
        # reshape:
        z_mu = z_mu.reshape(-1, self.latent_dim, 1)
        z_log_sd = z_log_sd.reshape(-1, self.latent_dim, 1)
        # define the z probabilities (in this case Normal for both)
        # p(z): N(z|0,I)
        pz = torch.distributions.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_log_sd))
        # q(z|x,phi): N(z|mu, z_var)
        qz_x = torch.distributions.Normal(loc=z_mu, scale=torch.exp(z_log_sd))

        # repameterization trick: z = z_mu + xi (*) z_log_var, xi~N(xi|0,I)
        z = qz_x.rsample()
        # Monte Carlo KL divergence: MCKL(p(z)||q(z|x,phi)) = log(p(z)) - log(q(z|x,phi))
        # sum over weight dim, leaves the batch dim 
        kl_divergence = (pz.log_prob(z) - qz_x.log_prob(z)).sum(dim=1)
        kl_loss = -kl_divergence.mean()

        return z, kl_loss 

    def forward(self, x):
        """
        Forward pass `input_img` through the network
        """
        # reconstruction: 
        # encode:
        # input reshape:
        x = x.reshape(-1, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE)
        # find size of different input dimensions
        b, seq_len, c, h, w = x.size()
        
        # encode: 
        # initialize hidden states
        h_enc, enc_state = self._convlstm.init_hidden(batch_size=b, image_size=(h, w))
        for t in range(seq_len): 
            x_in = x[:,t]
            h_enc, enc_state = self._convlstm(input_tensor=x_in,
                                              cur_state=[h_enc, enc_state])

        enc_in = h_enc
        z_mu, z_log_sd = self._encoder(enc_in)

        # get the latent vector through reparameterization:
        z, kl_loss = self.vae_reparameterize(z_mu, z_log_sd)
    
        # decode:
        # reshape:
        z = z.reshape(-1, 2, 16, 16)
        x_d = self._decoder_z_mu(z)
        prediction = self._decoder(x_d)

        return prediction, kl_loss


# knowledge distillation model:
class so_scope(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(so_scope, self).__init__()
        # parameters:
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_channels = output_channels

        # Constants
        num_hiddens = 128 
        num_residual_hiddens = 64 
        num_residual_layers = 2
        embedding_dim = 2 
    
        # prediction encoder:
        self._convlstm = ConvLSTMCell(input_dim=self.input_channels,
                                    hidden_dim=num_hiddens//4,
                                    kernel_size=(3, 3),
                                    bias=True)

        self._pred_conv = nn.Sequential(nn.Conv2d(in_channels=num_hiddens//4, 
                                        out_channels=1,
                                        kernel_size=1, 
                                        stride=1), 
                                        nn.Sigmoid()
                                        )

        self._encoder = VAE_Encoder(num_hiddens//4)

        
    def forward(self, x):
        """
        Forward pass `input_img` through the network
        """
        # reconstruction: 
        # encode:
        # input reshape:
        x = x.reshape(-1, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE)
        # find size of different input dimensions
        b, seq_len, c, h, w = x.size()
        
        # encode: 
        # initialize hidden states
        h_enc, enc_state = self._convlstm.init_hidden(batch_size=b, image_size=(h, w))
        for t in range(seq_len): 
            x_in = x[:,t]
            h_enc, enc_state = self._convlstm(input_tensor=x_in,
                                              cur_state=[h_enc, enc_state])

        enc_in = h_enc
        # CNN for prediction:
        rnn_prediction = self._pred_conv(enc_in)

        return rnn_prediction
        
#
# end of class

#
# end of file
