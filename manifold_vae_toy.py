# This script trains the mixture of VAEs for different toy examples, namely
# a ring in 2D, the swiss roll in 3D, two disconnected circles in 2D, the sphere in 3D
# and the torus in 3D.

import torch.nn as nn
import torch
from utils.injective_generators import InjectiveDenseGenerator
from utils.left_invertible_vae import Mix_VAE
import utils.sampling
import numpy as np
import os
from utils.plot import plot_all,plot_generated
from utils.general import get_device
from utils.datasets import ToyDataset
import random


###############################################
# Model parameters
###############################################

# Set True for computing the comparisons with one generator.
one_gen=False

# Select model
model_name='ring'
#model_name='swiss'
#model_name='two_circles'
#model_name='sphere'
#model_name='torus'

if model_name=='ring':
    sig_ld=0.01
    sig_hd=.01
    num_generators=2
    dim_hd=2
    dim_ld=2
    noise_level=0.01
elif model_name=='swiss':
    sig_ld=0.01
    sig_hd=.2
    num_generators=4
    dim_hd=3
    dim_ld=2
    noise_level=0.05
elif model_name=='two_circles':
    sig_ld=0.01
    sig_hd=.01
    num_generators=4
    dim_hd=2
    dim_ld=1
    noise_level=0.01
elif model_name=='sphere':
    sig_ld=0.01
    sig_hd=.01
    num_generators=2
    dim_hd=3
    dim_ld=2
    noise_level=0.01
elif model_name=='torus':
    sig_ld=0.01
    sig_hd=.05
    num_generators=6
    dim_hd=3
    dim_ld=2
    noise_level=0.01

if one_gen:
    num_generators=1

###############################################
# Training parameters
###############################################



n_data=1024*10
n_test_data=1024

learning_rate=1e-4
if one_gen:
    learning_rate=5e-5

lip_loss=1.

device = get_device()

deterministic=False
if deterministic:
    np.random.seed(20)
    torch.manual_seed(20)
    random.seed(20)

models_dir='models'
plots_dir='plots'

n_epochs=200
n_epochs_overlap=50
batch_size=128
test_batch_size=1024
train_dataset=ToyDataset(model_name,n_data,noise_level=noise_level)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
test_dataset=ToyDataset(model_name,n_test_data,noise_level=noise_level)
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=test_batch_size,shuffle=True,drop_last=True)
retrain=True
reseed=True
restart=True
only_last=False
use_seeding=True



###############################################
# Generate directories and model objects
###############################################



if not os.path.isdir(models_dir):
    os.mkdir(models_dir)

if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)


decoders=[]

for ng in range(num_generators):
    decoders.append(InjectiveDenseGenerator(dim_hd,dim_ld,sig_hd,sig_ld,utils.sampling.get_spline_latent,latent_nf=dim_ld>1).to(device))

mix_vae=Mix_VAE(decoders)

###############################################
# Training
###############################################

# skip if retrain is False

if retrain:
    if reseed:
        mix_vae.seeding(train_dataloader,init_epochs=2000,seeding_candidates=100)
        plot_all(mix_vae,train_dataset.data,plots_dir,model_name,show=False)
        loss_sum=mix_vae.test_step(test_dataloader)
        best_test_loss=loss_sum
        mix_vae.save_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_seeding.pt')
        print('After seeding, Test loss: {0:.2f}, Best test loss: {1:.2f}, sig_ld: {2:.4f}'.format(loss_sum,best_test_loss,sig_ld))
    else:
        mix_vae.load_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_seeding.pt')

    optimizer=torch.optim.Adam(mix_vae.parameters(), lr = learning_rate)

###############################################
# Training
###############################################

# skip if retrain is False

if retrain:
    if not restart:
        opt_state_dict,cp_epoch=mix_vae.load_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_tmp.pt')
        optimizer.load_state_dict(opt_state_dict)
    if only_last:
        opt_state_dict,_=mix_vae.load_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen.pt')
        optimizer.load_state_dict(opt_state_dict)
    else:
        best_test_loss=1e10
        best_train_loss=1e10
        for epoch in range(1,n_epochs+1):
            if not restart and epoch<=cp_epoch:
                print('Skip epoch',epoch)
                continue
            train_loss=mix_vae.train_epochs_full_grad(train_dataloader,optimizer,gradient_clipping=2.,Lipschitz_loss=lip_loss)
            if train_loss<1.0*best_train_loss:
                mix_vae.save_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_tmp.pt',optimizer,epoch)
                if train_loss<best_train_loss:
                    best_train_loss=train_loss
            if epoch==50:
                # Deactivate Lipschitz regularization after 50 epochs.
                lip_loss=None
            if epoch%10==0:
                plot_all(mix_vae,train_dataset.data,plots_dir,model_name,show=False)
                loss_sum=mix_vae.test_step(test_dataloader)
                if loss_sum<best_test_loss:
                    best_test_loss=loss_sum
                    mix_vae.save_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen.pt',optimizer,epoch)
                print('Epoch {0}, Test loss: {1:.2f}, Best test loss: {2:.2f}, sig_ld: {3:.4f}'.format(epoch,loss_sum,best_test_loss,sig_ld))
    mix_vae.learn_decoder_weights=True
    mix_vae.train_epochs_dl(train_dataloader,optimizer,scale=1.5,epochs=n_epochs_overlap,normalize=False,gradient_clipping=2.)
    mix_vae.save_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_last.pt')


###############################################
# Plot
###############################################

# load model, skip if retrain is True
if not retrain:
    mix_vae.load_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_last.pt')

lds=[]
for decoder in mix_vae.decoders:
    lds.append(torch.exp(decoder.sig_ld_log).detach().cpu().numpy())
print(lds)

plot_all(mix_vae,train_dataset.data,plots_dir,model_name,show=True)


