# This script trains the mixture of VAEs for the balls manifold which is 
# used for the EIT example in the paper.

import torch.nn as nn
import torch
from utils.injective_generators import InjectiveMultiscaleGenerator
from utils.left_invertible_vae import Mix_VAE
import utils.sampling
import numpy as np
import os
from utils.general import get_device,get_base_dir
from utils.datasets import BallsDataset,AddGaussianNoise,Jittering
import random
import torchvision.transforms as transforms
from utils.plot import generate_images,generate_image_charts


###############################################
# Model parameters
###############################################

sig_ld=0.01
sig_hd=.1
num_generators=2
dim_ld=6
noise_level=0.0
overlap_scale=1.5
model_name='balls'

###############################################
# Training parameters
###############################################

learning_rate=5e-5

device = get_device()

base_dir=get_base_dir()

deterministic=True
if deterministic:
    np.random.seed(20)
    torch.manual_seed(20)
    random.seed(20)

models_dir=base_dir+'/models'
plots_dir=base_dir+'/plots'

n_data=50000

n_epochs=200
n_epochs_overlap=50
batch_size=128
test_batch_size=128
lip_loss=0.05

train_dataset=BallsDataset(epoch_size=n_data,start_seed=0,std_intensity=.0,intensity_dist='uniform',transform=Jittering(),eps=6,fix_int=True)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

test_dataset=BallsDataset(epoch_size=10000,start_seed=0,std_intensity=.0,intensity_dist='uniform',transform=Jittering(),test=True,eps=6,fix_int=True)    
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

retrain=False
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
plots_dir=plots_dir+'/'+str(model_name)+str(num_generators)+'ng'
if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)

decoders=[]

for ng in range(num_generators):
    decoders.append(InjectiveMultiscaleGenerator(dim_ld,sig_hd,sig_ld,utils.sampling.get_spline_latent,latent_nf=dim_ld>1).to(device))

mix_vae=Mix_VAE(decoders)

###############################################
# Initialization
###############################################

#skip when retrain is False

if retrain and not only_last and use_seeding:
    if reseed:
        if num_generators>1:
            # seeding does not make sense for one generator.
            mix_vae.seeding(train_dataloader,num_samples=400,init_epochs=1000,seeding_candidates=batch_size,batch_size=200,learning_rate=1e-4)
        # compute test loss and save initialization
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

# skip when retrain is False

if retrain:
    if not restart:
        # Load model and optimizer parameters
        opt_state_dict,cp_epoch=mix_vae.load_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_tmp.pt')
        optimizer.load_state_dict(opt_state_dict)
    if only_last:
        # Load model and optimizer parameters
        opt_state_dict,_=mix_vae.load_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen.pt')
        optimizer.load_state_dict(opt_state_dict)
    else:
        best_test_loss=1e10
        best_train_loss=1e10
        for epoch in range(1,n_epochs+1):
            if not restart and epoch<=cp_epoch:
                # Skip epochs, when continuing training
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
                # Compute test loss end save checkpoint
                loss_sum=mix_vae.test_step(test_dataloader)
                if loss_sum<best_test_loss:
                    best_test_loss=loss_sum
                    mix_vae.save_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen.pt',optimizer,epoch)
                print('Epoch {0}, Test loss: {1:.2f}, Best test loss: {2:.2f}, sig_ld: {3:.4f}'.format(epoch,loss_sum,best_test_loss,sig_ld))

    # Overlapping procedure
    mix_vae.learn_decoder_weights=True
    mix_vae.train_epochs_dl(train_dataloader,optimizer,scale=overlap_scale,epochs=n_epochs_overlap,normalize=False,gradient_clipping=2.)
    mix_vae.save_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_last.pt')
    print('Training completed!')

###############################################
# Generate resulting images
###############################################

# load model, skip when retrain is True
if not retrain:
    if os.path.isfile(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_last.pt'):
        mix_vae.load_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_last.pt')
    else:
        mix_vae.load_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_tmp.pt')

mix_vae.eval()

generate_images(mix_vae,plots_dir,model_name,n_samples=10)

for ng in range(num_generators):
    generate_images(mix_vae,plots_dir,model_name,n_samples=10,generator=ng)
    generate_image_charts(mix_vae,plots_dir,model_name,ng,10)


