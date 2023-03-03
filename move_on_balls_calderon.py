# This script trains the mixture of VAEs for the balls.

import torch.nn as nn
import torch
from utils.injective_generators import InjectiveMultiscaleGenerator
from utils.left_invertible_vae import Mix_VAE
from utils.sampling import gen_torus
import utils.sampling
from utils.general import *
import numpy as np
import os
from utils.datasets import AddGaussianNoise,BallsDataset,Jittering
import matplotlib.pyplot as plt
import torchvision
from utils.calderon_utils import CalderonReconstruction,create_ground_truth_from_image
import random

###############################################
# Global parameters
###############################################

device = get_device()

deterministic=True
seed=20
if deterministic:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

device = get_device()

base_dir=get_base_dir()

n_runs=20
batch_size=n_runs
test_batch_size=n_runs
n_data=50000

train_dataset=BallsDataset(epoch_size=n_data,start_seed=0,std_intensity=.0,intensity_dist='uniform',transform=Jittering(),eps=6,fix_int=True)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

test_dataset=BallsDataset(epoch_size=10000,start_seed=0,std_intensity=.0,intensity_dist='uniform',transform=Jittering(),test=True,eps=6,fix_int=True)    
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

ground_truth_images=iter(test_dataloader).next()

###############################################
# Dataset and VAE parameters
###############################################

sig_ld=.01
sig_hd=.1
num_generators=2
dim_ld=6

models_dir=base_dir+'/models'
model_name='balls'
plots_dir=base_dir+'/plots'
plots_dir_full=plots_dir+'/'+model_name+'_'+str(num_generators)+'_gen_calderon'


###############################################
# Load VAE model
###############################################

if not os.path.isdir(models_dir):
    os.mkdir(models_dir)

if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)

if not os.path.isdir(plots_dir_full):
    os.mkdir(plots_dir_full)

im=torchvision.utils.make_grid(torch.tensor(ground_truth_images[0],dtype=torch.float,device=device),nrow=20).detach().cpu().numpy()
im=np.transpose(im,(1,2,0))
im=np.minimum(np.maximum(im,0.),1.)
plt.imsave(plots_dir_full+'/balls_ds_grid.png',im)

decoders=[]

for ng in range(num_generators):
    decoders.append(InjectiveMultiscaleGenerator(dim_ld,sig_hd,sig_ld,utils.sampling.get_spline_latent,latent_nf=True,logit_transform=None).to(device))

mix_vae=Mix_VAE(decoders)

mix_vae.load_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_last.pt')

for ng in range(len(mix_vae.decoders)):
    mix_vae.decoders[ng].set_sig_hd(1.)

reconstructions=[]

for run in range(n_runs):
    ###############################################
    # create forward operator
    ###############################################

    # parameters
    ground_truth_tensor=ground_truth_images[0][run:run+1].to(device)
    ground_truth=ground_truth_tensor.detach().cpu().numpy()
    print(mix_vae.classify(ground_truth_tensor).detach().cpu().numpy())
    local0=mix_vae.encoders[0](ground_truth_tensor)
    project0=mix_vae.decoders[0](local0)
    CR=CalderonReconstruction(create_ground_truth_from_image(ground_truth),N=15,noise_level=0.001)

    # objective functional
    F=CR.data_fidelity
    F_der=CR.data_fidelity_derivative
    im_gt=np.tile(np.minimum(np.maximum(CR.ground_truth,0.),1.)[:,:,None],(1,1,3))
    plt.imsave(plots_dir_full+'/move_on_balls_gt_'+str(run)+'.png',im_gt)
    im_gt_p0=np.tile(np.minimum(np.maximum(project0.squeeze().detach().cpu().numpy(),0.),1.)[:,:,None],(1,1,3))
    plt.imsave(plots_dir_full+'/move_on_balls_gt_p0'+str(run)+'.png',im_gt_p0)

    ###############################################
    # generate initialization
    ###############################################

    # generate candidates
    n_candidates=1
    candidates=mix_vae.sample(n_candidates)

    # eval objective
    best_obj=0
    candidate_index=-1
    for ind in range(n_candidates):
        objective_value=F(candidates[ind].detach().cpu().numpy())
        print(objective_value)
        if candidate_index==-1 or objective_value<best_obj:
            candidate_index=ind
            best_obj=objective_value
    print(candidate_index,best_obj)
    init=candidates[candidate_index]
    print(mix_vae.encoders[0](init[None,:,:,:]).detach().cpu().numpy())

    # plot initialization
    im_in=init.squeeze().detach().cpu().numpy()
    im_in=np.tile(np.minimum(np.maximum(im_in,0.),1.)[:,:,None],(1,1,3))
    plt.imsave(plots_dir_full+'/move_on_balls_init_'+str(run)+'.png',im_in)
    
    ###############################################
    # gradient descent
    ###############################################

    step_size=1e+4
    steps=100
    print(init.shape)
    trajectory,speed,objectives=mix_vae.create_trajectory_adaptive(init[None,:,:,:],F,step_size,steps,f_der=F_der,use_prior_lambda=None,print_steps=1,retraction='chart_retraction',min_step_size=step_size/100)

    best_val=np.argmin(np.array(objectives))    

    result=trajectory[best_val]
    reconstructions.append(torch.tensor(result,dtype=torch.float,device=device))
    im_res=result.squeeze()
    im_res=np.tile(np.minimum(np.maximum(im_res,0.),1.)[:,:,None],(1,1,3))
    plt.imsave(plots_dir_full+'/move_on_balls_result_'+str(run)+'.png',im_res)

    im_all=np.concatenate([im_gt,np.zeros((im_gt.shape[0],2,3)),im_res,np.zeros((im_gt.shape[0],2,3)),im_in],1)
    plt.imsave(plots_dir_full+'/move_on_balls_all_'+str(run)+'.png',im_all)

    import imageio
    images = []
    for im in trajectory:
        im=im.squeeze()
        im=np.tile(np.minimum(np.maximum(im,0.),1.)[:,:,None],(1,1,3))*255
        im=im.astype(np.uint8)
        images.append(im)
    imageio.mimsave(plots_dir_full+'/move_on_balls_gif_'+str(run)+'.gif', images)
    
reconstructions=torch.stack(reconstructions)[:,None,:,:]
im=torchvision.utils.make_grid(torch.tensor(reconstructions,dtype=torch.float,device=device),nrow=20).detach().cpu().numpy()
im=np.transpose(im,(1,2,0))
im=np.minimum(np.maximum(im,0.),1.)
plt.imsave(plots_dir_full+'/balls_res_grid.png',im)

