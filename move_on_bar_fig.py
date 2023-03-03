# This script the reconstruction figure from the deblurring example in the paper.

import torch.nn as nn
import torch
from utils.injective_generators import InjectiveMultiscaleGenerator
from utils.left_invertible_vae import Mix_VAE
from utils.sampling import gen_torus
from utils.generate_balls import generate_bar
import utils.sampling
from utils.plot import plot_all
from utils.general import *
import numpy as np
import os
import torchvision.datasets as datasets
from utils.datasets import BarDataset,AddGaussianNoise,Jittering
import matplotlib.pyplot as plt
import torchvision
import random
import math

device = get_device()
base_dir=get_base_dir()

deterministic=True
seed=10
if deterministic:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

n_runs=20
batch_size=n_runs
test_batch_size=n_runs
n_data=50000

train_dataset=BarDataset(epoch_size=n_data,start_seed=0,transform=Jittering(),centered=True)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

test_dataset=BarDataset(epoch_size=10000,start_seed=0,transform=Jittering(),centered=True)    
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

ground_truths=iter(test_dataloader).next()[0].to(device)

###############################################
# Dataset and VAE parameters
###############################################


sig_ld=.01
sig_hd=.1
num_generators=2
dim_ld=1
n_runs=20

models_dir=base_dir+'/models'
model_name='bar'
plots_dir=base_dir+'/plots'
plots_dir_full=plots_dir+'/'+model_name+'_'+str(num_generators)+'_gen'

batch_size=n_runs
test_batch_size=n_runs
n_data=50000

###############################################
# Load VAE model
###############################################

if not os.path.isdir(models_dir):
    os.mkdir(models_dir)

if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)

if not os.path.isdir(plots_dir_full):
    os.mkdir(plots_dir_full)

decoders=[]

for ng in range(num_generators):
    decoders.append(InjectiveMultiscaleGenerator(dim_ld,sig_hd,sig_ld,lambda: utils.sampling.get_spline_latent(lambda_bd=10000),latent_nf=dim_ld>1).to(device))

mix_vae=Mix_VAE(decoders)

mix_vae.load_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_last.pt')

###############################################
# create forward operator
###############################################

# parameters
kernel_size=40
width=15.

# build convolution kernel
x_cord = torch.arange(kernel_size,device=device)
x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
y_grid = x_grid.t()
xy_grid = torch.stack([x_grid, y_grid], dim=-1)
mean = (kernel_size - 1)/2.
variance = width**2.
gaussian_kernel = torch.exp(-.5*torch.sum((xy_grid - mean)**2./variance, dim=-1))
gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

# generate torch model and set convolution kernel
Convolution_Operator=torch.nn.Conv2d(1,1,kernel_size,padding='same',device=device,bias=False)
Convolution_Operator.weight.data=gaussian_kernel.view(1,1,kernel_size,kernel_size)
forward_operator=lambda x: Convolution_Operator(x-.5)+.5

#for ng in range(len(mix_vae.decoders)):
#    mix_vae.decoders[ng].set_sig_hd(1.)

noise_level=.1
F=lambda x: torch.sum(torch.reshape((forward_operator(x)-observation[0])**2,[x.shape[0],-1]),dim=-1)

im=torchvision.utils.make_grid(torch.tensor(ground_truths,dtype=torch.float,device=device),nrow=20).detach().cpu().numpy()
im=np.transpose(im,(1,2,0))
im=np.minimum(np.maximum(im,0.),1.)
plt.imsave('bar_ds_grid.png',im)

observations=forward_operator(ground_truths)+noise_level*torch.randn_like(ground_truths)
im=torchvision.utils.make_grid(torch.tensor(observations,dtype=torch.float,device=device),nrow=20).detach().cpu().numpy()
im=np.transpose(im,(1,2,0))
im=np.minimum(np.maximum(im,0.),1.)
plt.imsave('bar_obs_grid.png',im)

results=[]

for run in range(n_runs):
    ###############################################
    # generate ground truth, observation and objective functional
    ###############################################

    # parameters
    noise_level=.1

    # create observation
    observation=observations[run:run+1]

    # objective functional
    F=lambda x: torch.sum(torch.reshape((forward_operator(x)-observation[0])**2,[x.shape[0],-1]),dim=-1)
    print('###########################################')
    print(run,F(ground_truths[run]).item())


    ###############################################
    # generate initialization
    ###############################################

    # generate candidates
    n_candidates=2
    candidates=mix_vae.sample(n_candidates)

    # eval objective
    objective_values=F(candidates)
    candidate_index=torch.argmin(objective_values).detach().cpu().numpy()
    print(objective_values)
    print(candidate_index,objective_values[candidate_index])
    init=candidates[candidate_index]

    ###############################################
    # gradient descent
    ###############################################
    step_size=5e+0
    steps=500
    print(init.shape)
    trajectory,speed,objective=mix_vae.create_trajectory(init[None,:,:,:],F,step_size,steps,retraction='chart_retraction',use_prior_lambda=None)

    result=torch.tensor(trajectory[-1],device=device,dtype=torch.float)
    results.append(result)

results=torch.stack(results,0)[:,None,:,:]
im=torchvision.utils.make_grid(torch.tensor(results,dtype=torch.float,device=device),nrow=20).detach().cpu().numpy()
im=np.transpose(im,(1,2,0))
im=np.minimum(np.maximum(im,0.),1.)
plt.imsave('bar_res_grid.png',im)
