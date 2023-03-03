# Produces the trajectory figure in the deblurring example of the paper

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

deterministic=True
seed=20
if deterministic:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

###############################################
# Dataset and VAE parameters
###############################################


device = get_device()
base_dir=get_base_dir()

sig_ld=.01
sig_hd=.1
num_generators=1
dim_ld=1
n_runs=10

models_dir=base_dir+'/models'
model_name='bar'
plots_dir=base_dir+'/plots'
plots_dir_full=plots_dir+'/'+model_name+'_'+str(num_generators)+'_gen'

batch_size=10
test_batch_size=10
n_data=50000

train_dataset=BarDataset(epoch_size=n_data,start_seed=0,transform=Jittering(),centered=True)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

test_dataset=BarDataset(epoch_size=10000,start_seed=0,transform=Jittering(),centered=True)    
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)


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
    decoders.append(InjectiveMultiscaleGenerator(dim_ld,sig_hd,sig_ld,utils.sampling.get_spline_latent,latent_nf=dim_ld>1).to(device))

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

noise_level=.1
init_ang=.0*math.pi
ground_truth=torch.tensor(generate_bar(centered=True,angle=init_ang),dtype=torch.float,device=device)[None,:,:]
blurred=forward_operator(ground_truth)
observation=blurred+noise_level*torch.randn_like(blurred)
im_gt=ground_truth.squeeze().detach().cpu().numpy()
im_gt=np.tile(np.minimum(np.maximum(im_gt,0.),1.)[:,:,None],(1,1,3))
im_obs=observation.squeeze().detach().cpu().numpy()
im_obs=np.tile(np.minimum(np.maximum(im_obs,0.),1.)[:,:,None],(1,1,3))
F=lambda x: torch.sum(torch.reshape((forward_operator(x)-observation[0])**2,[x.shape[0],-1]),dim=-1)
plt.imsave(plots_dir_full+'/move_on_bar_observation.png',im_obs)
plt.imsave(plots_dir_full+'/move_on_bar_gt.png',im_gt)

graphic_imgs=[]
n_imgs=20
for nr in range(1,10):
    ang=(nr*.1*math.pi+init_ang)%(2*math.pi)
    init=torch.tensor(generate_bar(centered=True,angle=ang),dtype=torch.float,device=device)[None,:,:]

    im_in=init.squeeze().detach().cpu().numpy()
    im_in=np.tile(np.minimum(np.maximum(im_in,0.),1.)[:,:,None],(1,1,3))
    plt.imsave(plots_dir_full+'/move_on_bar_init_'+str(nr)+'.png',im_in)

    step_size=5e+0
    steps=250
    trajectory,speed,objective=mix_vae.create_trajectory(init[None,:,:,:],F,step_size,steps,retraction='chart_retraction',use_prior_lambda=None)

    result=trajectory[-1]
    res=torch.tensor(result[None,None,:],device=device,dtype=torch.float)
    im_res=result.squeeze()
    im_res=np.tile(np.minimum(np.maximum(im_res,0.),1.)[:,:,None],(1,1,3))
    plt.imsave(plots_dir_full+'/move_on_bar_result_'+str(nr)+'.png',im_res)

    im_all=np.concatenate([im_obs,np.zeros((im_gt.shape[0],2,3)),im_gt,np.zeros((im_gt.shape[0],2,3)),im_res,np.zeros((im_gt.shape[0],2,3)),im_in],1)
    plt.imsave(plots_dir_full+'/move_on_bar_all_'+str(nr)+'.png',im_all)
    for i in range(n_imgs):
        graphic_imgs.append(trajectory[round(i*(steps-1)/(n_imgs-1))][None,:,:])

    if num_generators==2:
        data=[[],[]]
        for i in range(steps):
            step=i
            probs=mix_vae.classify(torch.tensor(trajectory[step][None,None,:,:],dtype=torch.float,device=device)).detach().squeeze().cpu().numpy()
            data[0].append(probs[0])
            data[1].append(probs[1])
        fig=plt.figure(figsize=(20,1))
        plt.plot(data[0])
        plt.plot(data[1])
        plt.savefig(plots_dir_full+'/chart_selection_'+str(nr)+'.png', bbox_inches='tight',pad_inches = 0.05)
        plt.close(fig)

    import imageio
    images = []
    for im in trajectory:
        im=im.squeeze()
        im=np.tile(np.minimum(np.maximum(im,0.),1.)[:,:,None],(1,1,3))*255
        im=im.astype(np.uint8)
        images.append(im)
    imageio.mimsave(plots_dir_full+'/move_on_bar_gif_'+str(nr)+'.gif', images)

graphic_imgs=np.stack(graphic_imgs,0)
im=torchvision.utils.make_grid(torch.tensor(graphic_imgs,dtype=torch.float,device=device),nrow=n_imgs).detach().cpu().numpy()
im=np.transpose(im,(1,2,0))
im=np.minimum(np.maximum(im,0.),1.)
plt.imsave(plots_dir_full+'/move_on_balls_grid.png',im)

