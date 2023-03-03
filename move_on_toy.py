# This script minimizes a function f on the learned manifold of the toy examples.
# It is implemented for the toy examples 'ring', 'two_circles', 'sphere' and 'torus'

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
import math
from utils.plot import plot_trajectories
from utils.sampling import gen_torus,gen_swiss_roll

###############################################
# Global parameters
###############################################

device = get_device()

deterministic=True
if deterministic:
    np.random.seed(20)
    torch.manual_seed(20)
    random.seed(20)

models_dir='models'
plots_dir='plots'

###############################################
# Problem specific parameters
###############################################

model_name='ring'
#model_name='two_circles'
#model_name='sphere'
#model_name='torus'

if model_name=='ring':
    sig_ld=0.01
    sig_hd=.01
    num_generators=2
    dim_hd=2
    dim_ld=2
    points=[]
    point=torch.tensor([[1.,0.4]],device=device,dtype=torch.float)
    points.append(point)
    point=torch.tensor([[1.,-0.4]],device=device,dtype=torch.float)
    points.append(point)
    f=lambda x: .1*torch.sum((x-torch.tensor([[-1,0]],dtype=torch.float,device=device))**2)
    steps=3000
elif model_name=='two_circles':
    sig_ld=0.01
    sig_hd=.01
    num_generators=4
    dim_hd=2
    dim_ld=1
    points=[]
    point=torch.tensor([[-1.,.2]],device=device,dtype=torch.float)
    point/=torch.sqrt(torch.sum(point**2,1,keepdim=True))
    point+=torch.tensor([[-1.5,0.]],device=device,dtype=torch.float)
    points.append(point)
    point=torch.tensor([[-1.,-.2]],device=device,dtype=torch.float)
    point/=torch.sqrt(torch.sum(point**2,1,keepdim=True))
    point+=torch.tensor([[-1.5,0.]],device=device,dtype=torch.float)
    points.append(point)
    point=torch.tensor([[1.,.2]],device=device,dtype=torch.float)
    point/=torch.sqrt(torch.sum(point**2,1,keepdim=True))
    point+=torch.tensor([[1.5,0.]],device=device,dtype=torch.float)
    points.append(point)
    point=torch.tensor([[1.,-.2]],device=device,dtype=torch.float)
    point/=torch.sqrt(torch.sum(point**2,1,keepdim=True))
    point+=torch.tensor([[1.5,0.]],device=device,dtype=torch.float)
    points.append(point)
    f=lambda x: torch.sum(x**2)
    steps=500
elif model_name=='sphere':
    sig_ld=0.01
    sig_hd=.01
    num_generators=2
    dim_hd=3
    dim_ld=2
    points=[]
    n_starts=10
    scale=0.3
    for n in range(n_starts):
        angle=2*math.pi*(n*1./n_starts)
        point=torch.tensor([[scale*np.cos(angle),scale*np.sin(angle),1.]],device=device,dtype=torch.float)
        point/=torch.sqrt(torch.sum(point**2,1,keepdim=True))
        points.append(point)
    f=lambda x: torch.sum((x-torch.tensor([[0,0,-2]],dtype=torch.float,device=device))**2)
    steps=500
elif model_name=='torus':
    sig_ld=0.01
    sig_hd=.05
    num_generators=6
    dim_hd=3
    dim_ld=2
    points=[]
    n_starts=30
    for n in range(n_starts):
        points.append(gen_torus(1,noise_level=0))
    f=lambda x: torch.sum((x-torch.tensor([[-5,0,0]],dtype=torch.float,device=device))**2)
    steps=500

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
mix_vae.load_checkpoint(models_dir+'/mix_vae_'+model_name+'_'+str(num_generators)+'_gen_last.pt')

###############################################
# Gradient descent on manifold
###############################################

trajectories=[]
speeds=[]
objectives=[]
for point in points:
    trajectory,speed,objective=mix_vae.create_trajectory(point,f,1e-2,steps,retraction='project',use_prior_lambda=1e-1)
    trajectories.append(trajectory)
    speeds.append(speed)
    objectives.append(objective)

plot_trajectories(plots_dir,model_name,trajectories,speeds=speeds,objectives=objectives,show=True)

