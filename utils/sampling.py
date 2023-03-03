#!/usr/bin/env python

# This file contains all functions and specifications regarding the modelling of the latent space 
# and the ground truth distributions, i.e., sampling functions and energy functions

import torch
import numpy as np
from utils.general import get_device
import sklearn.datasets

device = get_device()

###############################################
# LATENT SPACE UTILS 
###############################################

def get_linear_smoothed_uniform_latent():
    # standard
    lambd_bd=100.
    def inverse_cdf_prior(x,lambd_bd):
        x*=(2*lambd_bd+2)/lambd_bd
        y=np.zeros_like(x)
        left=x<1/lambd_bd
        y[left]=np.log(x[left]*lambd_bd)-1
        middle=np.logical_and(x>=1/lambd_bd,x < 2+1/lambd_bd)
        y[middle]=x[middle]-1/lambd_bd-1
        right=x>=2+1/lambd_bd
        y[right]=-np.log(((2+2/lambd_bd)-x[right])*lambd_bd)+1
        return y
    sample_latent=lambda shape: torch.tensor(inverse_cdf_prior(np.random.uniform(size=shape),lambd_bd),dtype=torch.float,device=device)

    def latent_energy(x):
        if len(x.shape)>1:
            return lambd_bd*torch.sum(torch.maximum(x-1,torch.tensor(0.,dtype=torch.float,device=device))-torch.minimum(x+1,torch.tensor(0.,dtype=torch.float,device=device)),-1)
        else:
            return lambd_bd*(torch.maximum(x-1,torch.tensor(0.,dtype=torch.float,device=device))-torch.minimum(x+1,torch.tensor(0.,dtype=torch.float,device=device)))
    relevant_region=[-1.,1.]
    return sample_latent,latent_energy,relevant_region


def get_spline_latent(lambda_bd=100.):
    low=torch.tensor(.05,dtype=torch.float,device=device)
    supp=0.2
    der=(1-low)/supp
    def latent_energy(x):
        pw_energy=torch.zeros_like(x)
        linear_decay=torch.logical_and(torch.abs(x)<torch.tensor(1.,dtype=torch.float,device=device),torch.abs(x)>torch.tensor(1.-supp,dtype=torch.float,device=device))
        pw_energy[linear_decay]+=(-torch.log(low-der*(torch.abs(x[linear_decay])-1)))
        exp_decay=torch.abs(x)>=torch.tensor(1.,dtype=torch.float,device=device)
        pw_energy[exp_decay]+=(lambda_bd*(torch.abs(x[exp_decay])-1)-torch.log(low))
        if len(x.shape)>1:
            return torch.sum(pw_energy,-1)
        else:
            return pw_energy
    sample_latent= lambda shape: rejection_sampling(shape,latent_energy,0.,get_linear_smoothed_uniform_latent)
    relevant_region=[-1.,1.]
    return sample_latent,latent_energy,relevant_region

###############################################
# DATA GENERATION
###############################################

def gen_sphere(n,noise_level=0.01):
    x=torch.randn((n,3))
    x=x/torch.sqrt(torch.sum(x**2,dim=-1)[:,None])
    x+=noise_level*torch.randn_like(x)
    return x.to(device)

def gen_ring(n,noise_level=0.01):
    x=torch.randn((n,2))
    x=x/torch.sqrt(torch.sum(x**2,dim=-1)[:,None])
    length=.5+torch.rand(n)
    x=x*length[:,None]
    x+=noise_level*torch.randn_like(x)
    return x.to(device)

def gen_two_circles(n,noise_level=0.01):
    x=torch.randn((n,2))
    x=x/torch.sqrt(torch.sum(x**2,dim=-1)[:,None])
    shift=3
    add=torch.randint(low=0,high=2,size=(n,))*shift
    x[:,0]+=add-.5*shift
    x+=noise_level*torch.randn_like(x)
    return x.to(device)

def gen_torus(n,noise_level=0.01):
    circle1=torch.randn((n,2))
    circle1=3*circle1/torch.sqrt(torch.sum(circle1**2,dim=-1)[:,None])
    circle2=torch.randn((n,2))
    circle2=1.*circle2/torch.sqrt(torch.sum(circle2**2,dim=-1)[:,None])
    x=torch.zeros((n,3))
    x[:,:2]+=circle1
    x[:,2]+=circle2[:,1]
    x[:,:2]+=circle2[:,0:1]*x[:,:2]/3
    x+=noise_level*torch.randn_like(x)
    return x.to(device)

def gen_swiss_roll(n,noise_level=0.01):
    data=sklearn.datasets.make_swiss_roll(n_samples=n, noise=noise_level)[0]
    return torch.tensor(data,device=device,dtype=torch.float)

###############################################
# OTHER
###############################################

def rejection_sampling(shape,target_energy,energy_quotient_lower_bound,get_proposal):
    # general implementation of rejection sampling.
    sample_proposal,proposal_energy,_=get_proposal()
    num_samples=shape[0]
    samples=np.zeros([0]+list(shape[1:]))
    while samples.shape[0]<num_samples:
        proposal_samples=sample_proposal(shape)
        energies_prop=proposal_energy(proposal_samples).detach().cpu().numpy()
        energies_target=target_energy(proposal_samples).detach().cpu().numpy()
        proposal_samples=proposal_samples.detach().cpu().numpy()
        energy_quotient=energies_target-energies_prop-energy_quotient_lower_bound
        neg_log_uniforms=-np.log(np.random.uniform(size=energy_quotient.shape))
        accepted_samples=proposal_samples[neg_log_uniforms>energy_quotient]
        samples=np.concatenate([samples,accepted_samples],0)
    samples=samples[:num_samples]
    samples=np.reshape(samples,shape)
    return torch.tensor(samples,device=device,dtype=torch.float)
