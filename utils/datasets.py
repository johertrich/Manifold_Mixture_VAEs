#!/usr/bin/env python

import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import utils.generate_balls
import random
from utils.sampling import gen_ring,gen_swiss_roll,gen_torus,gen_two_circles,gen_sphere

class BallsDataset(Dataset):
    def __init__(self,  n_balls=2,epoch_size=50000,transform=None,start_seed=None,std_intensity=0.1,test=False,intensity_dist='gauss',eps=4,fix_int=False):     
        self.len=epoch_size
        self.transform=transform
        self.n_balls=n_balls
        self.start_seed=start_seed
        self.std_intensity=std_intensity
        self.test=test
        self.intensity_dist=intensity_dist
        self.eps=eps
        self.fix_int=fix_int
        
    def __len__(self):
        return self.len

    def reshuffle(self):
        self.start_seed+=self.len

    def __getitem__(self, idx):
        if self.start_seed is None:
            image = torch.tensor(utils.generate_balls.gen_balls_image(self.n_balls,std_intensity=self.std_intensity,intensity_dist=self.intensity_dist,eps=self.eps,fix_int=self.fix_int)[None,:],dtype=torch.float)
        else:
            seed=self.start_seed+idx
            if self.test:
                seed=-seed
            image = torch.tensor(utils.generate_balls.gen_balls_image(self.n_balls,seed=seed,std_intensity=self.std_intensity,intensity_dist=self.intensity_dist,eps=self.eps,fix_int=self.fix_int)[None,:],dtype=torch.float)
        if self.transform:
            image=self.transform(image)
        return image,idx

class BarDataset(Dataset):
    def __init__(self,  epoch_size=50000,transform=None,start_seed=None,test=False,centered=False):     
        self.len=epoch_size
        self.transform=transform
        self.start_seed=start_seed
        self.test=test
        self.centered=centered
        
    def __len__(self):
        return self.len

    def reshuffle(self):
        self.start_seed+=self.len

    def __getitem__(self, idx):
        if self.start_seed is None:
            image = torch.tensor(utils.generate_balls.generate_bar(centered=self.centered)[None,:],dtype=torch.float)
        else:
            seed=self.start_seed+idx
            if self.test:
                seed=-seed
            image = torch.tensor(utils.generate_balls.generate_bar(seed=seed,centered=self.centered)[None,:],dtype=torch.float)
        if self.transform:
            image=self.transform(image)
        return image,idx

class ToyDataset(Dataset):
    def __init__(self,sampler,length,noise_level=0.01,transform=None):
        self.length=length
        self.noise_level=noise_level
        self.sampler=sampler
        self.transform=transform
        self.reshuffle()

    def reshuffle(self):
        if self.sampler=='ring':
            self.data=gen_ring(self.length,noise_level=self.noise_level)
        elif self.sampler=='swiss':
            self.data=gen_swiss_roll(self.length,noise_level=self.noise_level)
        elif self.sampler=='sphere':
            self.data=gen_sphere(self.length,noise_level=self.noise_level)
        elif self.sampler=='torus':
            self.data=gen_torus(self.length,noise_level=self.noise_level)
        elif self.sampler=='two_circles':
            self.data=gen_two_circles(self.length,noise_level=self.noise_level)
        else:
            raise NameError('Sampler not found!')

    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        item=self.data[idx]
        if self.transform:
            item=self.transform(item)
        return item,idx

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=.01):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Jittering(object):
    def __call__(self, tensor):
        return (255*tensor + torch.rand_like(tensor))/256

    def __repr__(self):
        return self.__class__.__name__
