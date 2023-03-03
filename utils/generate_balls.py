#!/usr/bin/env python

import cv2
import math
import os
import numpy as np
import random

# original: eps=2
def overlaps(center,centers,radius,radii,eps=4):
    if centers.shape[0]==0:
        return False
    c_dists=np.sqrt(np.sum((centers-center)**2,-1))-eps-radius-radii
    if np.min(c_dists)<0:
        return True
    else:
        return False

def gen_balls_image(n_balls,seed=None,std_intensity=0.1,intensity_dist='gauss',eps=4,fix_int=False):
    x_c_v = 64
    y_c_v = 64
    centers=np.zeros((0,2))
    radii=np.zeros(0)
    if seed is None:
        rng=random.Random()
    else:
        rng=random.Random(seed)
    while centers.shape[0]<n_balls:
        radius = 12 + (25-12) * rng.uniform(0,1)    # tra 12 e 25
        radius_aux = 45 * rng.uniform(0,1)
        angle = 2*math.pi*rng.uniform(0,1)
        x_c = x_c_v + radius_aux * np.cos(angle)
        y_c = y_c_v + radius_aux * np.sin(angle)
        center_coordinates = np.array([x_c, y_c])
        dist_center = math.sqrt(pow((64-x_c),2) + pow((64-y_c),2))
        if dist_center+radius>=62:
            continue
        if overlaps(center_coordinates,centers,radius,radii,eps=eps):
            continue
        centers=np.concatenate([centers,center_coordinates[None,:]],0)
        radii=np.concatenate([radii,np.array([radius])],0)
    image = int(255*(1-0.2)/(1.8-0.2))*np.ones((128,128,3),np.uint8)
    conds=[]
    for i in range(n_balls):
        if intensity_dist=='gauss':
            cond = rng.gauss(1.5,std_intensity)
        elif intensity_dist=='uniform':
            cond = rng.uniform(1.5-std_intensity,1.5+std_intensity)
        else:
            raise NameError('Unknown intensity distribution!')
        if fix_int and i==n_balls-1:
            cond=1.5*n_balls-np.sum(np.array(conds))
        conds.append(cond)
        a = int(255*(cond-0.2)/(1.8-0.2))
        color = (a,a,a)
        image = cv2.circle(image, (int(centers[i,0]),int(centers[i,1])), int(radii[i]), color, -1)
    return np.mean(image,-1)/255.

def generate_bar(seed=None,width=50,height=10,centered=False,im_size=128,angle=None):
    if seed is None:
        rng=random.Random()
    else:
        rng=random.Random(seed)
    new_vals=True
    while new_vals:
        new_vals=False
        if angle is None:
            angle = 2*math.pi*rng.uniform(0,1)
        rotation_matrix=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        c1=rotation_matrix.dot(np.array([-.5*width,-.5*height]))
        c2=rotation_matrix.dot(np.array([.5*width,-.5*height]))
        c3=rotation_matrix.dot(np.array([.5*width,.5*height]))
        c4=rotation_matrix.dot(np.array([-.5*width,.5*height]))
        corners=np.stack([c1,c2,c3,c4])
        if not centered:
            shift=np.array([rng.uniform(0,1)*(im_size-1),rng.uniform(0,1)*(im_size-1)])
        else: 
            shift=np.array([.5*(im_size-1),.5*(im_size-1)])
        if np.max(corners+shift[None,:])>(im_size-5) or np.min(corners+shift[None,:])<5:
            new_vals=True
    rotation_matrix_inv=np.array([[np.cos(-angle),-np.sin(-angle)],[np.sin(-angle),np.cos(-angle)]])
    img=.5*np.ones((im_size,im_size))
    xv,yv=np.meshgrid(np.linspace(0,im_size-1,im_size),np.linspace(0,im_size-1,im_size))
    points=np.stack([xv,yv],-1)[:,:,:,None]
    points=points-shift[None,None,:,None]
    rot_points=np.matmul(rotation_matrix_inv,points).squeeze()
    within_square=np.logical_and(rot_points[:,:,0]>=-.5*width,rot_points[:,:,0]<=.5*width)
    within_square2=np.logical_and(rot_points[:,:,1]>=-.5*height,rot_points[:,:,1]<=.5*height)
    within_square=np.logical_and(within_square,within_square2)
    img[within_square]+=.25
    return img
