#!/usr/bin/env python

# This file contains all plot functions.

import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from utils.general import get_device
import torchvision

device = get_device()

def generate_images(mix_vae,plots_dir,model_name,n_samples=10,generator=None):
    if generator is None:
        samples=mix_vae.sample(n_samples**2)
    else:
        samples=mix_vae.sample_gen(n_samples**2,generator)
    xs_gen=torch.clamp(samples,min=0,max=1).detach().cpu().numpy()
    xs_gen=np.reshape(xs_gen,[n_samples,n_samples,-1])
    xs_gen=np.reshape(xs_gen,[xs_gen.shape[0],xs_gen.shape[1],samples.shape[-2],samples.shape[-1]])
    xs_gen=np.reshape(xs_gen,[n_samples**2,1,samples.shape[-2],samples.shape[-1]])
    im=torchvision.utils.make_grid(torch.tensor(xs_gen,dtype=torch.float,device=device),nrow=n_samples).detach().cpu().numpy()
    im=np.transpose(im,(1,2,0))
    im=np.minimum(np.maximum(im,0.),1.)
    if generator is None:
        plt.imsave(plots_dir+'/'+model_name+'_grid.png',im)
    else:
        plt.imsave(plots_dir+'/'+model_name+'_grid_ng_'+str(generator)+'.png',im)

def generate_image_charts(mix_vae,plots_dir,model_name,generator,n_samples):
    dim_ld=mix_vae.decoders[0].dim_ld
    if dim_ld==1:
        generate_image_charts_1(mix_vae,plots_dir,model_name,generator,n_samples)
    elif dim_ld==3:
        generate_image_charts_3(mix_vae,plots_dir,model_name,generator,n_samples)
    else:
        generate_image_charts_n(mix_vae,plots_dir,model_name,generator,n_samples,dim_ld)

def plot_trajectories(plots_dir,model_name,trajectories,speeds=None,objectives=None,show=False):
    dim=trajectories[0].shape[1]
    if dim==2:
        plot_trajectories_2(plots_dir,model_name,trajectories,speeds,objectives,show=show)
    elif dim==3:
        plot_trajectories_3(plots_dir,model_name,trajectories,speeds,objectives,show=show)
    else:
        print('No plot function for this dimensions! Skip plotting trajectories!')

def plot_all(mix_vae,data,plots_dir,model_name,show=False):
    plot_charts(mix_vae,plots_dir,model_name,show=show)
    plot_generated(mix_vae,plots_dir,model_name,show=show)
    plot_data(data,plots_dir,model_name,show=show)
    plot_classification(mix_vae,data,plots_dir,model_name,show=show)

def plot_charts(mix_vae,plots_dir,model_name,show=False):
    with torch.no_grad():
        if mix_vae.decoders[0].dim_hd==3 and mix_vae.decoders[0].dim_ld==2:
            plot_charts_32(mix_vae,plots_dir,model_name,show)
        elif mix_vae.decoders[0].dim_hd==2 and mix_vae.decoders[0].dim_ld==1:
            plot_charts_21(mix_vae,plots_dir,model_name,show)
        elif mix_vae.decoders[0].dim_hd==2 and mix_vae.decoders[0].dim_ld==2:
            plot_charts_22(mix_vae,plots_dir,model_name,show)
        else:
            print('No plot function for this dimensions! Skip plotting charts!')


def plot_generated(mix_vae,plots_dir,model_name,show=False):
    with torch.no_grad():
        if mix_vae.decoders[0].dim_hd==3:
            plot_generated_3(mix_vae,plots_dir,model_name,show)
        elif mix_vae.decoders[0].dim_hd==2:
            plot_generated_2(mix_vae,plots_dir,model_name,show)
        else:
            print('No plot function for this dimensions! Skip plotting generated samples!')

def plot_data(data,plot_dir,model_name,show=False):
    with torch.no_grad():
        if data.shape[1]==2:
            plot_data_2(data,plot_dir,model_name,show)
        elif data.shape[1]==3:
            plot_data_3(data,plot_dir,model_name,show)
        else:
            print('No plot function for this dimensions! Skip plotting data!')

def plot_classification(mix_vae,data,plot_dir,model_name,show=False):
    with torch.no_grad():
        if data.shape[1]==2:
            plot_classification_2(mix_vae,data,plot_dir,model_name,show)
        elif data.shape[1]==3:
            plot_classification_3(mix_vae,data,plot_dir,model_name,show)
        else:
            print('No plot function for this dimensions! Skip plotting classification!')

def plot_charts_32(mix_vae,plots_dir,model_name,show):
    num_generators=len(mix_vae.decoders)
    X = np.arange(mix_vae.relevant_region[0],mix_vae.relevant_region[1], 0.01)
    Y = np.arange(mix_vae.relevant_region[0],mix_vae.relevant_region[1], 0.01)
    X, Y = np.meshgrid(X, Y)
    X_shape=X.shape
    local_coords=torch.tensor(np.stack([np.reshape(X,[-1]),np.reshape(Y,[-1])],-1),device=device,dtype=torch.float)
    fig, ax = plt.subplots(1,num_generators,subplot_kw={"projection": "3d"},figsize=(4*num_generators,4))
    if num_generators>1:
        for ng in range(num_generators):
            xs_gen=mix_vae.decoders[ng](local_coords).detach().cpu().numpy()
            Xs=np.reshape(xs_gen[:,0],X_shape)
            Ys=np.reshape(xs_gen[:,1],X_shape)
            Zs=np.reshape(xs_gen[:,2],X_shape)
            surf = ax[ng].plot_surface(Xs, Ys, Zs, cmap=cm.coolwarm,linewidth=0)
            ax[ng].set_box_aspect([ub - lb for lb, ub in (getattr(ax[ng], f'get_{a}lim')() for a in 'xyz')])
    else:
        xs_gen=mix_vae.decoders[0](local_coords).detach().cpu().numpy()
        Xs=np.reshape(xs_gen[:,0],X_shape)
        Ys=np.reshape(xs_gen[:,1],X_shape)
        Zs=np.reshape(xs_gen[:,2],X_shape)
        surf = ax.plot_surface(Xs, Ys, Zs, cmap=cm.coolwarm,linewidth=0)
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    plt.tight_layout()
    fig.savefig(plots_dir+'/plot_'+model_name+'_'+str(num_generators)+'_gen.pdf', bbox_inches='tight',pad_inches = 0.05)
    if show:
        plt.show()
    plt.close(fig)

def plot_charts_22(mix_vae,plots_dir,model_name,show):
    num_generators=len(mix_vae.decoders)
    X = np.arange(mix_vae.relevant_region[0],mix_vae.relevant_region[1], 0.1)
    Y = np.arange(mix_vae.relevant_region[0],mix_vae.relevant_region[1], 0.1)
    X, Y = np.meshgrid(X, Y)
    X_shape=X.shape
    fig=plt.figure()
    colors=['b','g','r','c','m','y','k']
    local_coords=torch.tensor(np.stack([np.reshape(X,[-1]),np.reshape(Y,[-1])],-1),device=device,dtype=torch.float)
    for ng in range(num_generators):
        color_ind=ng
        while color_ind>len(colors):
            color_ind-=len(colors)
        xs_gen=mix_vae.decoders[ng](local_coords).detach().cpu().numpy()
        Xs=np.reshape(xs_gen[:,0],X_shape)
        Ys=np.reshape(xs_gen[:,1],X_shape)
        for i in range(X_shape[0]):
            plt.plot(Xs[:,i],Ys[:,i],colors[ng]+'-',linewidth=.2)
            plt.plot(Xs[i,:],Ys[i,:],colors[ng]+'-',linewidth=.2)
    plt.axis('equal')
    plt.tight_layout()
    fig.savefig(plots_dir+'/plot_'+model_name+'_'+str(num_generators)+'_gen.pdf', bbox_inches='tight',pad_inches = 0.05)
    if show:
        plt.show()
    plt.close(fig)

def plot_charts_21(mix_vae,plots_dir,model_name,show):
    num_generators=len(mix_vae.decoders)
    fig=plt.figure()
    points=torch.linspace(mix_vae.relevant_region[0],mix_vae.relevant_region[1],10000)
    points=points[:,None]
    for ng in range(num_generators):
        xs_gen=mix_vae.decoders[ng](points).detach().cpu().numpy()
        plt.plot(xs_gen[:,0],xs_gen[:,1])
    plt.axis('equal')
    plt.tight_layout()
    fig.savefig(plots_dir+'/plot_'+model_name+'_'+str(num_generators)+'_gen.pdf', bbox_inches='tight',pad_inches = 0.05)
    if show:
        plt.show()
    plt.close(fig)

def plot_generated_3(mix_vae,plots_dir,model_name,show):
    num_generators=len(mix_vae.decoders)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for ng in range(num_generators):
        xs_gen=mix_vae.sample_gen(2000,ng).detach().cpu().numpy()
        if model_name=='torus':
            take=np.logical_not(np.logical_or(np.any(xs_gen>4.2,1),np.any(xs_gen<-4.2,1)))
            xs_gen=xs_gen[take,:]
        elif model_name=='sphere':
            take=np.logical_not(np.logical_or(np.any(xs_gen>1.2,1),np.any(xs_gen<-1.2,1)))
            xs_gen=xs_gen[take,:]
        elif model_name=='swiss':
            take_not1=np.logical_or(xs_gen[:,0]>16.,xs_gen[:,0]<-11.)
            take_not2=np.logical_or(xs_gen[:,1]>24.,xs_gen[:,1]<-2.)
            take_not3=np.logical_or(xs_gen[:,2]>16.,xs_gen[:,2]<-12.)
            take=np.logical_not(np.logical_or(take_not1,np.logical_or(take_not2,take_not3)))
            xs_gen=xs_gen[take,:]
        ax.scatter(xs_gen[:,0],xs_gen[:,1],xs_gen[:,2],s=1)
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    plt.tight_layout()
    fig.savefig(plots_dir+'/plot_relevant_'+model_name+'_'+str(num_generators)+'_gen.pdf', bbox_inches='tight',pad_inches = 0.05)
    if show:
        plt.show()
    plt.close(fig)

def plot_generated_2(mix_vae,plots_dir,model_name,show):
    fig=plt.figure()
    for ng in range(len(mix_vae.decoders)):
        xs_gen=mix_vae.sample_gen(20*(10**mix_vae.decoders[0].dim_ld),ng).detach().cpu().numpy()
        if model_name=='two_circles':
            take=np.logical_not(np.logical_or(np.abs(xs_gen[:,0])>3.,np.abs(xs_gen[:,1])>2.))
            xs_gen=xs_gen[take,:]
        if model_name=='ring':
            take=np.logical_not(np.logical_or(np.abs(xs_gen[:,0])>1.6,np.abs(xs_gen[:,1])>1.6))
            xs_gen=xs_gen[take,:]
        plt.scatter(xs_gen[:,0],xs_gen[:,1],s=1)
    plt.axis('equal')
    plt.tight_layout()
    fig.savefig(plots_dir+'/plot_relevant_'+model_name+'_'+str(len(mix_vae.decoders))+'_gen.pdf', bbox_inches='tight',pad_inches = 0.05)
    if show:
        plt.show()
    plt.close(fig)

def plot_data_3(data,plots_dir,model_name,show):
    dat=data.detach().cpu().numpy()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(dat[:,0],dat[:,1],dat[:,2],s=1)
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    plt.tight_layout()
    fig.savefig(plots_dir+'/data_'+model_name+'.pdf', bbox_inches='tight',pad_inches = 0.05)
    if show:
        plt.show()
    plt.close()

def plot_data_2(data,plots_dir,model_name,show):
    fig=plt.figure()
    dat=data.detach().cpu().numpy()
    plt.scatter(dat[:,0],dat[:,1],s=1)
    plt.axis('equal')
    plt.tight_layout()
    fig.savefig(plots_dir+'/data_'+model_name+'.pdf', bbox_inches='tight',pad_inches = 0.05)
    if show:
        plt.show()
    plt.close()

def plot_classification_3(mix_vae,data,plots_dir,model_name,show):
    num_generators=len(mix_vae.decoders)
    dat=torch.tensor(data.detach().cpu().numpy(),dtype=torch.float,device=device)
    probs=mix_vae.classify(dat).detach().cpu().numpy()
    labs=np.argmax(probs,axis=-1)
    dat=dat.detach().cpu().numpy()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for ng in range(num_generators):
        classified_xs=dat[labs==ng]
        ax.scatter(classified_xs[:,0],classified_xs[:,1],classified_xs[:,2],s=1)
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    plt.tight_layout()
    if show:
        plt.show()
    fig.savefig(plots_dir+'/class_'+model_name+'_'+str(num_generators)+'_gen.pdf', bbox_inches='tight',pad_inches = 0.05)
    plt.close(fig)

def plot_classification_2(mix_vae,data,plots_dir,model_name,show):
    num_generators=len(mix_vae.decoders)
    dat=torch.tensor(data.detach().cpu().numpy(),dtype=torch.float,device=device)
    probs=mix_vae.classify(dat).detach().cpu().numpy()
    labs=np.argmax(probs,axis=-1)
    fig=plt.figure()
    dat=dat.detach().cpu().numpy()
    for ng in range(num_generators):
        classified_xs=dat[labs==ng]
        plt.scatter(classified_xs[:,0],classified_xs[:,1],s=1)
    plt.axis('equal')
    plt.tight_layout()
    fig.savefig(plots_dir+'/class_'+model_name+'_'+str(num_generators)+'_gen.pdf', bbox_inches='tight',pad_inches = 0.05)
    if show:
        plt.show()
    plt.close(fig)

def plot_trajectories_2(plots_dir,model_name,trajectories,speeds,objectives,show):
    if not speeds is None:
        fig=plt.figure()
        for speed in speeds:
            plt.plot(speed)
        if show:
            plt.show()
        plt.close(fig)
    if not objectives is None:
        fig=plt.figure()
        for objective in objectives:
            plt.plot(objective)
        if show:
            plt.show()
        plt.close(fig)
    fig=plt.figure()
    for trajectory in trajectories:
        plt.plot(trajectory[:,0],trajectory[:,1])
    plt.axis('equal')
    plt.tight_layout()
    fig.savefig(plots_dir+'/trajectory_'+model_name+'.pdf', bbox_inches='tight',pad_inches = 0.05)
    if show:
        plt.show()
    plt.close(fig)

def plot_trajectories_3(plots_dir,model_name,trajectories,speeds,objectives,show):
    if not speeds is None:
        fig=plt.figure()
        for speed in speeds:
            plt.plot(speed)
        if show:
            plt.show()
        plt.close(fig)
    if not objectives is None:
        fig=plt.figure()
        for objective in objectives:
            plt.plot(objective)
        if show:
            plt.show()
        plt.close(fig)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for trajectory in trajectories:
        plt.plot(trajectory[:,0],trajectory[:,1],trajectory[:,2])
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    plt.tight_layout()
    fig.savefig(plots_dir+'/trajectory_'+model_name+'.pdf', bbox_inches='tight',pad_inches = 0.05)
    if show:
        plt.show()
    plt.close(fig)

def generate_image_charts_3(mix_vae,plots_dir,model_name,generator,n_samples):
    X = np.arange(mix_vae.relevant_region[0],mix_vae.relevant_region[1], (mix_vae.relevant_region[1]-mix_vae.relevant_region[0])/n_samples)
    Y = np.arange(mix_vae.relevant_region[0],mix_vae.relevant_region[1], (mix_vae.relevant_region[1]-mix_vae.relevant_region[0])/n_samples)
    X, Y = np.meshgrid(X, Y)
    Z = np.arange(mix_vae.relevant_region[0],mix_vae.relevant_region[1], (mix_vae.relevant_region[1]-mix_vae.relevant_region[0])/n_samples)
    X_shape=X.shape

    for ng in range(len(mix_vae.decoders)):
        ims=[]
        for i,z_val in enumerate(Z):
            z=np.ones_like(X)*z_val
            local_coords=torch.tensor(np.stack([np.reshape(X,[-1]),np.reshape(Y,[-1]),np.reshape(z,[-1])],-1),device=device,dtype=torch.float)
            samples=mix_vae.decoders[ng](local_coords)
            xs_gen=torch.clamp(samples,min=0,max=1).detach().cpu().numpy()
            xs_gen=np.reshape(xs_gen,[X_shape[0],X_shape[1],-1])
            xs_gen=np.reshape(xs_gen,[xs_gen.shape[0],xs_gen.shape[1],samples.shape[-2],samples.shape[-1]])
            xs_gen=np.reshape(xs_gen,[n_samples**2,1,samples.shape[-2],samples.shape[-1]])
            im=torchvision.utils.make_grid(torch.tensor(xs_gen,dtype=torch.float,device=device),nrow=n_samples).detach().cpu().numpy()
            im=np.transpose(im,(1,2,0))
            im=np.minimum(np.maximum(im,0.),1.)
            plt.imsave(plots_dir+'/'+model_name+'_grid_ng_'+str(ng)+'_num_'+str(i)+'.png',im)
            ims.append(im)
        import imageio
        images = []
        for im in ims:
            im=im*255
            im=im.astype(np.uint8)
            images.append(im)
        imageio.mimsave(plots_dir+'/'+model_name+'_grid_ng_'+str(ng)+'_gif.gif', images,duration=2)

def generate_image_charts_2(mix_vae,plots_dir,model_name,generator,n_samples):
    X = np.arange(mix_vae.relevant_region[0],mix_vae.relevant_region[1], (mix_vae.relevant_region[1]-mix_vae.relevant_region[0])/n_samples)
    Y = np.arange(mix_vae.relevant_region[0],mix_vae.relevant_region[1], (mix_vae.relevant_region[1]-mix_vae.relevant_region[0])/n_samples)
    X, Y = np.meshgrid(X, Y)
    X_shape=X.shape

    for ng in range(len(mix_vae.decoders)):
        ims=[]
        local_coords=torch.tensor(np.stack([np.reshape(X,[-1]),np.reshape(Y,[-1])],-1),device=device,dtype=torch.float)
        samples=mix_vae.decoders[ng](local_coords)
        xs_gen=torch.clamp(samples,min=0,max=1).detach().cpu().numpy()
        xs_gen=np.reshape(xs_gen,[X_shape[0],X_shape[1],-1])
        xs_gen=np.reshape(xs_gen,[xs_gen.shape[0],xs_gen.shape[1],samples.shape[-2],samples.shape[-1]])
        xs_gen=np.reshape(xs_gen,[n_samples**2,1,samples.shape[-2],samples.shape[-1]])
        im=torchvision.utils.make_grid(torch.tensor(xs_gen,dtype=torch.float,device=device),nrow=n_samples).detach().cpu().numpy()
        im=np.transpose(im,(1,2,0))
        im=np.minimum(np.maximum(im,0.),1.)
        plt.imsave(plots_dir+'/'+model_name+'_grid_ng_'+str(ng)+'_chart.png',im)

def generate_image_charts_n(mix_vae,plots_dir,model_name,generator,n_samples,dim_ld):
    X = np.arange(mix_vae.relevant_region[0],mix_vae.relevant_region[1], (mix_vae.relevant_region[1]-mix_vae.relevant_region[0])/n_samples)
    Y = np.arange(mix_vae.relevant_region[0],mix_vae.relevant_region[1], (mix_vae.relevant_region[1]-mix_vae.relevant_region[0])/n_samples)
    X, Y = np.meshgrid(X, Y)
    X_shape=X.shape

    for ng in range(len(mix_vae.decoders)):
        z=np.zeros((X.shape[0]*X.shape[1],dim_ld-2))
        local_coords=torch.tensor(np.concatenate([np.stack([np.reshape(X,[-1]),np.reshape(Y,[-1])],-1),z],-1),device=device,dtype=torch.float)
        samples=mix_vae.decoders[ng](local_coords)
        xs_gen=torch.clamp(samples,min=0,max=1).detach().cpu().numpy()
        xs_gen=np.reshape(xs_gen,[X_shape[0],X_shape[1],-1])
        xs_gen=np.reshape(xs_gen,[xs_gen.shape[0],xs_gen.shape[1],samples.shape[-2],samples.shape[-1]])
        xs_gen=np.reshape(xs_gen,[n_samples**2,1,samples.shape[-2],samples.shape[-1]])
        im=torchvision.utils.make_grid(torch.tensor(xs_gen,dtype=torch.float,device=device),nrow=n_samples).detach().cpu().numpy()
        im=np.transpose(im,(1,2,0))
        im=np.minimum(np.maximum(im,0.),1.)
        plt.imsave(plots_dir+'/'+model_name+'_grid_ng_'+str(ng)+'_chart.png',im)


def generate_image_charts_1(mix_vae,plots_dir,model_name,generator,n_samples):
    X = np.arange(mix_vae.relevant_region[0],mix_vae.relevant_region[1], (mix_vae.relevant_region[1]-mix_vae.relevant_region[0])/n_samples)
    X_shape=X.shape

    for ng in range(len(mix_vae.decoders)):
        local_coords=torch.tensor(X,device=device,dtype=torch.float)[:,None]
        samples=mix_vae.decoders[ng](local_coords)
        xs_gen=torch.clamp(samples,min=0,max=1).detach().cpu().numpy()
        xs_gen=np.reshape(xs_gen,[n_samples,1,samples.shape[-2],samples.shape[-1]])
        im=torchvision.utils.make_grid(torch.tensor(xs_gen,dtype=torch.float,device=device),nrow=n_samples).detach().cpu().numpy()
        im=np.transpose(im,(1,2,0))
        im=np.minimum(np.maximum(im,0.),1.)
        plt.imsave(plots_dir+'/'+model_name+'_grid_ng_'+str(ng)+'_chart.png',im)

