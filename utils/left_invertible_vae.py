#!/usr/bin/env python

# This file contains the class implementing the mixture of VAEs including its training routines.

import torch.nn as nn
import torch
import numpy as np
from utils.general import get_device
import copy

device = get_device()

class Mix_VAE(nn.Module):
    def __init__(self,decoders,learn_decoder_weights=False):
        super(Mix_VAE,self).__init__()
        self.decoders=decoders
        self.encoders=[]
        for ng,decoder in enumerate(self.decoders):
            self.add_module("Generator_"+str(ng),decoder)
            self.encoders.append(decoders[ng].backward)
        self.learn_decoder_weights=learn_decoder_weights
        log_decoder_weights=torch.zeros(len(self.decoders),dtype=torch.float,device=device)/len(self.decoders)
        self.log_decoder_weights=nn.Parameter(log_decoder_weights,requires_grad=False)
        self.relevant_region=self.decoders[0].relevant_region

    def get_log_decoder_weights(self):
        log_decoder_weights=self.log_decoder_weights-torch.max(self.log_decoder_weights)
        log_decoder_weights=log_decoder_weights-torch.log(torch.sum(torch.exp(log_decoder_weights)))
        return log_decoder_weights

    def set_sig_ld(self,sig_ld):
        for decoder in self.decoders:
            decoder.set_sig_ld(sig_ld)

    def classify(self,xs,return_logs=False,return_losses=False):
        losses=[]
        log_decoder_weights=self.get_log_decoder_weights()
        for ng in range(len(self.decoders)):
            loss=self.decoders[ng].negative_ELBO(xs)
            loss-=log_decoder_weights[ng]
            losses.append(loss.squeeze())
        losses=torch.stack(losses,-1)
        weights=-torch.clone(losses)
        weights=weights-torch.max(weights,-1,keepdim=True)[0]
        weights=torch.maximum(weights,torch.tensor(-25,dtype=torch.float,device=device))
        weights_sum=torch.log(torch.sum(torch.exp(weights),-1,keepdim=True))
        weights=weights-weights_sum
        if return_losses:
            if return_logs:
                return weights,losses
            weights=torch.exp(weights)
            return weights,losses
        if return_logs:
            return weights
        weights=torch.exp(weights)
        return weights

    def overlap_classify(self,xs,scale,return_logs=False,return_losses=False):
        losses=[]
        losses_add=[]
        log_decoder_weights=self.get_log_decoder_weights()
        for ng in range(len(self.encoders)):
            loss,loss_add=self.decoders[ng].negative_ELBO(xs,scale=scale)
            losses_add.append(loss_add)
            loss-=log_decoder_weights[ng]
            losses.append(loss.squeeze())
        losses=torch.stack(losses,-1)
        losses_add=torch.stack(losses_add,-1)
        if len(losses.shape)<2:
            losses=losses[None,:]
            losses_add=losses_add[None,:]
        weight_list=[]
        for ng in range(len(self.encoders)):
            losses_ng=torch.clone(losses)
            losses_ng[:,ng]+=losses_add[:,ng]
            losses_ng=-losses_ng
            losses_ng=losses_ng-torch.max(losses_ng,-1,keepdim=True)[0]
            losses_ng=torch.maximum(losses_ng,torch.tensor(-25,dtype=torch.float,device=device))
            losses_ng_sum=torch.log(torch.sum(torch.exp(losses_ng),-1,keepdim=True))
            weights_ng=losses_ng-losses_ng_sum
            weight_list.append(weights_ng)
        weights=torch.stack(weight_list,-1)
        weights=torch.max(weights,-1)[0]
        weights=weights-torch.max(weights,-1,keepdim=True)[0]
        weights=torch.maximum(weights,torch.tensor(-25,dtype=torch.float,device=device))
        weights_sum=torch.log(torch.sum(torch.exp(weights),-1,keepdim=True))
        weights=weights-weights_sum
        if return_losses:
            if return_logs:
                return weights,losses
            weights=torch.exp(weights)
            return weights,losses
        if return_logs:
            return weights
        weights=torch.exp(weights)
        return weights

    def create_trajectory(self,point,f,step_size,steps,f_der=None,retraction='project',print_steps=10,use_prior_lambda=None,add_noise=0.):
        with torch.no_grad():
            trajectory=[point.detach().cpu().squeeze().numpy()]
            speed=[]
            objectives=[]
            step=0
            while step<steps:
                chart_probs=self.classify(point).detach().cpu().squeeze().numpy()
                if len(chart_probs.shape)==0:
                    chart_probs=chart_probs[None]
                if step%print_steps==0:
                    dists=[]
                    dists_m=[]
                    for chart_nr in range(len(self.decoders)):
                        point2=point
                        point_chart=self.decoders[chart_nr](self.encoders[chart_nr](point2))
                        dists.append((torch.sum(point_chart-point2)**2).item())
                        dists_m.append((torch.mean(point_chart-point2)**2).item())
                    print(dists,dists_m)
                    print(step,chart_probs)
                point_new=torch.zeros_like(point)
                local_coord_norm_list=[]
                for chart_nr in range(len(self.decoders)):
                    if chart_probs[chart_nr]<1e-6:
                        # ignore all charts whose probability is approximately zero
                        continue
                    local_coord=self.encoders[chart_nr](point)
                    local_coord+=add_noise*torch.randn_like(local_coord)
                    local_coord_norm_list.append(torch.sum(local_coord**2).item())
                    local_coord.requires_grad_()
                    with torch.enable_grad():
                        jacobian=torch.autograd.functional.jacobian(self.decoders[chart_nr],local_coord,vectorize=True,strategy='forward-mode')
                    jacobian=torch.reshape(jacobian,(point.numel(),self.decoders[chart_nr].dim_ld))
                    jacobian_sym=torch.matmul(jacobian.transpose(0,1),jacobian)
                    if f_der is None:
                        with torch.enable_grad():
                            global_coords=self.decoders[chart_nr](local_coord)
                            obj=f(global_coords)
                            derivative=torch.autograd.grad(obj,local_coord)[0]
                    else:
                        global_coords=self.decoders[chart_nr](local_coord)
                        der,obj=f_der(global_coords.detach().cpu().numpy(),forward_sensitivities=jacobian.detach().cpu().numpy())
                        derivative=torch.tensor(der,dtype=torch.float,device=device)[None,:]
                    if not use_prior_lambda is None:
                        with torch.enable_grad():
                            loss=use_prior_lambda*torch.sum(torch.nn.functional.relu(torch.abs(local_coord)-1))
                            derivative_add=torch.autograd.grad(loss,local_coord)[0]
                        derivative=derivative+derivative_add
                    if retraction=='project':
                        riemannian_gradient=torch.linalg.solve(jacobian_sym,derivative[0])[None,:]
                        riemannian_gradient=torch.matmul(riemannian_gradient,jacobian.transpose(0,1))
                        riemannian_gradient=torch.reshape(riemannian_gradient,point.shape)
                        local_coord=self.encoders[chart_nr](point-step_size*riemannian_gradient)
                        chart_point_new=self.decoders[chart_nr](local_coord)
                    if retraction=='chart_retraction':
                        riemannian_gradient_chart_basis=torch.linalg.solve(jacobian_sym,derivative[0])[None,:]
                        local_coord=local_coord-step_size*riemannian_gradient_chart_basis
                        chart_point_new=self.decoders[chart_nr](local_coord)                
                    if retraction=='non-riemannian':
                        riemannian_gradient_chart_basis=derivative
                        local_coord=local_coord-step_size*riemannian_gradient_chart_basis
                        chart_point_new=self.decoders[chart_nr](local_coord)                
                    if step%print_steps==0:
                        print(step,obj,local_coord.detach().cpu().numpy())
                    point_new+=chart_probs[chart_nr]*chart_point_new
                if f_der is None:
                    objectives.append(obj.detach().cpu().numpy())
                else:
                    objectives.append(obj)
                if step>0:
                    speed.append(torch.sum((point_new-point)**2).detach().cpu().numpy())    
                    if step%print_steps==0:
                        print(speed[-1])
                point=point_new
                if step%print_steps==0:
                    print(point)
                trajectory.append(point.detach().cpu().squeeze().numpy())
                step+=1
            loss=torch.tensor(0,dtype=torch.float,device=device)
            if f_der is None:
                obj=f(point)+loss
            else:
                obj=f(point.detach().cpu().squeeze().numpy())+loss
            objectives.append(obj.detach().cpu().squeeze().numpy())
            trajectory=np.stack(trajectory)
            return trajectory,speed,objectives

    def create_trajectory_adaptive(self,point,f,step_size,steps,f_der=None,retraction='project',print_steps=1,use_prior_lambda=None,min_step_size=None,max_step_size=None):
        best_chart=-1
        with torch.no_grad():
            trajectory=[point.detach().cpu().squeeze().numpy()]
            speed=[]
            objectives=[]
            step=0
            loss=torch.tensor(0,dtype=torch.float,device=device)
            if f_der is None:
                old_obj=f(point)+loss
            else:
                old_obj=f(point.detach().cpu().squeeze().numpy())+loss
            objectives.append(old_obj.detach().cpu().squeeze().numpy())
            while step<steps:
                chart_probs=self.classify(point).detach().cpu().squeeze().numpy()
                dists=[]
                for chart_nr in range(len(self.decoders)):
                    local_coords=self.encoders[chart_nr](point)
                    point_chart=self.decoders[chart_nr](local_coords)
                    print(self.decoders[chart_nr].negative_ELBO(point),self.decoders[chart_nr].negative_ELBO(point_chart))
                    dist=torch.sum((point_chart-point)**2)
                    dists.append(dist.item())              
                print(dists)
                if len(chart_probs.shape)==0:
                    chart_probs=chart_probs[None]
                if step%print_steps==0:
                    print(step,chart_probs)
                local_coord_norm_list=[]
                riemannian_gradients_list=[]
                local_coord_list=[]
                omitted_prob=0
                for chart_nr in range(len(self.decoders)):
                    if chart_probs[chart_nr]<1e-6:
                        local_coord_list.append(None)
                        riemannian_gradients_list.append(None)
                        omitted_prob+=chart_probs[chart_nr]
                        # ignore all charts whose probability is approximately zero
                        continue
                    local_coord=self.encoders[chart_nr](point)                      
                    local_coord_list.append(local_coord)
                    local_coord_norm_list.append(torch.sum(local_coord**2).item())
                    local_coord.requires_grad_()
                    with torch.enable_grad():
                        jacobian=torch.autograd.functional.jacobian(self.decoders[chart_nr],local_coord,vectorize=True,strategy='forward-mode')
                    jacobian=torch.reshape(jacobian,(point.numel(),self.decoders[chart_nr].dim_ld))
                    jacobian_sym=torch.matmul(jacobian.transpose(0,1),jacobian)
                    if f_der is None:
                        with torch.enable_grad():
                            global_coords=self.decoders[chart_nr](local_coord)
                            obj=f(global_coords)
                            derivative=torch.autograd.grad(obj,local_coord)[0]
                    else:
                        global_coords=self.decoders[chart_nr](local_coord)
                        der,obj=f_der(global_coords.detach().cpu().numpy(),forward_sensitivities=jacobian.detach().cpu().numpy())
                        derivative=torch.tensor(der,dtype=torch.float,device=device)[None,:]
                    if not use_prior_lambda is None:
                        with torch.enable_grad():
                            loss=use_prior_lambda*torch.sum(torch.nn.functional.relu(-1-local_coord)+torch.nn.functional.relu(1+local_coord))
                            derivative_add=torch.autograd.grad(loss,local_coord)[0]
                        derivative=derivative+derivative_add
                    if retraction=='project':
                        riemannian_gradient=torch.linalg.solve(jacobian_sym,derivative[0])[None,:]
                        riemannian_gradient=torch.matmul(riemannian_gradient,jacobian.transpose(0,1))
                        riemannian_gradient=torch.reshape(riemannian_gradient,point.shape)
                    if retraction=='chart_retraction':
                        riemannian_gradient=torch.linalg.solve(jacobian_sym,derivative[0])[None,:]
                    if retraction=='non-riemannian':
                        riemannian_gradient=derivative
                    riemannian_gradients_list.append(riemannian_gradient)
                for i in range(20):
                    point_new=torch.zeros_like(point)
                    for chart_nr in range(len(self.decoders)):
                        if local_coord_list[chart_nr] is None:
                            continue
                        riemannian_gradient=riemannian_gradients_list[chart_nr]
                        local_coord=local_coord_list[chart_nr]
                        if retraction=='project':
                            local_coord=self.encoders[chart_nr](point-step_size*riemannian_gradient)
                            chart_point_new=self.decoders[chart_nr](local_coord)
                        if retraction=='chart_retraction' or retraction=='non-riemannian':
                            riemannian_gradient_chart_basis=riemannian_gradient
                            local_coord=local_coord-step_size*riemannian_gradient_chart_basis
                            chart_point_new=self.decoders[chart_nr](local_coord)
                        point_new+=chart_probs[chart_nr]*chart_point_new
                    point_new/=(1-omitted_prob)
                    loss=torch.tensor(0,dtype=torch.float,device=device)
                    if f_der is None:
                        obj_new=f(point_new)+loss
                    else:
                        obj_new=f(point_new.detach().cpu().numpy())+loss
                    if obj_new<=1.001*old_obj:
                        step_size*=1.5
                        if not max_step_size is None and step_size>max_step_size:
                            step_size=max_step_size
                        print('New step size:',step_size)
                        break
                    if not min_step_size is None and step_size<min_step_size:
                        step_size=min_step_size
                        print('New step size:',step_size)
                        break
                    step_size/=2.
                    print('reduce step size! Step:',step,'New step size:',step_size,'obj old:',old_obj.detach().cpu().numpy(),'obj new:',obj_new.detach().cpu().numpy(),'asdf',((point_new-point)**2).sum().detach().cpu().numpy())
                if obj_new>old_obj:
                    step_size/=3
                    print('reduce step size! New step size: ',step_size)
                if i>=99:
                    print('Step size could not be small enough!')
                    print(step_size)
                    break
                print('New objective:',obj_new.detach().cpu().squeeze().numpy())
                old_obj=obj_new
                objectives.append(obj_new.detach().cpu().numpy())
                if step>0:
                    speed.append(torch.sum((point_new-point)**2).detach().cpu().numpy())    
                    if step%print_steps==0:
                        print(speed[-1])
                point=point_new
                if step%print_steps==0:
                    print(local_coord)
                trajectory.append(point.detach().cpu().squeeze().numpy())
                step+=1
            trajectory=np.stack(trajectory)
            return trajectory,speed,objectives

    def train_epochs_dl(self,train_dataloader,optimizer,epochs=1,scale=None,set_sig_ld=None,normalize=True,gradient_clipping=None,first_epochs=None):
        if not train_dataloader.drop_last:
            return ValueError('Need drop last!')
        with torch.no_grad():
            log_weights_ten=torch.zeros((len(train_dataloader.dataset),len(self.decoders)),dtype=torch.float,device=device)
            losses_ten=torch.zeros((len(train_dataloader.dataset),len(self.decoders)),dtype=torch.float,device=device)
            i=0
            for xs,inds in train_dataloader:
                i+=1      
                xs=xs.to(device)
                if scale is None:
                    log_weights,losses=self.classify(xs,return_logs=True,return_losses=True)
                else:
                    log_weights,losses=self.overlap_classify(xs,scale,return_logs=True,return_losses=True)
                log_weights=log_weights.detach()
                log_weights_ten[inds]=log_weights
                losses_ten[inds]=losses
            log_weights=log_weights_ten
            losses=losses_ten-torch.min(losses,0,keepdim=True)[0]
            unnormalized_weights=torch.exp(log_weights)
            unnormalized_weight_mean=torch.mean(unnormalized_weights,0)
            print(unnormalized_weight_mean.detach().cpu().numpy())
            if self.learn_decoder_weights:
                self.log_decoder_weights.data=torch.log(unnormalized_weight_mean.detach())
            if normalize:
                # start normalize
                log_weights=log_weights-torch.max(log_weights,0,keepdim=True)[0]
                log_weights_sum=torch.log(torch.sum(torch.exp(log_weights),0,keepdim=True))                
                log_weights=log_weights-log_weights_sum
                # end normalize
            weights=torch.exp(log_weights)
            if not first_epochs is None:
                weights=weights*torch.exp(-first_epochs*losses.detach())
        if not set_sig_ld is None:
            self.set_sig_ld(set_sig_ld)
        for epoch in range(epochs):     
            if (epoch+1)%10==0:
                print(F'Inner epoch: {epoch+1}!')
            losses=[]
            losses_gs=0.
            weights_gs=0.
            for xs,inds in train_dataloader:
                if xs.shape[0]<train_dataloader.batch_size:
                    continue
                xs=xs.to(device)
                ws=weights[inds]
                loss,losses_g=self.train_step(xs,ws,optimizer,gradient_clipping=gradient_clipping)
                losses.append(loss)
                losses_gs+=losses_g
                weights_gs+=torch.sum(ws,0).detach().cpu().numpy()
            print('Train loss:',np.mean(losses))
            print((losses_gs/weights_gs))
        return np.mean(losses)

    def train_epochs_full_grad(self,train_dataloader,optimizer,scale=None,epochs=1,gradient_clipping=None,first_epochs=None,penalizer=None,equal_weights=False,Lipschitz_loss=None):
        for ep in range(epochs):
            if (ep+1)%10==0:
                print(F'Inner epoch: {ep+1}!')
            loss_vals=[]
            loss_vals_gen=[]
            penalizers=[]
            weight_sum=torch.zeros(len(self.decoders),dtype=torch.float,device=device)
            for xs,_ in train_dataloader:
                if xs.shape[0]<train_dataloader.batch_size:
                    continue
                xs=xs.to(device)
                optimizer.zero_grad()
                losses=[]
                losses_add=[]
                logpzs=[]
                log_decoder_weights=self.get_log_decoder_weights()
                for ng in range(len(self.decoders)):
                    loss,logpz=self.decoders[ng].negative_ELBO(xs,return_latent_probs=True)
                    loss-=log_decoder_weights[ng]
                    losses.append(loss.squeeze())
                    logpzs.append(logpz)
                logpzs=torch.stack(logpzs,-1)
                losses=torch.stack(losses,-1)
                weights=-torch.clone(losses)
                weights=weights-torch.max(weights,-1,keepdim=True)[0]
                weights_sum=torch.log(torch.sum(torch.exp(weights),-1,keepdim=True))
                weights=weights-weights_sum
                loss_penalizer=0.
                weights=torch.exp(weights)                    
                weights_sum=torch.sum(weights,0)
                weight_sum+=weights_sum
                loss_sum_0=torch.sum(losses*weights,0)
                loss_sum=torch.sum(loss_sum_0)+loss_penalizer
                loss_sum.backward()
                if not Lipschitz_loss is None:
                    for ng in range(len(self.encoders)):
                        loss=Lipschitz_loss*self.decoders[ng].Lipschitz_loss(xs.shape[0])
                        loss.backward()
                if not gradient_clipping is None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(),gradient_clipping)
                optimizer.step()
                loss_vals.append(loss_sum.item()/train_dataloader.batch_size)
                loss_vals_gen.append((loss_sum_0).detach().cpu().numpy())
            loss_vals_gen=np.stack(loss_vals_gen)/weight_sum.detach().cpu().numpy()
            decoder_weights=weight_sum/weight_sum.sum()
            pen_loss=np.sum(penalizers)/np.sum(weight_sum.detach().cpu().numpy())
            print(decoder_weights.detach().cpu().numpy())
            print('Train loss:',np.mean(loss_vals)-pen_loss,pen_loss)
            print(np.sum(loss_vals_gen,0))
        return np.mean(loss_vals)

    def train_step(self,xs,weights,optimizer,gradient_clipping=None):
        optimizer.zero_grad()
        losses=[]
        for ng in range(len(self.decoders)):
            loss=self.decoders[ng].negative_ELBO(xs)
            losses.append(loss.squeeze())
        losses=torch.stack(losses,-1)
        loss_sum_gs=torch.sum(losses*weights,0)
        loss_sum=torch.sum(loss_sum_gs)
        loss_sum.backward()
        avg_xs=torch.zeros_like(xs)
        if not gradient_clipping is None:
            torch.nn.utils.clip_grad_norm_(self.parameters(),gradient_clipping)
        optimizer.step()
        return (loss_sum/xs.shape[0]).item(),loss_sum_gs.detach().cpu().numpy()


    def seeding(self,train_dataloader,batch_size=None,num_samples=100,centers=None,init_epochs=500,learning_rate=1e-3,seeding_candidates=None):
        print('SEEDING...')
        if batch_size is None:
            batch_size=num_samples
        if seeding_candidates is None:
            seeding_candidates=len(self.decoders)
        closest=[]
        i=0
        for xs,inds in train_dataloader:
            xs.to(device)
            if i==0:
                if centers is None:
                    centers=[]
                    for c in range(seeding_candidates):
                        centers.append(xs[c])
                    centers=torch.stack(centers)
                centers_vec=centers.view(centers.shape[0],-1)
                centers_squared_norm=torch.sum(centers_vec**2,-1)
                centers_dists=centers_squared_norm[:,None]+centers_squared_norm[None,:]-2*torch.matmul(centers_vec,centers_vec.transpose(0,1))
                for i in range(centers.shape[0]):
                    centers_dists[i,i]=1e10
                while centers.shape[0]>len(self.decoders):
                    smallest_dists=torch.min(centers_dists,-1)
                    useless_center=torch.argmin(smallest_dists[0])
                    useless_center2=smallest_dists[1][useless_center]
                    centers_dists[useless_center,useless_center2]=1e10
                    centers_dists[useless_center2,useless_center]=1e10
                    second_smallest_dist=torch.min(centers_dists[useless_center,:])
                    second_smallest_dist2=torch.min(centers_dists[useless_center2,:])
                    if second_smallest_dist2<second_smallest_dist:
                        useless_center=useless_center2
                    centers=torch.cat((centers[:useless_center],centers[useless_center+1:]),0)
                    centers_dists=torch.cat((centers_dists[:useless_center,:],centers_dists[useless_center+1:,:]),0)
                    centers_dists=torch.cat((centers_dists[:,:useless_center],centers_dists[:,useless_center+1:]),1)
                for ng in range(len(self.decoders)):
                    closest.append(xs)
            else:
                for ng in range(len(self.decoders)):
                    closest[ng]=torch.cat((closest[ng],xs),0)
                    if closest[ng].shape[0]>num_samples:
                        dists=torch.sum(torch.reshape((closest[ng]-centers[ng])**2,[closest[ng].shape[0],-1]),dim=1)
                        perm=torch.argsort(dists)
                        closest[ng]=closest[ng][perm[:num_samples]]
            i+=1
        print('found closest...')
        for ng in range(len(self.decoders)):
            print(F'Initialize decoder {ng}')
            optimizer=torch.optim.Adam(self.decoders[ng].parameters(), lr = learning_rate)
            best_loss=1e10
            best_states=None
            for ep in range(init_epochs):
                i=0
                loss_vals=[]
                while i<closest[ng].shape[0]:
                    optimizer.zero_grad()
                    xs=closest[ng][i:i+batch_size].to(device)
                    loss=torch.mean(self.decoders[ng].negative_ELBO(xs))
                    loss.backward()
                    optimizer.step()
                    i+=batch_size
                    loss_vals.append(loss.item())
                mean_loss=np.mean(loss_vals)
                if mean_loss<best_loss:
                    best_loss=mean_loss
                    best_states=copy.deepcopy(self.decoders[ng].state_dict())
                if (ep+1)%100==0:
                    print('Init generator:',ng,'step:',ep+1,'loss:',mean_loss,'best loss:',best_loss)
            self.decoders[ng].load_state_dict(best_states)
        optimizer=torch.optim.Adam(self.parameters(),lr=learning_rate)
        print('SEEDING COMPLETED')

    def test_step(self,test_data_loader):
        with torch.no_grad():
            loss_sum=0
            weight_sum=torch.zeros(len(self.decoders),dtype=torch.float,device=device)
            i=0
            for xs,_ in test_data_loader:
                i+=1
                xs=xs.to(device)
                losses=[]
                log_decoder_weights=self.get_log_decoder_weights()
                for ng in range(len(self.decoders)):
                    loss=self.decoders[ng].negative_ELBO(xs)
                    loss-=log_decoder_weights[ng]
                    losses.append(loss.squeeze())
                losses=torch.stack(losses,-1)
                weights=-torch.clone(losses)
                weights=weights-torch.max(weights,-1,keepdim=True)[0]
                weights=torch.maximum(weights,torch.tensor(-25,dtype=torch.float,device=device))
                weights_sum=torch.log(torch.sum(torch.exp(weights),-1,keepdim=True))
                weights=weights-weights_sum
                weights=torch.exp(weights)
                weight_sum+=torch.sum(weights,0)
                loss_sum+=torch.sum(losses*weights)
            print((weight_sum/weight_sum.sum()).detach().cpu().numpy())
        return loss_sum.item()/len(test_data_loader.dataset)


    def sample(self,n):
        log_decoder_weights=self.get_log_decoder_weights()
        weights=torch.exp(log_decoder_weights).detach().cpu().numpy()
        generator=np.random.choice(len(self.decoders),size=n,p=weights/weights.sum())
        samples=[]
        for ng in range(len(self.decoders)):
            my_inds=generator==ng
            n_samples=np.sum(1*my_inds)
            if n_samples==0:
                continue
            samples.append(self.sample_gen(n_samples,ng))
        samples=torch.cat(samples,0)
        perm=torch.randperm(n)
        return samples[perm]

    def sample_gen(self,n,ng):
        samples=self.decoders[ng].sample(n)
        return samples

    def save_checkpoint(self,path,optimizer=None,epoch=None):
        checkpoint={}
        checkpoint["model_state_dict"]=self.state_dict()
        if not optimizer is None:
            checkpoint["optimizer_state_dict"]=optimizer.state_dict()
        if not epoch is None:
            checkpoint["epoch"]=epoch
        torch.save(checkpoint,path)

    def load_checkpoint(self,path,strict=True):
        if device=='cpu':
            checkpoint=torch.load(path,map_location=device)
        else:
            checkpoint=torch.load(path)
        if not "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint,strict=True)
            return None,None
        self.load_state_dict(checkpoint["model_state_dict"],strict=True)
        opt_state_dict=None
        if "optimizer_state_dict" in checkpoint:
            opt_state_dict=checkpoint["optimizer_state_dict"]
        epoch=None
        if "epoch" in checkpoint:
            epoch=checkpoint["epoch"]
        return opt_state_dict,epoch


