#!/usr/bin/env python

from dolfin import *
import pickle
import math
from scipy.sparse import csr_matrix
from utils.general import *
import numpy as np

device = get_device()

############################################################
# INTERNAL CLASSES AND FUNCTIONS
############################################################


class K(UserExpression):
    def set_parameters(self, k_0, k_1, k_2,x_c,y_c,radius):
        self.k_0, self.k_1, self.k_2 = k_0, k_1, k_2
        self.radius=radius
        self.x_c=x_c
        self.y_c=y_c
    def eval(self, value, x):
        tol = 1E-14
        if pow(x[0]-self.x_c[0],2) + pow(x[1]-self.y_c[0],2) <= pow(self.radius[0],2) + tol and pow(x[0]-self.x_c[1],2) + pow(x[1]-self.y_c[1],2) <= pow(self.radius[1],2) + tol:
            value[0] = self.k_0 
        elif pow(x[0]-self.x_c[0],2) + pow(x[1]-self.y_c[0],2) <= pow(self.radius[0],2) + tol:
            value[0] = self.k_0
        elif pow(x[0]-self.x_c[1],2) + pow(x[1]-self.y_c[1],2) <= pow(self.radius[1],2) + tol:
            value[0] = self.k_2 
        else:
            value[0] = self.k_1

# Create mesh to generate data
def mesh_g(Nx,Ny):
    m = RectangleMesh(Point(0,0), Point(1,1), Nx, Ny)
    x = m.coordinates()
    x[:] = (x-0.5)*2.
    x[:] = 0.5 * (np.cos(np.pi *(x - 1.) / 2.) + 1.)
    return m


# Define boundary data 
class MyUserExpression(UserExpression):
    def set_num(self,num,N):
        self.num = num
        self.N=N
    def eval(self,value,x):
        if math.atan2(x[1]-0.5,x[0]-0.5) > 0 and math.atan2(x[1]-0.5,x[0]-0.5) < 2*math.pi/(self.N+1):
            value[0] = (self.N+1)/2        
        elif math.atan2(x[1]-0.5,x[0]-0.5) > 2*self.num*math.pi/(self.N+1) and math.atan2(x[1]-0.5,x[0]-0.5) < 2*(self.num+1)*math.pi/(self.N+1) and self.num < (self.N+1)/2:
            value[0] = -(self.N+1)/2
        elif math.atan2(x[1]-0.5,x[0]-0.5) < -2*(self.num-(self.N+1)/2)*math.pi/(self.N+1) and math.atan2(x[1]-0.5,x[0]-0.5) > -2*(self.num+1-(self.N+1)/2)*math.pi/(self.N+1) and self.num >= (self.N+1)/2:
            value[0] = -(self.N+1)/2
        else:
            value[0] = 0
    def value_shape(self):
        return()

def gen_mat(N):
    mat=np.zeros((N,N+1))
    length=N+1
    i=0
    while length>=1:
        if length%2!=0 and length !=1:
            raise ValueError('N+1 must be a power of 2!')
        segments=(N+1)//length
        for s in range(segments//2):
            mat[i,2*s*length:(2*s+1)*length]=1
            mat[i,(2*s+1)*length:(2*s+2)*length]=-1
            i+=1
        length=length//2
    return mat

# Define boundary data 
class OtherBoundaryValues(UserExpression):
    def set_num(self,num,N,mat):
        self.num = num
        self.N=N
        self.mat=mat
    def eval(self,value,x):
        ang=(np.angle(x[0]-.5+1j*(x[1]-.5))+math.pi/4)%(2*math.pi)
        seg=int(np.floor((self.N+1)*ang/(2*math.pi)))
        value[0]=self.mat[self.num-1,seg]*(self.N+1)/np.sum(np.abs(self.mat[self.num-1,:]))
    def value_shape(self):
        return()

def on_boundary(x):
    tol=1e-10
    out=min(abs(x[0]),abs(1-x[0]))<tol or min(abs(x[1]),abs(1-x[1]))<tol
    return out

def create_ground_truth_continuous(x_c= [0.25 + 0.5, 0.5],y_c= [0.2 + 0.5, 0.5],val= [1.55, 1.55],radius = [0.15, 0.12]):
    # Initialize conductivity
    k_real = K(degree=0)
    k_real.set_parameters(val[0], 1, val[1],x_c,y_c,radius)
    return k_real

def create_ground_truth_from_image(img):
    img_vec=np.reshape(img,-1)
    mesh_r_c = UnitSquareMesh(128,128,diagonal="left")
    C = FunctionSpace(mesh_r_c, 'DG', 0)
    k_real= Function(C)
    with open('utils/P.pickle','rb') as f:
        P_sparse=pickle.load(f)
    k_real.vector()[:]=P_sparse.dot(img_vec)*1.6+0.2 
    return k_real

############################################################
# RECONSTRUCTION CLASS
############################################################
# generates meshes, observation function spaces and so on
# provides the following functions for a specific ground-truth conductivity:
#       - data_fidelity: Takes a conductivity Ek and computes the distance of ND(ground_truth) to ND(Ek)
#       - data_fidelity_derivative: Takes a conductivity Ek and computes the derivative of data_fidelity
#
# BOTH FUNCTIONS ARE BASED ON NUMPY/FENICS/DOLFIN AND DO NOT SUPPORT AUTOGRAD!!!

class CalderonReconstruction():
    def __init__(self,ground_truth_conductivity=create_ground_truth_continuous(),N=15,noise_level=0.):
        #####################################################
        # PARAMETERS
        #####################################################
        generation_mesh = Mesh('utils/generation_mesh.xml')
        reconstruction_mesh = Mesh('utils/reconstruction_mesh.xml')
        num_refinements=2
        for _ in range(num_refinements):
            generation_mesh=refine(generation_mesh)
            reconstruction_mesh=refine(reconstruction_mesh)

        k_real=ground_truth_conductivity
        mesh = generation_mesh 

        P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        R = FiniteElement("Real", mesh.ufl_cell(), 0)
        W = FunctionSpace(mesh, P1 * R)
        C_ref = FunctionSpace(mesh, 'DG', 0)

        # P matrix from pixel to mesh (appling this matrix to a data discretized in pixel as an image, we obtain its representation on the previous mesh)
        with open('utils/P.pickle','rb') as f:
            P_sparse=pickle.load(f)
        # Q matrix from mesh to pixel (appling this matrix to a data define on the previous mesh, we obtain its pixel representation)
        Q_sparse=.5*P_sparse.transpose()

        # Define variational problem
        (u, c) = TrialFunction(W)
        (v, d) = TestFunctions(W)

        a_real = (interpolate(k_real,C_ref) *inner(grad(u), grad(v)) + c*v + u*d)*dx(mesh)
        mat=gen_mat(N)
        boundary_values=lambda: OtherBoundaryValues(degree=1)

        #####################################################
        # SIMULATE OBSERVATION
        #####################################################
        # ND map associated to the real conductivity
        ND_real = []

        for n in range(N):
            print(n)
            g_1 = boundary_values()
            g_1.set_num(n+1,N,mat)
            L_1 = g_1*v*ds(mesh)
            # Compute solution
            set_log_active(False)
            w_real_1 = Function(W)
            solve(a_real == L_1, w_real_1, solver_parameters={'linear_solver': 'umfpack'})
            (u_real_1, c_real_1) = w_real_1.split()  
            if n == 0:
                one_fun= Function(W)
                (one_fun, c_fun) = one_fun.split()
                one_fun.vector()[:] = 1
                norm = assemble(one_fun*ds)
            # subtract the mean of the solution on the boundary to the solution, in order to find the solution with zero boundary mean 
            boundary_mean = assemble(u_real_1*ds(mesh))/norm
            u_real_1.vector()[:] = u_real_1.vector()[:]-boundary_mean
            ND_real.append(u_real_1)
        ND_real=ND_real


        #####################################################
        # INITIALIZE RECONSTRUCTION FUNCTIONS
        #####################################################

        mesh_r_c = UnitSquareMesh(128,128,diagonal="left")
        Ntriang = mesh_r_c.num_cells()
        mesh_r_u = reconstruction_mesh#mesh_r_c


        # Build function space with Lagrange multiplier
        P1 = FiniteElement("Lagrange", mesh_r_u.ufl_cell(), 1)
        R = FiniteElement("Real", mesh_r_u.ufl_cell(), 0)
        W = FunctionSpace(mesh_r_u, P1 * R)
        C = FunctionSpace(mesh_r_c, 'DG', 0)
        C_ref = FunctionSpace(mesh_r_u, 'DG', 0) 
        F = FunctionSpace(mesh_r_u, 'CG', 1) 

        
        # ground_truth as 128x128 image
        ground_truth=(project(k_real,C).vector()[:]-0.2)/1.6
        self.ground_truth=np.reshape(Q_sparse.dot(ground_truth),(128,128))

        # identify boundary
        BC=DirichletBC(F,1,AutoSubDomain(on_boundary))
        bd=Function(F)
        BC.apply(bd.vector())

        # project real measurement onto the correct mesh
        ND_real_pr = []
        ND_real_pr_bd = []
        for n in range(N):
            ND_real[n].set_allow_extrapolation(True)
            nd_real_pr=interpolate(ND_real[n],F)
            nd_real_pr_bd=(nd_real_pr.vector()[bd.vector()==1])[:]
            nd_real_pr.vector()[bd.vector()==0][:]=0.
            nd_real_pr.vector()[:]=nd_real_pr.vector()[:]+noise_level*np.random.normal(size=nd_real_pr.vector()[:].shape)
            ND_real_pr.append(nd_real_pr)

        # Define variational problem
        kappa_prec = Function(C)
        (u, c) = TrialFunction(W)
        (v, d) = TestFunctions(W)

        def data_fidelity(Ek):
            Ek_prec=np.reshape(Ek,(128**2))
            kappa_prec.vector()[:] = P_sparse.dot(Ek_prec)*1.6+0.2    
            a = (interpolate(kappa_prec,C_ref)*inner(grad(u), grad(v)) + c*v + u*d)*dx
            w_1 = Function(W)
            new_dist = 0
            for n in range(N):
                g_1 = boundary_values()
                g_1.set_num(n+1,N,mat)
                L_1 = g_1*v*ds(mesh_r_u)
                # Compute solution
                solve(a == L_1, w_1, solver_parameters={'linear_solver': 'umfpack'})
                (u_1, c_1) = w_1.split()   
                if n == 0:
                    one_fun= Function(W)
                    (one_fun, c_fun) = one_fun.split()
                    one_fun.vector()[:] = 1
                    norm = assemble(one_fun*ds)
                # subtract the mean of the solution on the boundary to the solution, in order to find the solution with zero boundary mean
                boundary_mean = assemble(u_1*ds(mesh_r_u))/norm
                u_1.vector()[:] = u_1.vector()[:]-boundary_mean
                new_dist += assemble(((ND_real_pr[n]-project(u_1,F))**2)*ds)
            return new_dist
        self.data_fidelity=data_fidelity

        def data_fidelity_derivative(Ek,forward_sensitivities=None):
            Ek_prec=np.reshape(Ek,(128**2))
            kappa_prec.vector()[:] = P_sparse.dot(Ek_prec)*1.6+0.2    
            a = (interpolate(kappa_prec,C_ref)*inner(grad(u), grad(v)) + c*v + u*d)*dx

            w_1 = Function(W)
            w_aux = []
            u_aux = []
            z_1 = Function(W)
            f_1 = Function(F)
            obj=0
            for n in range(N):
                g_1 = boundary_values()
                g_1.set_num(n+1,N,mat)
                L_1 = g_1*v*ds(mesh_r_u)
                # Compute solution
                solve(a == L_1, w_1, solver_parameters={'linear_solver': 'umfpack'})
                (u_1, c_1) = w_1.split()   
                if n == 0:
                    one_fun= Function(W)
                    (one_fun, c_fun) = one_fun.split()
                    one_fun.vector()[:] = 1
                    norm = assemble(one_fun*ds)
                # subtract the mean of the solution on the boundary to the solution, in order to find the solution with zero boundary mean
                boundary_mean = assemble(u_1*ds(mesh_r_u))/norm
                u_1.vector()[:] = u_1.vector()[:]-boundary_mean
                obj += assemble(((ND_real_pr[n]-project(u_1,F))**2)*ds)

                f_1.vector()[:] = project(u_1,F).vector()[:] - ND_real_pr[n].vector()[:] 
                L_1 = f_1*v*ds
                solve(a == L_1, z_1, solver_parameters={'linear_solver': 'umfpack'})
                (z1, c1) = z_1.split()
                if n == 0:
                    one_fun= Function(W)
                    (one_fun, c_fun) = one_fun.split()
                    one_fun.vector()[:] = 1
                    norm = assemble(one_fun*ds(mesh_r_u))
                # subtract the mean of the solution on the boundary to the solution, in order to find the solution with zero boundary mean
                boundary_mean1 = assemble(z1*ds(mesh_r_u))/norm
                z1.vector()[:] = z1.vector()[:]-boundary_mean1
                w_aux.append(grad(project(z1,F)))           
                u_aux.append(grad(project(u_1,F)))
            if forward_sensitivities is None:
                forward_sensitivities=np.eye(128)    
            else:
                forward_sensitivities=np.reshape(forward_sensitivities,(128**2,-1))
            der=np.zeros(forward_sensitivities.shape[1])    
            dumm_fun = Function(C)   
            for q in range(forward_sensitivities.shape[1]):
                aux = 0
                dumm_fun.vector()[:] = P_sparse.dot(forward_sensitivities[:,q])*1.6  
                dumm_fun = project(dumm_fun,C)  
                           
                for n in range(N):                     
                    aux -= assemble( dumm_fun*inner(w_aux[n],u_aux[n]) * dx(mesh_r_u) )
                der[q] = aux
            return der,obj
        self.data_fidelity_derivative=data_fidelity_derivative

