# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:13:45 2020

@author: Shujian Yu, Xi Yu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    """calculate gram matrix for variables x
        Args:
        x: random variable with two dimensional (N,d).
        sigma: kernel size of x (Gaussain kernel)
    Returns:
        Gram matrix (N,N)
    """
    dist= pairwise_distances(x.flatten(-1))
    return torch.exp(-dist /sigma)

def renyi_entropy(x,sigma,alpha=1.001):
    
    """calculate entropy for single variables x (Eq.(9) in paper)
        Args:
        x: random variable with two dimensional (N,d).
        sigma: kernel size of x (Gaussain kernel)
        alpha:  alpha value of renyi entropy
    Returns:
        renyi alpha entropy of x. 
    """
    
    k = calculate_gram_mat(x,sigma)
    k = k/torch.trace(k) 
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x,y,s_x,s_y,alpha=1.001):
    
    """calculate joint entropy for random variable x and y (Eq.(10) in paper)
        Args:
        x: random variable with two dimensional (N,d).
        y: random variable with two dimensional (N,d).
        s_x: kernel size of x
        s_y: kernel size of y
        alpha:  alpha value of renyi entropy
    Returns:
        joint entropy of x and y. 
    """
    
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy

def calculate_MI(x,y,s_x,s_y,normalize=True):
    
    """calculate Mutual information between random variables x and y

    Args:
        x: random variable with two dimensional (N,d).
        y: random variable with two dimensional (N,d).
        s_x: kernel size of x
        s_y: kernel size of y
        normalize: bool True or False, noramlize value between (0,1)
    Returns:
        Mutual information between x and y (scale)

    """
    Hx = renyi_entropy(x,sigma=s_x)
    Hy = renyi_entropy(y,sigma=s_y)
    Hxy = joint_entropy(x,y,s_x,s_y)
    if normalize:
        Ixy = Hx+Hy-Hxy
        Ixy = Ixy/(torch.max(Hx,Hy)+1e-16)
    else:
        Ixy = Hx+Hy-Hxy
    return Ixy

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)


def cal_HSIC(x, y, s_x, s_y):
    
    """ calculate HSIC from https://github.com/danielgreenfeld3/XIC"""
    m,_ = x.shape #batch size
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC

def loss_fn(inputs, outputs, targets, name):
    """ loss function for different method"""
    inputs_2d = inputs.view(inputs.shape[0], -1) #input space
    error = F.softmax(outputs,dim=1) - F.one_hot(targets,10) #error space

    if name == 'HSIC':
        loss = cal_HSIC(inputs_2d, error, s_x=1, s_y=1)
    if name == 'MBD':
        loss = calculate_MI(inputs_2d,error,s_x=1,s_y=20)

class MEELoss(nn.Module):
    def __init__(self, sigma=1):
        super().__init__()
        self.sigma = sigma
    def forward(self, target, pred):
        error = pred - target
        loss = renyi_entropy(error.flatten(-1),sigma=self.sigma)
        return loss 

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduce=False)
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

def gaussiankernel(x, sigma):
    return torch.exp(-(x*x) / float( 2*(sigma**2) ) )

class MCCLoss(nn.Module):
    def __init__(self, sigma = 0.05*math.sqrt(2)):
        super(MCCLoss, self).__init__()
        self.sigma = sigma
    
    def forward(self, target, prev):
        err = (prev - target).squeeze()
        loss = gaussiankernel(err, self.sigma).mean()
        return loss

class MEELoss2(nn.Module):
    def __init__(self, sigma = 0.05*math.sqrt(2)):
        super(MEELoss2, self).__init__()
        self.sigma = sigma
    
    def forward(self, target, prev):
        err = (prev - target).squeeze()
        # loss = torch.zeros(1, requires_grad=True).cuda()
        # for e1 in err:
        #     for e2 in err:
        #         loss = loss + gaussiankernel(e1-e2, self.sigma)
        # loss / len(err)**2
        err_matrix = err.view(1,-1).T - err
        loss = gaussiankernel(err_matrix, sigma=self.sigma).mean()
        return loss

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

if __name__=='__main__':
    a = torch.tensor([1,2,3], dtype=torch.float32)
    b = torch.zeros_like(a)
    print(MCCLoss()(a,b))
    print(MEELoss2()(a,b))