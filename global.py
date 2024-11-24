import torch
from loss import loss_function
from module_2nd_layer import x_space, y_space,pde,psy_trial,f
import numpy as np


lf,details = loss_function(x_space, y_space,pde,psy_trial,f)

X,Y = np.meshgrid(x_space,y_space)
nx = x_space.shape[0]
ny = y_space.shape[0]
X = X.reshape(nx,ny,1)
Y = Y.reshape(nx,ny,1)
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)
XY = torch.cat((X,Y),dim=2)
func_global = torch.tensor([f(x) for x in X for X in XY]).reshape(nx,ny)
qq = 0
