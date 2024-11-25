import torch.nn as nn
import torch
# нейронная сеть для решения уравнения Пуассона
class PDE_2D_net(nn.Module):
    def __init__(self,N,nx,ny):
        super(PDE_2D_net,self).__init__()
        self.N  = N
        self.nx  = nx
        self.ny = ny
        fc1 = nn.Linear(2*self.nx*self.ny,self.N) # первый слой
        fc1.weight = nn.Parameter(fc1.weight.double())
        fc1.bias = nn.Parameter(torch.zeros(fc1.bias.shape).double())

        fc2 = nn.Linear(self.N, self.nx*self.ny) # второй слой
        fc2.weight = nn.Parameter(fc2.weight.double())
        fc2.bias = nn.Parameter(torch.zeros(fc2.bias.shape).double())
        self.fc1 = fc1

        self.fc2 = fc2

    def forward(self,x):
        x = x.reshape(self.nx*self.ny*2)
        y = self.fc1(x.double())
        y = torch.sigmoid(y)
        y = self.fc2(y.reshape(1, self.N))
        y = y.reshape(self.nx,self.ny)
        return y

def A(x):
    return (x[1] * torch.sin(np.pi * x[0]))

def psy_trial(x, net_out):
    return A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out


if __name__ == '__main__':
    pde2 = PDE_2D_net(3, 3, 3)
    from module_2nd_layer import x_space,y_space
    import numpy as np
    X,Y = np.meshgrid(x_space,y_space)
    nx = x_space.shape[0]
    ny = y_space.shape[0]
    X = X.reshape(nx,ny,1)
    Y = Y.reshape(nx,ny,1)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    XY = torch.cat((X,Y),dim=2)
    y = pde2(XY)
