import torch
import torch.nn as nn

class Multiply2D(nn.Module):
    def get_weights(self,nx,ny,w):
        W = w.reshape(1, 1, 3, 2)
        W = W.repeat(nx,ny,1,1)
        return W

    def __init__(self, nx,ny,w):
        super().__init__()
        self.nx = nx
        self.ny = ny
        W = self.get_weights(nx,ny,w)
        self.weight = nn.Parameter(W.double())


    def forward(self, x):
        res = torch.einsum('abij,abjk->abi', self.weight, x)
        return res

if __name__ == '__main__':
    nx = 3
    ny = 3
    from weights import w0
    fc1 = nn.Linear(2,nx)
    fc1.weight = nn.Parameter(torch.from_numpy(w0).T.double())
    fc1.bias   = nn.Parameter(torch.zeros(fc1.bias.shape).double())
    ml = Multiply2D(3,3,w0)
