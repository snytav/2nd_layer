
import torch
import numpy as np
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian
from torch.autograd import grad


def reshape_to_global(nx,ny,a):
    l = len(a.shape)
    if l == 1:
        a = a.reshape(nx,ny)
        # a = a.repeat(nx, ny, 1)
    if l == 2:
        a = a.reshape(nx, ny)
        # a = a.repeat(nx, ny, 1)
    if l == 3:
        b = a.reshape(1,a.shape[0],a.shape[1],a.shape[2])
        a = b.reshape(nx,ny,a.shape[1],a.shape[2])
    if l == 4:
        b = a.reshape(1, a.shape[0], a.shape[1], a.shape[2], a.shape[3])
        a = b.reshape(nx, ny, a.shape[2], a.shape[3])
    return a

def loss_function(x, y ,pde ,psy_trial ,f):
    loss_sum = 0.


    net_out_all               = []
    net_out_w_all             = []
    net_out_jacobian_all      = []
    net_out_hessian_all       = []
    psy_t_jacobian_all        = []
    psy_t_hessian_all         = []
    gradient_of_trial_d2x_all = []
    gradient_of_trial_d2y_all = []
    func_all                  = []
    func_t_all                = []
    err_sqr_all               = []
    psy_t_all                 = []



    for xi in x:
        for yi in y:
            input_point = torch.Tensor([xi, yi]).double()
            input_point.requires_grad_()

            net_out = pde.forward(input_point)
            net_out_all.append(net_out)
            net_out_w = grad(outputs=net_out, inputs=pde.fc1.weight, grad_outputs=torch.ones_like(net_out),
                             retain_graph=True ,create_graph=True)
            net_out_w_all.append(net_out_w[0])


            net_out_jacobian = jacobian(pde.forward ,input_point ,create_graph=True)
            net_out_jacobian_all.append(net_out_jacobian)
            # jac1  = get_jacobian(pde.forward,input_point,2)
            net_out_hessian = hessian(pde.forward ,input_point ,create_graph=True)
            net_out_hessian_all.append(net_out_hessian)
            psy_t = psy_trial(input_point, net_out)
            psy_t_all.append(psy_t)


            inputs = (input_point, net_out)
            psy_t_jacobian = jacobian(psy_trial, inputs ,create_graph=True)[0]
            psy_t_jacobian_all.append(psy_t_jacobian)
            psy_t_hessian  = hessian(psy_trial ,inputs ,create_graph=True)

            psy_t_hessian = psy_t_hessian[0][0]
            psy_t_hessian_all.append(psy_t_hessian)
            # acobian(jacobian(psy_trial))(input_point, net_out

            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2x_all.append(gradient_of_trial_d2x)

            gradient_of_trial_d2y = psy_t_hessian[1][1]
            gradient_of_trial_d2y_all.append(gradient_of_trial_d2y)

            # D_gradient_of_trial_d2x_D_W0 = grad(outputs=gradient_of_trial_d2x, inputs=pde.fc1.weight, grad_outputs=torch.ones_like(gradient_of_trial_d2x), retain_graph=True)
            # D_gradient_of_trial_d2y_D_W0 = grad(outputs=gradient_of_trial_d2y, inputs=pde.fc1.weight, grad_outputs=torch.ones_like(gradient_of_trial_d2y), retain_graph=True)
            # D_func_D_W0 = grad(outputs=func,iputs=pde.fc1.weight,grad_outputs=torch.ones_like(func))
            func = f(input_point)
            func_all.append(func)
            func_t = torch.Tensor([func])
            func_t_all.append(func_t)
            func_t.requires_grad_()

            err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func_t) ** 2
            # D_err_sqr_D_W0 = 2*((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)*(
            #                     (D_gradient_of_trial_d2x_D_W0 + D_gradient_of_trial_d2y_D_W0) -D_func_D_W0
            #                     )
            err_sqr_all.append(err_sqr)

            loss_sum += err_sqr
            qq = 0

    nx = x.shape[0]
    ny = y.shape[0]
    net_out_all = torch.tensor(net_out_all)
    net_out_all = reshape_to_global(nx,ny,net_out_all)
    net_out_w_all = torch.stack(net_out_w_all, dim=0)
    net_out_w_all = reshape_to_global(nx,ny,net_out_w_all)
    net_out_jacobian_all = torch.stack(net_out_jacobian_all, dim=0)
    net_out_jacobian_all = reshape_to_global(nx, ny, net_out_jacobian_all)
    net_out_hessian_all = torch.stack(net_out_hessian_all,dim=0)
    net_out_hessian_all = reshape_to_global(nx, ny, net_out_hessian_all)
    psy_t_jacobian_all = torch.stack(psy_t_jacobian_all,dim=0)
    net_out_hessian_all = reshape_to_global(nx, ny, net_out_hessian_all)
    psy_t_hessian_all  = torch.stack(psy_t_hessian_all,dim=0)
    psy_t_hessian_all = reshape_to_global(nx, ny,  psy_t_hessian_all)
    gradient_of_trial_d2x_all = torch.stack(gradient_of_trial_d2x_all,dim=0)
    gradient_of_trial_d2x_all = reshape_to_global(nx, ny, gradient_of_trial_d2x_all)
    gradient_of_trial_d2y_all = torch.stack(gradient_of_trial_d2y_all,dim=0)
    gradient_of_trial_d2y_all = reshape_to_global(nx, ny,  gradient_of_trial_d2y_all)
    func_all = torch.tensor(func_all)
    func_all = reshape_to_global(nx,ny,func_all)
    func_t_all = torch.stack(func_t_all,dim=0)
    func_t_all = reshape_to_global(nx,ny,func_t_all)
    err_sqr_all = torch.stack(err_sqr_all,dim=0)
    err_sqr_all = reshape_to_global(nx,ny,err_sqr_all)
    psy_t_all = torch.stack(psy_t_all,dim=0)
    psy_t_all = reshape_to_global(nx,ny,psy_t_all)

    return loss_sum
