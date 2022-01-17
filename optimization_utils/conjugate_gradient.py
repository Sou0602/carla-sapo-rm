import torch

from torch_utils.torch_utils import get_device

def cg_solver(Avp_fun, b, max_iter=10):
    '''
    Finds an approximate solution to a set of linear equations Ax = b

    Parameters
    ----------
    Avp_fun : callable
        a function that right multiplies a matrix A by a vector

    b : torch.FloatTensor
        the right hand term in the set of linear equations Ax = b

    max_iter : int
        the maximum number of iterations (default is 10)

    Returns
    -------
    x : torch.FloatTensor
        the approximate solution to the system of equations defined by Avp_fun
        and b
    '''

    device = get_device()
    x = torch.zeros_like(b).to(device)
#     r = b - Avp_fun(x, retain_graph=True)
#     p = r.clone()
    r = b.clone()
    p = b.clone()
    rtr = []
    ptavp = []
    alp = []
    brt = []
    alpha_avp = []
    
    for i in range(max_iter):
        Avp = Avp_fun(p, retain_graph=True)

        alpha = torch.matmul(r, r) / torch.matmul(p, Avp)
        rtr.append(torch.matmul(r, r))
        ptavp.append(torch.matmul(p, Avp))
        alp.append(alpha)
        if (alpha != alpha):
            pass
#             print('r',r.shape,r)
#             print('p',p.shape,p)
#             print('Avg',Avp.shape,Avp)
#             print('r times r:',rtr)
#             print('p times Avp:',ptavp)
#             print('alpha:',alp)
#             print('beta',brt)
        if torch.matmul(p,Avp) <= 0.0001:
            return x
        
        x += alpha * p

        if i == max_iter - 1:
            return x
        r_new = r - alpha * Avp

        beta = torch.matmul(r_new, r_new) / torch.matmul(r, r)
        brt.append(beta)
        r = r_new
        p = r + beta * p

        
#          device = get_device()
#     x = torch.zeros_like(b).to(device)
#     r = b.clone()
#     p = b.clone()
#     rtr = []
#     ptavp = []
#     alp = []
#     for i in range(max_iter):
#         Avp = Avp_fun(p, retain_graph=True)

#         alpha = torch.matmul(r, r) / torch.matmul(p, Avp)
#         rtr.append(torch.matmul(r, r))
#         ptavp.append(torch.matmul(p, Avp))
#         alp.append(alpha)
#         if (alpha != alpha):
#             print('r',r.shape,r)
#             print('p',p.shape,p)
#             print('Avg',Avp.shape,Avp)
#             print('r times r:',rtr)
#             print('p times Avp:',ptavp)
#             print('alpha:',alp)
#             return ValueError
#         x += alpha * p

#         if i == max_iter - 1:
#             return x

#         r_new = r - alpha * Avp
#         beta = torch.matmul(r_new, r_new) / torch.matmul(r, r)
#         r = r_new
#         p = r + beta * p