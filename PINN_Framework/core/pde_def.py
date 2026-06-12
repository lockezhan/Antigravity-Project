import deepxde as dde
import numpy as np

def get_ns_equation_data(scale_factor="small"):
    nu = 0.05
    lam = 10.0 - np.sqrt(100.0 + 4 * (np.pi ** 2))
    
    def pde(x, y):
        u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        du_y = dde.grad.jacobian(y, x, i=0, j=1)
        dv_x = dde.grad.jacobian(y, x, i=1, j=0)
        dv_y = dde.grad.jacobian(y, x, i=1, j=1)
        dp_x = dde.grad.jacobian(y, x, i=2, j=0)
        dp_y = dde.grad.jacobian(y, x, i=2, j=1)
        
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
        
        eq_u = u * du_x + v * du_y + dp_x - nu * (du_xx + du_yy)
        eq_v = u * dv_x + v * dv_y + dp_y - nu * (dv_xx + dv_yy)
        eq_mass = du_x + dv_y
        
        return [eq_u, eq_v, eq_mass]
        
    def u_func(x):
        return 1 - np.exp(lam * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])

    def v_func(x):
        return lam / (2 * np.pi) * np.exp(lam * x[:, 0:1]) * np.sin(2 * np.pi * x[:, 1:2])

    def p_func(x):
        return 0.5 * (1 - np.exp(2 * lam * x[:, 0:1]))

    geom = dde.geometry.Rectangle([-0.5, -0.5], [1.0, 1.5])
    
    if scale_factor == "small":
        num_domain, num_boundary, num_test = 2000, 200, 2000
    elif scale_factor == "large":
        num_domain, num_boundary, num_test = 80000, 8000, 10000
    elif scale_factor == "extreme":
        num_domain, num_boundary, num_test = 1000000, 100000, 200000
    else:
        num_domain, num_boundary, num_test = 2000, 200, 2000
    
    return geom, pde, (u_func, v_func, p_func), num_domain, num_boundary, num_test
