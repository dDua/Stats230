import nympy as np
import torch
from  hamiltonian import hamilton
from torch.autograd import Variable

class Sampler:

    def __init__(self,sample_size, position_dim, step_size=0.05,num_steps_in_leap=20,acceptance_thr=None):

        self.step_size = step_size
        self.init_velocity = None
        self.half_step = 0.5*self.step_size
        self.sample_size = sample_size
        self.position_dim = position_dim
        self.num_steps_in_leap = num_steps_in_leap
        self.acceptance_thr = acceptance_thr
        self.hamiltonian_dynamics = hamilton()

        self.init_velocity = np.random.multivariate_normal(np.zeros(self.position_dim),\
                                                    np.eye(self.position_dim,self.position_dim))
        
        self.gradient = torch.ones(2 * self.position_dim)
