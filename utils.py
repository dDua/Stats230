import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn

class Hamiltonian():
    
    def __init__(self, decoder, position_dim, step_size, num_steps_in_leap, num_samples):
        self.decoder = decoder
        self.position_dim = position_dim
        self.step_size = step_size
        self.half_step = 0.5*step_size
        self.num_steps_in_leap = num_steps_in_leap
        self.num_samples = num_samples
        self.loss_ce = nn.BCELoss()
        self.gradient = torch.ones(1)
        self.sigmoid = nn.Sigmoid()
        
    def energy(self, z, x):
        if z.__class__.__name__=="ndarray":
            z = Variable(torch.FloatTensor(z))
        probs = self.sigmoid(self.decoder(z.unsqueeze(0)))
        log_posteriors = self.loss_ce(probs, x.view(-1,784))
        prior = (0.5 * torch.sum(z**2))
        return log_posteriors + prior

    def kinetic(self, v):
        return 0.5* torch.dot(v,v)
    
    def hamiltonian(self, z, x, v, potential=None, gpu=False):
        ke = self.kinetic(v) 
        if gpu:
            ke = ke.cuda()
            z = z.cuda()
        if potential:
            return potential + ke
        else:
            return self.energy(z, x) + ke

    def leap_frog_step(self, phase_tensor, x, gpu=False):
        on_going_phase = phase_tensor
        orig_hamitlonian = self.hamiltonian(on_going_phase[:self.position_dim], x, on_going_phase[self.position_dim:])
        orig_hamitlonian.backward(self.gradient)
        phase_grad = on_going_phase.grad
    
        for step in range(0, self.num_steps_in_leap):
            # print "step=",step
            tmp_array = torch.cat((on_going_phase[:self.position_dim] + self.step_size * phase_grad[self.position_dim:],
                                    on_going_phase[self.position_dim:] - self.half_step * phase_grad[:self.position_dim]), 0)
            zz = Variable(torch.FloatTensor(tmp_array[:self.position_dim].data), requires_grad=True)
    
            potential = self.energy(zz, x)
            potential.backward(self.gradient)
            tmp_array[self.position_dim:] = tmp_array[self.position_dim:] - self.half_step * zz.grad
    
            velocity = Variable(tmp_array[self.position_dim:].data, requires_grad=True)
            on_goingrig_hamitlonian = self.hamiltonian(tmp_array[:self.position_dim],x, \
                                            velocity, potential=potential.data[0])
    
            "Prepare Hamiltonian for next iteration"
            on_goingrig_hamitlonian.backward(self.gradient)
            phase_grad = torch.cat((zz.grad, velocity.grad), 0)
    
        # current_hamiltonian = hamiltonian_measure(tmp_array[:pos_dim], tmp_array[pos_dim:], potential_function, weight_mat, bias_array, pot_val= potential.data[0])
        p_accept = min(1.0, np.exp(orig_hamitlonian.data[0] - on_goingrig_hamitlonian.data[0]))
    
        acceptance_thr = np.random.uniform()
        
        if p_accept > acceptance_thr:
            termination_val= tmp_array[:self.position_dim]
        else:
            termination_val= phase_tensor[:self.position_dim]
    
    
        return termination_val.data.numpy()
    
    def get_hmc_sample(self, sample_array, original_input, gpu=False):
        #bad_decline_cntr = 0
        inverse_term = 0
        other_term = 0
        energy_loss = 0
        for it in range(0, self.num_samples):
            init_velocity = np.random.multivariate_normal(np.zeros(self.position_dim),\
                                            np.eye(self.position_dim, self.position_dim))
            tmp_tensor = np.concatenate((sample_array[-1],init_velocity),0)
    
            phase_tensor= Variable(torch.FloatTensor(tmp_tensor),requires_grad=True)
    
            new_sample = self.leap_frog_step(phase_tensor, original_input, gpu=gpu)
    
#             if not(np.array_equal(new_sample,sample_array[-1])):
            sample_array = np.vstack((sample_array, new_sample))
            
#             energy_diff = self.energy(new_sample, original_input)
#             inverse_term += np.mean(new_sample)
#             other_term -= np.mean(new_sample)
#             energy_loss += np.mean(1 / energy_diff) - np.mean(energy_diff)
            
#        sampler_loss = inverse_term + other_term + 0.1 * energy_loss
        return torch.FloatTensor(sample_array)
    
    