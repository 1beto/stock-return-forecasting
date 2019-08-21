"""
This is a simple program to generate random process from simple stochastic
differential equations. I want to create a large amount of data to simulate from
any sde that the python package sdeint could generate.
"""

import sdeint
import numpy as np
import h5py
import os
import global_var
# %%
N = global_var.N
def main():

    alpha = np.linspace(0.001,0.1,N)
    kappa = np.linspace(2,0.02,N)
    avgdev = np.linspace(0.1,0.001,N)
    rho = np.linspace(-0.95,-0.4,N)
    tspan = np.linspace(0,10,1001)
    xo = np.array([0,0.05])

    foda_men = h5py.File('stochastic_heston.h5','a')
    foda_men.require_dataset("Heston",((20*N**4),1001,2),dtype='float32')

    i=1
    for al in alpha:
        for k in kappa:
            for m in avgdev:
                for r in rho:
                    def f(x,t):
                        return np.array([0,-al*(x[1]-m)])

                    def G(x,t):
                        return np.array([[x[1],0],[k*r,k*np.sqrt(1-r*r)]])
                    for j in range(20):
                        foda_men["Heston"][i,:,:] = sdeint.itoint(f, G, xo, tspan)

                        i += 1
                        if (i%100 == 0):
                            print(i)
    print(i)
main()
