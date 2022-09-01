#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:50:13 2022

@author: hannahhaider
email : hzhaider@ucsd.edu 

This script computes the reduced order model for a linear, advection equation. 
"""
# importing needed packages
import numpy as np
import matplotlib.pyplot as plt
import rom_operator_inference as opinf
from rom_operator_inference.pre import ddt
import scipy.linalg as la
import scipy.sparse as sparse
import pandas as pd 
import imageio
import os 
import matplotlib
#%% preamble for plotting 
font = {'family' : 'serif',
        'size'   : 18}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
#%%
# defining solution matrix indices and parameters
c = 10 # advection velocity (km/s)
n = 2**12 # degrees of freedom for the spatial discretization 
k = 2000 # time instances 
Ntest = 50 # testing parameter instances 
x = np.linspace(0, 1, n) # linearly spaced spatial domain 
t = np.linspace(0, 0.1, k) # linearly spaced time domain
dt = t[1] - t[0] # time step 

# storing solution matrix 
s = np.zeros((len(x), len(t))) # solution matrix of n x k 
s_cent = np.zeros((len(x), len(t))) # centered solution matrix 
#%%
# initial condition 
mu0 = 0.10 # initial mu 
s0 = (1 / np.sqrt(0.0002 * np.pi)) * np.exp( - (x - mu0)**2 / 0.0002) # Gaussian initial condition 
s[:, 0] = s0

# plotting initial condition 
fig, ax = plt.subplots(1, figsize = (5,5))
plt.plot(x, s[:, 0])
ax.set_xlabel("x - coordinate", fontsize = 10)
ax.set_ylabel("initial condition, solution s(x,0)", fontsize = 10)
filename = "gaussian_IC_ROM.png"
plt.savefig(filename, dpi = 'figure', bbox_inches = 'tight', pad_inches = 0.2)

# creates a traveling/propogated gaussian hump - full order model 
for jj in range(len(t) - 1):
    s[:, jj] = (1 / np.sqrt(0.0002 * np.pi)) * np.exp( - (x - c*(t[jj]) -mu0)**2 / 0.0002) 
    
    sref = (s[:,-1] - s[:, 0]) / k  # time averaged, reference state
    s_cent[:, jj] = s[:, jj] - sref # centered state 
    
# plots solution for increasing time step and creates a gif of propogated wave 
filenames = []
for jj in range(len(t) - 1): 
    fig1, ax1 = plt.subplots(1, figsize = (7,7))
    plt.plot(x, s_cent[:, jj])
    fig1.suptitle("Full Order Model Solution s(x,t) t = " + str(round(t[jj], 3))) 
    ax1.set_ylabel("Solution s(x,t)", fontsize = 18)
    ax1.set_xlabel("x-coordinate", fontsize = 18)
    plt.tight_layout()
    jj =  jj + 100 # increments time step to only plot every 100th time step 
    
    filename = f'{jj}.png' # saving each timestep to a different filename 
    filenames.append(filename)
    fig1.savefig(filename, dpi = 'figure', bbox_inches = 'tight', pad_inches = 0.2)
    plt.close()

# build gif for increasing time step 
with imageio.get_writer('FOM.gif', mode = 'I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
for filename in set(filenames):
    os.remove(filename)  

# could also use matrices and scipy 
# dx = x[1] - x[0]
# diags = c * np.array([1,-1] / (dx)) # forward differencing 
# A = sparse.diags(diags, [1,0], (n,n)) 
#%% plotting exact solution as p colormesh plot 
fig, ax = plt.subplots(figsize = (7,7))
pos = ax.pcolormesh(x, t, s_cent.T, shading = 'gourad', cmap = 'viridis')
cbar = fig.colorbar(pos, ax = ax, orientation = 'vertical')
cbar.ax.set_ylabel("exact solution s(x,t)")
ax.set_xlabel("x-coordinate", fontsize = 18)
ax.set_ylabel("time", fontsize = 18)
filename = "exact_sol_FOM.png"
plt.savefig(filename, dpi = 'figure', bbox_inches = 'tight', pad_inches = 0.2)
#%% beginning Reduced order modeling 

# splitting training and testing datasets 
split = int(.7*(len(t))) # splitting into 70, 30 
s_train = s_cent[:, :split]
s_test = s_cent[:, split:]
t_train = t[:split]
t_test = t[split:]

# in order to compute the ROM, we must first compute the singular 
# value decomposition and the relative cumulative energy
# using scipy to compute SVD
svdavls = la.svdvals(s_train)
cum_energy = np.cumsum(svdavls) / np.sum(svdavls) # using cumsum because we neglect any zero singualar values
threshold = .99 

# the rank (r dimension, number of POD modes) can be found by summing 
# the number of singular values in the s.v.index below the threshold
rank = sum([element < threshold for element in cum_energy]) # for element in cum_energy < threshold 
print(f"r = {rank}")
#%%
#plotting cumulative energy vs singular value index
fig2, ax2 = plt.subplots(1, figsize = (7,7))
plt.plot(np.arange(1, svdavls.size + 1), cum_energy, 'x-', c = "r")
plt.grid()
ax2.set_xlim(0, rank)
ax2.set_yticks(ticks = [0.4, 0.6, 0.8, 0.9, 0.95, 1])
ax2.set_xlabel("Singular Value Index", fontsize = 12) 
ax2.set_ylabel("Cumulative Energy", fontsize = 12)
plt.tight_layout()
filename = "svd_cum_energy.png"
plt.savefig(filename, dpi = 'figure', bbox_inches = 'tight', pad_inches = 0.2)
#%%
# using ROM operator inference from Wilcox 
# compute basis functions - POD (# = rank) from the state data
Vr = opinf.pre.pod_basis(s_train, rank)[0]  # [1] computes SVD values

# to get a sense of our solution, we plot the columns of Vr - dominant r left singular values of state, s
fig, ax = plt.subplots(figsize = (15, 10))
for j in range(Vr.shape[1]):
    if j %5 == 0: 
        ax.plot(x, Vr[:len(x),j], label = f"POD mode {j+1}")
ax.set_xlabel("x-coordinate", fontsize = 15)
ax.set_title("POD Modes", fontsize = 15)
plt.legend(loc = "upper right")

filename = "POD_modes.png"
plt.savefig(filename, dpi = 'figure', bbox_inches = 'tight', pad_inches = 0.2)

# getting projection error 
err = opinf.pre.projection_error(s_train, Vr)
print(f"projection error = {err}")
#%%    
# estimate the time derivatives of the state with finite differences 
sdot = opinf.pre.ddt(s_train, dt)

# define a reduced order model with quadratic manifold structure 
rom = opinf.ContinuousOpInfROM(modelform = "AH")

# define a reduced order model with linear projection structure 
rom_linear = opinf.ContinuousOpInfROM(modelform = "A") 

# fit the model (progression and regression)
# regularizer =  # tikhonov regularization 
rom.fit(Vr, s_train, sdot, regularizer = 1e5) #, P = 1e6)
rom_linear.fit(Vr, s_train, sdot, regularizer = 1e-2)

# simulate the learned model over the time domain 
S_ROM = rom.predict(s[:,0], t, method = "BDF")
S_ROM_linear = rom_linear.predict(s[:,0], t, method = "BDF")

#%% plotting ROM 
fig, ax = plt.subplots(nrows = 2, figsize = (5,10))
pos = ax[0].pcolormesh(x, t[:S_ROM.T.shape[0]], S_ROM.T, shading = 'gourad', cmap = 'viridis')
pos = ax[1].pcolormesh(x, t[:S_ROM_linear.T.shape[0]], S_ROM_linear.T, shading = 'gourad', cmap = 'viridis')

cbar = fig.colorbar(pos, ax = ax[0], orientation = 'vertical')
cbar = fig.colorbar(pos, ax = ax[1], orientation = 'vertical')
cbar.ax.set_ylabel("ROM solution s(x,t)")

ax[1].set_xlabel("x-coordinate", fontsize = 18)
ax[0].set_ylabel("time", fontsize = 18)
ax[1].set_ylabel("time", fontsize = 18)
fig.suptitle("Quadratic Manifold ROM (top) vs. Linear ROM (bottom)")

filename = "sol_ROM.png"
plt.savefig(filename, dpi = 'figure', bbox_inches = 'tight', pad_inches = 0.2)
#%% computing absolute and relative error between FOM and ROM
rel_err_quad = opinf.post.frobenius_error(s, S_ROM)[1] # quadratic ROM error 
rel_err_lin = opinf.post.frobenius_error(s, S_ROM_linear)[1] # linear ROM error 

# creating comparison plots 
fig, ax = plt.subplots(ncols = 5, sharey = True, figsize = (12, 3))
pos = ax[0].imshow(s.T , extent = [0, 1, t[0], t[-1]], vmax = 40, vmin = 0, aspect = "auto", origin = "lower")
ax[0].set_xlabel("x - coordinate")
ax[0].set_ylabel("time")
ax[0].set_title("Exact Solution")
cbar = fig.colorbar(pos, ax = ax[0])
cbar.ax.set_ylabel("s(x,t)", rotation = 90)

pos = ax[1].imshow(S_ROM.T , extent = [0, 1, t[0], t[-1]], vmax = 40, vmin = 0, aspect = "auto", origin = "lower")
ax[1].set_xlabel("x-coordinate")
ax[1].set_title("Quadratic Manifold ROM")
ax[1].plot(x, t_train[-1] * np.ones(len(x)), c = "white")
cbar = fig.colorbar(pos, ax = ax[1])
cbar.ax.set_ylabel("s(x,t)", rotation = 90)

pos = ax[2].imshow(100 * np.abs(S_ROM.T - s.T) / np.abs(s.T) , extent = [0, 1, t[0], t[-1]], cmap = 'plasma', aspect = "auto", origin = "lower")
ax[2].set_xlabel("x-coordinate")
ax[2].set_title("% Relative Error, Quad Manifold ")
ax[2].plot(x, t_train[-1] * np.ones(len(x)), c = "white")
cbar = fig.colorbar(pos, ax = ax[2])
cbar.ax.set_ylabel("s(x,t)", rotation = 90)

pos = ax[3].imshow(S_ROM_linear.T , extent = [0, 1, t[0], t[-1]], vmax = 40, vmin = 0, aspect = "auto", origin = "lower")
ax[1].set_xlabel("x-coordinate")
ax[1].set_title("Quadratic Manifold ROM")
ax[1].plot(x, t_train[-1] * np.ones(len(x)), c = "white")
cbar = fig.colorbar(pos, ax = ax[1])
cbar.ax.set_ylabel("s(x,t)", rotation = 90)

pos = ax[4].imshow(100 * np.abs(S_ROM_linear.T - s.T) / np.abs(s.T), extent = [0, 1, t[0], t[-1]], cmap = 'plasma', aspect = "auto", origin = "lower")
ax[4].set_xlabel("x-coordinate")
ax[4].set_title("% Relative Error, Linear Projection")
ax[4].plot(x, t_train[-1] * np.ones(len(x)), c = "white")
cbar = fig.colorbar(pos, ax = ax[4])
cbar.ax.set_ylabel("s(x,t)", rotation = 90)