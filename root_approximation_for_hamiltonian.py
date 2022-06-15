# # Analytic Approximation to Root finding

# ## created by Scott Sims 06/09/2022
#=========================================================
# IMPORT PACKAGES
#=========================================================
import os
import sys
import pdb
import time
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import yaml
import bubble_methods as bub
from bubble_methods import get_num_steps

#=========================================================
# Input Arguments
#=========================================================
with open("parameters.yml", 'r') as stream:
    D = yaml.safe_load(stream)
for key in D:
    globals()[str(key)] = D[key]
    print(f"{str(key)}: {D[key]}")
    # transforms key-names from dictionary into global variables, then assigns them the dictionary-values
#---------------------------------------------------
#print('TO CONTINUE, PRESS [c] THEN [ENTER]')
#print('TO QUIT, PRESS [q] THEN [ENTER]')
#pdb.set_trace()
#=========================================================
root_folder = f"roots_of_F(Req,Cp)=0"
root_dir = os.path.join(os.getcwd(), root_folder)
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
#----------------------------------------------------
bubble = bub.Bubble(D)
#----------------------------------------------------
# PRINT CHARACTERISTICS
#----------------------------------------------------
print(f"--------------- SUMMARY -----------------")
print(f"equlibrium radius = {bubble.R_eq*1e6: .2f} micro-meters")
print(f"time-constant = {bubble.time_constant: .3e} sec")
print(f"------- NONDIMENSIONAL PARAMETERS -------")
print(f"Re = {bubble.reynolds: .2e}")
print(f"S_hat = {bubble.S: .4e}")
print(f"Ca = {bubble.Ca: .4f}")
print(f"natural frequency = {bubble.freq_natural: .4f}")
print(f"natural period = {bubble.T_natural: .4f}")
print(f"-----------------------------------------")
#=========================================================
# Initialize Intervals of Frequencies and Periods
#=========================================================
freq_range = [1/period_max, 1/period_min]
amp_range = [amp_min, amp_max]
n_steps = bub.get_num_steps(dt, model_steps, step_sizes[-1], period_min, n_periods)
#---------------------------------------------------
print(f"n_steps = {n_steps}")
t_final = dt * (n_steps)
print(f"t_final = {t_final}")
t_space = np.arange(0,t_final+dt,dt)[0:n_steps+1]
print(f"t_space = {t_space[0:3]}...{t_space[n_steps-2:n_steps+1]}")
#=========================================================
# Solve for roots 'Req' for a range of Cp(t) values
#=========================================================
def F(R,Cp,Ca,S):
    return ((Ca+Cp)*R + 2/S)*(R**(3.2)) - (2/S+Ca)

#---------------------------------------------------
delta = np.round(10**(-n_decimal_cp), decimals=n_decimal_cp)
asymptote = -bubble.Ca
#---------------------------------------------------
N1 = np.int64(np.round((amp_max-asymptote)/delta + 1))
Cp = np.linspace(asymptote, amp_max, N1)
print(f"N1 = {N1}")
print(f"Cp = {Cp[0]: .3f},{Cp[1]: .3f}, {Cp[2]: .3f},...")
A1 = Cp + bubble.Ca
R1 = np.zeros(N1)
R1[0] = fsolve(F, x0=0.2, args=(Cp[0], bubble.Ca, bubble.S))
for j in range(1,N1):
    R1[j] = fsolve(F, R1[j-1], args=(Cp[j], bubble.Ca, bubble.S))
#----------------------------------------------------
N2 = np.int64(np.round((asymptote+amp_max)/delta + 1))
Cp = np.linspace(-amp_max, asymptote, N2)
print(f"N2 = {N2}")
print(f"Cp = {Cp[-3]: .3f},{Cp[-2]: .3f}, {Cp[-1]: .3f},...")
A2 = Cp + bubble.Ca
R2 = np.zeros(N2)
R2[0] = fsolve(F, x0=0.2, args=(Cp[0], bubble.Ca, bubble.S))
for j in range(1,N2):
    R2[j] = fsolve(F, R2[j-1], args=(Cp[j], bubble.Ca, bubble.S)) 
#=========================================================
# Plot Data 'Req' vs Cp(t)
#=========================================================
print('==============================')
print('generating images ...')
print('==============================')
fig, axs = plt.subplots(2, 1, figsize=(plot_x_dim, 1.1 * plot_y_dim * 2))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
#-------------------------------------------------------------------------------
axs[0].plot(A1, R1, color='tab:blue', linewidth=line_width)
axs[0].set_xlabel('$C_p(t) + C_a$', fontsize=x_label_fontsize)
axs[0].set_ylabel('$R_{eq}$  of $G(R_{eq}, C_p(t))$', fontsize=y_label_fontsize)
parameters = f"{asymptote} $ \leq C_p(t) \leq $ {amp_max}  \n \n  $C_a = $ {bubble.Ca} \n \n  $\hat{{S}} = $ {bubble.S:.1f}"
axs[0].text(A1[np.int64(N1/2)], max(R1)*0.8, parameters, fontsize=box_fontsize, verticalalignment='top', bbox=props)
#axs[0].set_title('Roots of $0 = (C_a + C_p(t)) R_{eq}^{3 \gamma} + \\frac{2}{\hat{S}} R_{eq}^{3 \gamma - 1} - (\\frac{2}{\hat{S} + C_a)', fontsize=title_fontsize)
axs[0].tick_params(axis='both', which='major', labelsize=axis_fontsize)
#-------------------------------------------------------------------------------
axs[1].plot(A2, R2, color='tab:blue', linewidth=line_width)
axs[1].set_xlabel('$C_p(t) + C_a$', fontsize=x_label_fontsize)
axs[1].set_ylabel('$R_{eq}$  of $G(R_{eq}, C_p(t))$', fontsize=y_label_fontsize)
parameters = f"{-amp_max} $ \leq C_p(t) \leq $ {asymptote}  \n \n  $C_a = $ {bubble.Ca} \n \n  $\hat{{S}} = $ {bubble.S:.1f}"
axs[1].text(A2[np.int64(N2/2)], max(R2)*0.8, parameters, fontsize=box_fontsize, verticalalignment='top', bbox=props)
#axs[1].set_title('Roots of $0 = (C_a + C_p(t)) R_{eq}^{3 \gamma} + \\frac{2}{\hat{S}} R_{eq}^{3 \gamma - 1} - (\\frac{2}{\hat{S} + C_a)', fontsize=title_fontsize)
axs[1].tick_params(axis='both', which='major', labelsize=axis_fontsize)
#-------------------------------------------------------------------------------
fig.tight_layout(pad=2.0)
file_fig_root = os.path.abspath(os.path.join(root_dir, f"root_plot_Cp=[{-amp_max},{amp_max}]_Ca={bubble.Ca}_S={bubble.S:.1f}.png" ))
fig.savefig(file_fig_root, dpi=200)
#-------------------------------------------------------------------------------
print("images saved")
print('==============================')

