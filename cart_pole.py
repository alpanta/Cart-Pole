import random
from math import pi, cos, sin, tanh
import matplotlib.pyplot as plt
import numpy as np
import time

# Sigmoid function
def sig(value):
    a = np.exp(-value)
    return 1.0/ (1.0 + a)

# Constants
Alpha = 0.5
Beta = 0.5
Delta = 0.5
Gamma = 0.95
Lambda = 0.8
dt = 1E-2   # State variables' step size

M = 1   # Cart mass
m = 0.1     # Pole mass
m1 = 1.1    # Total mass
l = 0.5     # Pole length
g = -9.8    # Gravitational acceleration
mi_c = 0.0005   # Coefficient of friction between car and surface
mi_p = 0.000002 # Coefficient of friction between car and pole

# Initial conditions
x = 0.05   # Initial position
x_dot = 0   # Initial velocity
theta = -0.05    # Initial angle
theta_dot = 0   # Initial angular velocity
F = 0   # Initial force
sens = 10   # Force sensitivity (F = y*sens)

state = np.array([[0, x, x_dot, theta, theta_dot, F]]) # Initial state variables

# Initial weights
w = np.reshape(np.array([random.uniform(0.01, 0.1), random.uniform(0.01, 0.1),
 random.uniform(0.01, 0.1), random.uniform(0.01, 0.1)]), (4,1))       # ASE
v = np.reshape(np.array([random.uniform(0.01, 0.1), random.uniform(0.01, 0.1),
 random.uniform(0.01, 0.1), random.uniform(0.01, 0.1)]), (4,1))       # ACE
e_t = np.reshape(np.array([random.uniform(0.01, 0.1), random.uniform(0.01, 0.1),
 random.uniform(0.01, 0.1), random.uniform(0.01, 0.1)]), (4,1))    # Eligibility

xhat_t = np.reshape(np.array([random.uniform(0.01, 0.1),
  random.uniform(0.01, 0.1), random.uniform(0.01, 0.1),
  random.uniform(0.01, 0.1)]), (4,1))  # Initial condition for xhat(t)

n_t = random.uniform(0, 0.05)        # Noise signal
p_t_old = random.uniform(0, 0.05)    # Initial condition for p(t-1)

# Main loop
t = 0   # Step counter
t_step = 100    # Desired total step number
tri_no = 1  # Desired trial number
r_tp = 1    # Reward signal value
r_tn = -1   # Penalty signal value
p_rf = 0    # Total reward value for single trial
res = 0     # Total resets for single trial

x_list = list()
x_dot_list = list()
theta_list = list()
theta_dot_list = list()    # Lists to hold values for plotting
f_list = list()
t_vec = list()
rhat_t_list = list()
r_t_list = list()

for trial in range(tri_no):
    print('\n                           -----Trial '+str(trial+1)+'-----\n')
    t1 = time.time()
    while(t < t_step):

        if((state[0,3] > -12) and (state[0,3] < 12) and (state[0,1] > -2.4) and
        (state[0,1] < 2.4)):    # Reward conditions
            r_t = r_tp
            p_rf += 1

        elif((state[0,3] <= -12) or (state[0,3] >= 12) or
        (state[0,1] <= -2.4) or (state[0,1] >= 2.4)): # Reset conditions
            r_t = r_tn
            res += 1
            state = np.array([[0, x, x_dot, theta, theta_dot, F]])

        f_list.append(state[0,5])

        # Updates of equations
        x_t = np.transpose(state[:,1:5])
        r_t_list.append(r_t)

        # ACE equations
        p_t = sig(np.matmul(np.transpose(v),x_t))
        v = v + Beta*(r_t + Gamma*p_t - p_t_old)*xhat_t
        xhat_t = Lambda*xhat_t + (1-Lambda)*x_t
        rhat_t = r_t + Gamma*p_t - p_t_old
        p_t_old = p_t

        # ASE equations
        y = tanh(np.matmul(np.transpose(w), x_t) + n_t)
        w = w + Alpha*rhat_t*e_t
        e_t = Delta*e_t + (1-Delta)*y*x_t

        # Updates of state variables
        f1 = m*l*(state[0,4]**2)*sin((pi/180)*state[0,3])
        - mi_c*np.sign(state[0,2])
        f2 = g*sin((pi/180)*state[0,3]) + cos((pi/180)*state[0,3])*(-state[0,5]
         - f1)/m1 - mi_p*state[0,4]/m*l

        state[0,1] = state[0,1] + dt*state[0,2]
        state[0,2] = state[0,2] + dt*(state[0,5] + (f1 - m*l*f2*cos((pi/180)
        *state[0,3])))/m1
        state[0,3] = state[0,3] + dt*state[0,4]
        state[0,4] = state[0,4] + dt*f2/(l*(4/3 - m*cos((pi/180)*state[0,3])
        *cos((pi/180)*state[0,3])/m1))
        state[0,5] = y*sens

        x_list.append(state[0,1])
        x_dot_list.append(state[0,2])
        theta_list.append(state[0,3])
        theta_dot_list.append(state[0,4])
        rhat_t_list.append(rhat_t[0,0])
        t_vec.append(t)
        t += 1

        print('   Prediction Error ='+ (str(abs(rhat_t[0,0] - r_t)))
        + "           F =" +str(state[0,5]))

        # # Stopping Conditions
        #
        # if p_rf == 100:
        #     print('\nPole balanced successfully for at
        # least {} steps.format(p_rf))
        #     break
        # elif res == 40:
        #     print('\nPole not balanced.
        # Stopping after {} failures.'.format(res))
        #     break

    print('\n                           -----End of Trial '+str(trial+1)
    + '-----\n')

    t1 = time.time() - t1

    print('Total Resets for Present Trial = ', res)
    print('Total Reward Value for Present Trial = ', p_rf)
    print('\nTrial lasted for' + str(t_step) + ' steps and %0.2f seconds.'%t1 )

    # Initializing weights and state variables for next trial
    w = np.reshape(np.array([random.uniform(0.01, 0.1),
     random.uniform(0.01, 0.1), random.uniform(0.01, 0.1),
      random.uniform(0.01, 0.1)]), (4,1))
    v = np.reshape(np.array([random.uniform(0.01, 0.1),
     random.uniform(0.01, 0.1), random.uniform(0.01, 0.1),
      random.uniform(0.01, 0.1)]), (4,1))
    e_t = np.reshape(np.array([random.uniform(0.01, 0.1),
     random.uniform(0.01, 0.1), random.uniform(0.01, 0.1),
      random.uniform(0.01, 0.1)]), (4,1))
    xhat_t = np.reshape(np.array([random.uniform(0.01, 0.1),
     random.uniform(0.01, 0.1), random.uniform(0.01, 0.1),
      random.uniform(0.01, 0.1)]), (4,1))
    n_t = random.uniform(0, 0.05)
    p_t_old = random.uniform(0, 0.05)

    state = np.array([[0, x, x_dot, theta, theta_dot, F]])

    res=0
    p_rf=0
    t = 0

    # Plots
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5)
    ax1.plot(t_vec,x_list, 'b')
    ax2.plot(t_vec,x_dot_list, 'c')
    ax3.plot(t_vec,theta_list, 'r')
    ax4.plot(t_vec,theta_dot_list, 'm')
    ax5.plot(t_vec,f_list, 'g')
    ax1.title.set_text('Position')
    ax2.title.set_text('Velocity')
    ax3.title.set_text('Angle')
    ax4.title.set_text('Angular Velocity')
    ax5.title.set_text('Force')
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(t_vec, r_t_list, '-r', label='R')
    plt.plot(t_vec,rhat_t_list, '--g', label='RHAT')
    plt.ylabel('Reinforcement Signal')
    plt.xlabel('Epoch')
    plt.title('Reinforcement Signal vs. Prediction of Reinforcement Signal')
    leg = ax.legend()
    plt.show()

    x_list.clear()
    x_dot_list.clear()
    theta_list.clear()
    theta_dot_list.clear()         # Deleting list items for next trial
    f_list.clear()
    t_vec.clear()
    rhat_t_list.clear()
    r_t_list.clear()
