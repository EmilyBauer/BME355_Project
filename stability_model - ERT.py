"""
Simple model of standing postural stability, consisting of foot and body segments,
and two muscles that create moments about the ankles, tibialis anterior and soleus.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from musculoskeletal import HillTypeMuscle, get_velocity, force_length_tendon


def soleus_length(theta):
    """
    :param theta: body angle (up from prone horizontal)
    :return: soleus length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    origin = np.dot(rotation, [.3, .03])
    insertion = [-.05, -.02]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def tibialis_length(theta):
    """
    :param theta: body angle (up from prone horizontal)
    :return: tibialis anterior length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    origin = np.dot(rotation, [.3, -.03])
    insertion = [.06, -.03]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def gravity_moment(theta):
    """
    :param theta: angle of body segment (up from prone)
    :return: moment about ankle due to force of gravity on body
    """
    mass = 75 # body mass (kg; excluding feet)
    centre_of_mass_distance = 1 # distance from ankle to body segment centre of mass (m)
    g = 9.81 # acceleration of gravity
    return mass * g * centre_of_mass_distance * np.sin(theta - np.pi / 2)


def dynamics(x, soleus, tibialis, control):
    """
    :param x: state vector (ankle angle, angular velocity, soleus normalized CE length, TA normalized CE length)
    :param soleus: soleus muscle (HillTypeModel)
    :param tibialis: tibialis anterior muscle (HillTypeModel)
    :param control: True if balance should be controlled
    :return: derivative of state vector
    """

    # WRITE CODE HERE TO IMPLEMENT THE MODEL
    tau_soleus = soleus.f0M * force_length_tendon(soleus.norm_tendon_length(soleus_length(x[0]), x[2])) * 0.05
    tau_TA = tibialis.f0M * force_length_tendon(tibialis.norm_tendon_length(tibialis_length(x[0]), x[3])) * 0.03


    #when x[1] is between -0.002 and 0.002 lay off on the activation a bit more, like we need to chill
    if control:
        #falling forwards
        if x[0] < np.pi/2 and x[1] < 0 :
            activation_soleus = 0.05
            activation_tibialis = 0
        #leaning forwards but coming back to straight
        elif x[0] < np.pi/2 and x[1] > 0:
            if x[1] < 0.002:
                activation_soleus = 0.001
                activation_tibialis = 0
            else:
                activation_soleus = 0.005
                activation_tibialis = 0
        #falling backwards
        elif x[0] > np.pi/2 and x[1] > 0:
            activation_soleus = 0
            activation_tibialis = 0.4
        #leaning backwards but coming back to straight
        elif x[0] > np.pi/2 and x[1] < 0:
            if x[1] > -0.002:
                activation_soleus = 0
                activation_tibialis = 0.02
            else:
                activation_soleus = 0
                activation_tibialis = 0.04
        else:
            activation_soleus = 0
            activation_tibialis = 0
    else:
        activation_soleus = 0.05
        activation_tibialis = 0.4

    x1dot = x[1]
    x2dot = (tau_soleus - tau_TA + gravity_moment(x[0])) / 90
    x3dot = get_velocity(activation_soleus, x[2],soleus.norm_tendon_length(soleus_length(x[0]), x[2]))
    x4dot = get_velocity(activation_tibialis, x[3],tibialis.norm_tendon_length(tibialis_length(x[0]), x[3]))

    return [x1dot, x2dot, x3dot, x4dot]

def simulate(control, T):
    """
    Runs a simulation of the model and plots results.
    :param control: True if balance should be controlled
    :param T: total time to simulate, in seconds
    """
    rest_length_soleus = soleus_length(np.pi/2)
    rest_length_tibialis = tibialis_length(np.pi/2)

    soleus = HillTypeMuscle(16000, .6*rest_length_soleus, .4*rest_length_soleus)
    tibialis = HillTypeMuscle(2000, .6*rest_length_tibialis, .4*rest_length_tibialis)

    def f(t, x):
        return dynamics(x, soleus, tibialis, control)

    sol = solve_ivp(f, [0, T], [np.pi / 2 - .001, 0, 1, 1], rtol=1e-5, atol=1e-8)
    time = sol.t
    theta = sol.y[0,:]
    soleus_norm_length_muscle = sol.y[2,:]
    tibialis_norm_length_muscle = sol.y[3,:]

    soleus_moment_arm = .05
    tibialis_moment_arm = .03
    soleus_moment = []
    tibialis_moment = []
    for th, ls, lt in zip(theta, soleus_norm_length_muscle, tibialis_norm_length_muscle):
        soleus_moment.append(soleus_moment_arm * soleus.get_force(soleus_length(th), ls))
        tibialis_moment.append(-tibialis_moment_arm * tibialis.get_force(tibialis_length(th), lt))

    plt.figure()
    plt.plot(time, sol.y[1,:])
    plt.show()


    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time, sol.y[0,:])
    plt.ylabel('Body angle (rad)')
    plt.subplot(2,1,2)
    plt.plot(time, soleus_moment, 'r')
    plt.plot(time, tibialis_moment, 'g')
    plt.plot(time, gravity_moment(sol.y[0,:]), 'k')
    plt.legend(('soleus', 'tibialis', 'gravity'))
    plt.xlabel('Time (s)')
    plt.ylabel('Torques (Nm)')
    plt.tight_layout()
    plt.show()


simulate(True, 10)