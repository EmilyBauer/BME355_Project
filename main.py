"""
Final 355 Group Project
main.py

Tigger Wong 		20673283
Rachel DiMaio	    20704330
Erin Roulston 		20718053
Emily Bauer 		20727725
Sarah Schwartzel 	20710946
2020/03/21
"""

import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.linear_model import Ridge
from scipy.special import expit
from scipy.integrate import solve_ivp
import math

alphaF # is the angular motion of the foot. alphaF = 0 when foot is horizontal
alphaS # is the shank orientation in 3D space. alphaS = 0 when shank is vertical
omegaS # is the angular velocity of the shank
ax,A # is the acceleration of the ankle in the x-direction
aY,A # is the acceleration of the ankle in the y-direction
Tgrav # is the torque generated by gravity
mF # is the mass of the foot
cF # is the foot’s center of mass location with respect to the ankle
TTA # is the torque generated by the tibialis anterior
d # is the lever arm of the tibialis anterior at the ankle. d = 0.037
FTA # is the muscle force of the tibialis anterior. FTA = FCE + FPE + FPD 
FPE # is the parallel elastic force
FPD # is the parallel damping element force
FCE # is the force in the contractile element. FCE = Fiso,max* fad* ftl* ffv
fad # is the activation level. 0 ≤ fad ≤ 1
muscle_excitation # is the muscle excitation. 0 ≤ muscle_excitation  ≤ 1
Tact # is the time constant for activation. Tact = 0.04s
Tdeact # is the time constant for deactivation. Tdeact = 0.04s
lMT # is the length of the muscle-tendon complex for the tibialis anterior. lMT = lMT,0 - dankle*phi
lMT,0 # is the rest length.  lMT,0 = 0.381
phi # is the current relative angle at the foot. phi :=  alphaS - alphaF. phi = 0 when there is a right angle at the ankle
lT # is the tendon length constant. lT = 0.317
lCE # is the length of the contractile element. lCE = lMT - lT  
VCE # is the contraction speed of the muscle. VCE =  - dankle*phi (*** the phi has a lil dot on him but idk how to draw it)
ffl # is the force-length factor. ffl = 
ICE, opt # is the fiber length at which the optimum force can be generated. ICE, opt = 0.082
W # is the width parameter describing the overlapping of filaments in the sarcomere. W = 0.56
ffv # is the force-velocity factor. It models the dependency of the muscle force on the contraction speed of the muscle. ffv =  if vCE  > 0 (extension).  ffv = else (contraction)
gmax = 0.82
LCE, opt = 0.82   
#---- not complete (but almost) ----


class SwingPhase:
    def __init__(self, muscle_excitation, phi, soleus_length):
        self.soleus_length = 0.3
        self.resting_MT_length = 0.381
        self.mass_foot = 5  # mF is the mass of the foot
        self.foot_center = 0.08    # cF is the foot’s center of mass location with respect to the ankle
        self.lever_arm = 0.037     # d is the lever arm of the tibialis anterior at the ankle. d = 0.037
        self.tendon_length = 0.317      # lT is the tendon length constant. lT = 0.317
        self.time_activation = 0.04     # is the time constant for activation. Tact = 0.04s
        self.time_deactivation = 0.04   # is the time constant for deactivation. Tdeact = 0.04s  
        self.gmax = 0.82          # idk what this is for
        self.LCE_opt = 0.082     # optimal contractile element length (fibre length at which optimal force can be generated)
        self.width_parameter = 0.56 # describing the overlapping of filaments in the sarcomere
        self.v_max = #NOT SURE YET
        self.max_isometric_force = #NOT SURE YET

    def get_angle_foot():   # alphaF is the angular motion of the foot. alphaF = 0 when foot is horizontal
        return
    
    def get_angle_shank():  # alphaS is the shank orientation in 3D space. alphaS = 0 when shank is vertical
        return
    
    def get_angular_velocity_shank():   # omegaS is the angular velocity of the shank; THIS IS A CONTROL VARIABLE (we have to provide values)
        return
    
    def get_ankle_x_acceleration(): # ax,A  is the acceleration of the ankle in the x-direction; THIS IS A CONTROL VARIABLE (we have to provide values)
        return
    
    def get_ankle_y_acceleration(): # aY,A is the acceleration of the ankle in the y-direction; THIS IS A CONTROL VARIABLE (we have to provide values)
        return
    
    def get_gravity_torque():   # Tgrav is the torque generated by gravity
        return -1*self.foot_center*math.cos(angle_foot())*self.mass_foot*9.81
    
    def get_TA_torque(): #TTA is the torque generated by the tibialis anterior
        return self.get_TA_force() * self.lever_arm
    
    def get_soleus_length():
        return
    
    def get_TA_force():     # FTA is the muscle force of the tibialis anterior. FTA = FCE + FPE + FPD 
        return get_contractile_element_force() + get_parallel_elastic_force() + get_parallel_dampening_force()
    
    def get_parallel_elastic_force():   #FPE is the parallel elastic force
        return
    
    def get_parallel_dampening_force(): # FPD is the parallel damping element force
        b = 0.001*max_isometric_force/self.LCE_opt
        return b*self.get_contraction_speed
    
    def get_contractile_element_force():    # FCE is the force in the contractile element. FCE = Fiso,max* fad* ftl* ffv
        return max_isometric_force*get_activation_level()*force_length_factor*force_velocity_factor()
    
    def get_activation_level(): # fad is the activation level. 0 ≤ fad ≤ 1
        return 
    
    def force_length_factor(): 
        return math.exp(-1*((self.get_CE_length - self.LCE_opt)/(self.width_parameter*self.LCE_opt))**2)
    
    def force_velocity_factor(): 
        return (self.get_lambda*self.v_max + self.get_continuous_derivative_factor)/(self.get_lambda*self.v_max - self.get_contraction_speed/0.25)
    
    def get_relative_angle_foot():  # is the current relative angle at the foot. phi :=  angleS - angleF. phi = 0 when there is a right angle at the ankle
        return (self.get_angle_shank() - self.get_angle_foot())
    
    def get_MT_length(): # is the length of the muscle-tendon complex for the tibialis anterior. lMT = lMT,0 - dankle*phi
        return self.resting_MT_length - self.lever_arm*self.get_relative_angle_foot()
    
    def get_CE_length(): # is the length of the contractile element. lCE = lMT - lT  
        return self.get_MT_length() - self.tendon_length
    
    def get_lambda():
        return (1 - math.exp(-3.82 * self.get_activation_level()) + self.get_activation_level() * math.exp(-3.82))
    
    def get_contraction_speed(): # is the contraction speed of the muscle. VCE =  - dankle*phi
        return -1*self.lever_arm*self.get_relative_angle_foot() # MUST TAKE TIME DERIVATIVE OF RELATIVE ANGLE
    
    def get_continuous_derivative_factor(): 
        return ((self.get_lambda() * self.v_max * 0.25 * (self.gmax - 1))/(0.25 + 1))


"""
Swing phase between 0 and 0.3 -  80% of swing cycle
"""

# multiple functions to model/optimise epsilon
# define function

def muscle_excitation_1(time):
    percents = (time / 0.375 * 100) % 100
    muscle_excitations = []
    for percent in percents:
        if percent < 0:
            print("this is an issue! (x is less than 0)")
            muscle_excit = 0
        elif percent < 10:
            muscle_excit = percent / 10
        elif percent < 30:
            muscle_excit = 1
        elif percent < 40:
            muscle_excit = -percent / 20 + 2.5
        elif percent < 60:
            muscle_excit = 0.5
        elif percent < 80:
            muscle_excit = -percent / 40 + 2
        else:
            muscle_excit = 0
        muscle_excitations.append(muscle_excit)

    return muscle_excitations

def muscle_excitation_2(time):
    percents = (time/0.375 *100)%100
    muscle_excitations = []
    for percent in percents:
        if percent < 0:
            print("this is an issue! (x is less than 0)")
            muscle_excit = 0
        elif percent < 15:
            muscle_excit = 1
        else:
            muscle_excit = 2 ** (-1 * (percent - 12.678)) + 0.3

        muscle_excitations.append(muscle_excit)

    return muscle_excitations
