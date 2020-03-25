import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.linear_model import Ridge
from scipy.special import expit
from scipy.integrate import solve_ivp
import math
import sympy as sym



def muscle_excitation_2(time):
    percents = (time/0.375 *100)%100
    muscle_excitations = []
    print(percents)
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

def muscle_excitation_2_singleval(time):
    percent = (time/0.375 *100)%100
    if percent < 0:
        print("this is an issue! (x is less than 0)")
        return 0
    elif percent < 15:
        return 1
    else:
        return 2 ** (-1 * (percent - 12.678)) + 0.3



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

def muscle_excitation_1_singleval(time):
    percent = (time / 0.375 * 100) % 100
    if percent < 0:
        print("this is an issue! (x is less than 0)")
        return 0
    elif percent < 10:
        return percent / 10
    elif percent < 30:
        return 1
    elif percent < 40:
        return -percent / 20 + 2.5
    elif percent < 60:
        return 0.5
    elif percent < 80:
        return -percent / 40 + 2
    else:
        return 0

def muscle_excitation_3_singleval(time):
    percent = (time/0.375 *100)%100
    if percent < 0:
        print("this is an issue! (x is less than 0)")
        return 0
    elif percent < 15:
        return 1
    else:
        return 2 ** (-1 * (percent - 12.678))



def muscle_excitation_3(time):
    percents = (time/0.375 *100)%100
    muscle_excitations = []
    print(percents)
    for percent in percents:
        if percent < 0:
            print("this is an issue! (x is less than 0)")
            muscle_excit = 0
        elif percent < 15:
            muscle_excit = 1
        else:
            muscle_excit = 2 ** (-1 * (percent - 12.678))

        muscle_excitations.append(muscle_excit)

    return muscle_excitations

time = np.linspace(0.0001, 0.3, 100)
muscle_excitation = muscle_excitation_1(time)
#muscle_excitation2 = muscle_excitation_2(time)

def activation_dot(t, fad):
    return (muscle_excitation_3_singleval(t) - fad) * (muscle_excitation_3_singleval(t) /0.04 - (1 - muscle_excitation_3_singleval(t)) / 0.04)


sol = solve_ivp(activation_dot, [0, 0.3], [0], max_step=.01, rtol=1e-5, atol=1e-8)

#print(sol.y)
print(sol.y[0, -1])
print('kaljdfhoaiep oewihcfpoiaewf')
for i in range(len(sol.y[0])):
    if sol.y[0,i] > 1:
        sol.y[0,i] = 1
    elif sol.y[0,i] < 0:
        sol.y[0, i] = 0

print(sol.y)
muscle_excitations = muscle_excitation_3(sol.t)
plt.figure()
plt.plot(sol.t, muscle_excitations)
plt.xlabel('Time (s)')
plt.ylabel('muscle excitation')
plt.tight_layout()
plt.show()
#

"""
0 < fad < 1
"""
plt.figure()
plt.plot(sol.t, sol.y[0])
plt.xlabel('Time (s)')
plt.ylabel('fad')
plt.tight_layout()
plt.show()


#def f(y):
#    return 4*y**3 + 3*y**2 + 7*y
#
#def derivative(function):
#    return sym.diff(function)
#
#def evaluate(function, x):
#    return function
#
#x = sym.symbols('x')
#
#derivative_function = sym.diff(f(x))
##derivative_value = evaluate(derivative_function(f(x)), 4)
#value = derivative_function.evalf(subs={x: 1})
#print(derivative_function)
#print(value)






#plt.figure()
#plt.plot(time, muscle_excitation)
#plt.xlabel('Time (s)')
#plt.ylabel('muscle excitation')
#plt.tight_layout()
#plt.show()
#
#plt.figure()
#plt.plot(time, muscle_excitation2)
#plt.xlabel('Time (s)')
#plt.ylabel('muscle excitation')
#plt.tight_layout()
#plt.show()
