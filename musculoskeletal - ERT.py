import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.linear_model import Ridge
from scipy.special import expit
from scipy.integrate import solve_ivp


class HillTypeMuscle:
    """
    Damped Hill-type muscle model adapted from Millard et al. (2013). The
    dynamic model is defined in terms of normalized length and velocity.
    To model a particular muscle, scale factors are needed for force, CE
    length, and SE length. These are given as constructor arguments.
    """

    def __init__(self, f0M, resting_length_muscle, resting_length_tendon):
        """
        :param f0M: maximum isometric force
        :param resting_length_muscle: actual length (m) of muscle (CE) that corresponds to
            normalized length of 1
        :param resting_length_tendon: actual length of tendon (m) that corresponds to
            normalized length of 1
        """
        self.f0M = f0M
        self.resting_length_muscle = resting_length_muscle
        self.resting_length_tendon = resting_length_tendon

    def norm_tendon_length(self, muscle_tendon_length, normalized_muscle_length):
        """
        :param muscle_tendon_length: non-normalized length of the full muscle-tendon
            complex (typically found from joint angles and musculoskeletal geometry)
        :param normalized_muscle_length: normalized length of the contractile element
            (the state variable of the muscle model)
        :return: normalized length of the tendon
        """
        return (muscle_tendon_length - self.resting_length_muscle * normalized_muscle_length) / self.resting_length_tendon

    def get_force(self, total_length, norm_muscle_length):
        """
        :param total_length: muscle-tendon length (m)
        :param norm_muscle_length: normalized length of muscle (the state variable)
        :return: muscle tension (N)
        """
        return self.f0M * force_length_tendon(self.norm_tendon_length(total_length, norm_muscle_length))


def get_velocity(a, lm, lt):
    """
    :param a: activation (between 0 and 1)
    :param lm: normalized length of muscle (contractile element)
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized lengthening velocity of muscle (contractile element)
    """
    beta = 0.1 # damping coefficient (see damped model in Millard et al.)

    def funk(vm):
        return (a*force_length_muscle(lm)*force_velocity_muscle(vm) + force_length_parallel(lm) + beta*vm) - force_length_tendon(lt)

    result = fsolve(funk, 0)
    return result



def force_length_tendon(lt):
    """
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized tension produced by tendon
    """
    slack_length = 1

    if isinstance(lt, float) or isinstance(lt, int):
        if lt < slack_length:
            result = 0
        else:
            result = 10 * (lt - slack_length) + 240 * pow((lt - slack_length), 2)

    else:
        result = np.zeros(len(lt))
        for i in range(len(lt)):
            if lt[i] < slack_length:
                result[i] = 0

            else:
                result[i] = 10 * (lt[i] - slack_length) + 240 * pow((lt[i] - slack_length), 2)

    return result




def force_length_parallel(lm):
    """
    :param lm: normalized length of muscle (contractile element)
    :return: normalized force produced by parallel elastic element
    """
    slack_length = 1
    if isinstance(lm, int) or isinstance(lm, float):
        if lm < slack_length:
            result = 0
        else:
            result = 3 * pow((lm - slack_length),2)/(0.6 + lm - slack_length)
    else:
        result = np.zeros(len(lm))
        for i in range(len(lm)):
            if lm[i] < slack_length:
                result[i] = 0

            else:
                result[i] = 3 * pow((lm[i] - slack_length),2)/(0.6 + lm[i] - slack_length)

    return result

def plot_curves():
    """
    Plot force-length, force-velocity, SE, and PE curves.
    """
    lm = np.arange(0, 1.8, .01)
    vm = np.arange(-1.2, 1.2, .01)
    lt = np.arange(0, 1.07, .01)
    plt.subplot(2,1,1)
    plt.plot(lm, force_length_muscle(lm), 'r')
    plt.plot(lm, force_length_parallel(lm), 'g')
    plt.plot(lt, force_length_tendon(lt), 'b')
    plt.legend(('CE', 'PE', 'SE'))
    plt.xlabel('Normalized length')
    plt.ylabel('Force scale factor')
    plt.subplot(2, 1, 2)
    plt.plot(vm, force_velocity_muscle(vm), 'k')
    plt.xlabel('Normalized muscle velocity')
    plt.ylabel('Force scale factor')
    plt.tight_layout()
    plt.show()


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-(x-self.mu)**2/2/self.sigma**2)


class Sigmoid:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return expit((x-self.mu) / self.sigma)


class Regression():
    """
    1D regression model with Gaussian basis functions.
    """

    def __init__(self, x, t, centres, width, regularization_weight=1e-6, sigmoids=False):
        """
        :param x: samples of an independent variable
        :param t: corresponding samples of a dependent variable
        :param centres: a vector of Gaussian centres (should have similar range of values as x)
        :param width: sigma parameter of Gaussians
        :param regularization_weight: regularization strength parameter
        """
        if sigmoids:
            self.basis_functions = [Sigmoid(centre, width) for centre in centres]
        else:
            self.basis_functions = [Gaussian(centre, width) for centre in centres]
        self.ridge = Ridge(alpha=regularization_weight, fit_intercept=False)
        self.ridge.fit(self._get_features(x), t)

    def eval(self, x):
        """
        :param x: a new (or multiple samples) of the independent variable
        :return: the value of the curve at x
        """
        return self.ridge.predict(self._get_features(x))

    def _get_features(self, x):
        if not isinstance(x, collections.Sized):
            x = [x]

        phi = np.zeros((len(x), len(self.basis_functions)))
        for i, basis_function in enumerate(self.basis_functions):
            phi[:,i] = basis_function(x)
        return phi


def get_muscle_force_velocity_regression():
    data = np.array([
        [-1.0028395556708567, 0.0024834319945283845],
        [-0.8858611825192801, 0.03218792009622429],
        [-0.5176245843258415, 0.15771090304473967],
        [-0.5232565269687035, 0.16930496922242444],
        [-0.29749770052593094, 0.2899790099290114],
        [-0.2828848376217543, 0.3545364496120378],
        [-0.1801231103040022, 0.3892195938775034],
        [-0.08494610976156225, 0.5927831890757294],
        [-0.10185137142991896, 0.6259097662790973],
        [-0.0326643239546236, 0.7682365981934388],
        [-0.020787245583830716, 0.8526638522676352],
        [0.0028442725407418212, 0.9999952831301149],
        [0.014617579774061973, 1.0662107025777694],
        [0.04058866536166583, 1.124136223202283],
        [0.026390887007381902, 1.132426122025424],
        [0.021070257776939272, 1.1986556920827338],
        [0.05844673474682183, 1.2582274002971627],
        [0.09900238201929201, 1.3757434966156459],
        [0.1020023112662436, 1.4022310794556732],
        [0.10055894908138963, 1.1489210160137733],
        [0.1946227683309354, 1.1571212943090965],
        [0.3313459588217258, 1.152041225442796],
        [0.5510200231126625, 1.204839508502158]
    ])

    velocity = data[:,0]
    force = data[:,1]

    centres = np.arange(-1, 0, .2)
    width = .15
    result = Regression(velocity, force, centres, width, .1, sigmoids=True)

    return result


def get_muscle_force_length_regression():
    """
    CE force-length data samples from Winters et al. (2011) Figure 3C,
    normalized so that max force is ~1 and length at max force is ~1.
    The samples were taken form the paper with WebPlotDigitizer, and
    cut-and-pasted here.

    WRITE CODE HERE 1) Use WebPlotDigitizer to extract force-length points
    from Winters et al. (2011) Figure 3C, which is on Learn. Click
    "View Data", select all, cut, and paste below. 2) Normalize the data
    so optimal length = 1 and peak = 1. 3) Return a Regression object that
    uses Gaussian basis functions.
    """
    data = np.array([
       [41.8, 1.5384615384615188],
       [39.39999999999999, 3.5384615384614904],
       [37.36, 9.538461538461519],
       [41.36, 14.769230769230745],
       [41.36, 15.69230769230768],
       [38.4, 14.461538461538439],
       [40.4, 17.84615384615384],
       [40.4, 21.53846153846152],
       [39.28, 24.615384615384585],
       [41.36, 27.076923076923066],
       [42.96, 23.384615384615373],
       [43.44, 23.69230769230768],
       [43.92, 22.153846153846132],
       [41.36, 31.999999999999986],
       [42.16, 31.999999999999986],
       [40.4, 36.61538461538461],
       [43.44, 35.076923076923066],
       [43.44, 34.46153846153845],
       [42.56, 42.46153846153845],
       [43.44, 44.92307692307692],
       [44.400000000000006, 45.53846153846153],
       [45.6, 43.69230769230768],
       [45.519999999999996, 46.76923076923076],
       [45.6, 46.153846153846146],
       [46.800000000000004, 44.92307692307692],
       [42.96, 46.46153846153845],
       [42.96, 48.61538461538461],
       [43.12, 50.46153846153847],
       [43.36, 54.46153846153845],
       [43.760000000000005, 57.53846153846153],
       [45.36, 53.53846153846153],
       [45.2, 54.153846153846146],
       [44.400000000000006, 60.923076923076934],
       [45.68000000000001, 68],
       [46.480000000000004, 63.07692307692308],
       [47.36, 62.769230769230774],
       [48.96000000000001, 63.07692307692308],
       [47.760000000000005, 67.0769230769231],
       [47.44, 72.00000000000001],
       [45.92, 71.07692307692308],
       [46.400000000000006, 71.69230769230771],
       [46.400000000000006, 74.15384615384616],
       [46.64, 75.69230769230771],
       [47.040000000000006, 80.92307692307693],
       [47.44, 81.84615384615385],
       [47.6, 81.84615384615385],
       [48.24000000000001, 83.69230769230771],
       [48.96000000000001, 82.76923076923079],
       [48.96000000000001, 81.84615384615385],
       [49.760000000000005, 82.15384615384616],
       [49.44, 76.61538461538463],
       [50.56, 74.76923076923077],
       [50.72, 78.15384615384616],
       [50.96000000000001, 80.61538461538464],
       [51.120000000000005, 79.0769230769231],
       [50.88, 79.38461538461539],
       [49.760000000000005, 85.53846153846155],
       [49.6, 87.0769230769231],
       [48.96000000000001, 85.53846153846155],
       [50.64, 85.53846153846155],
       [50.64, 84.92307692307693],
       [50.64, 87.3846153846154],
       [50.32, 88.00000000000001],
       [50.72, 91.0769230769231],
       [51.52, 90.15384615384617],
       [51.760000000000005, 91.38461538461542],
       [52.72000000000001, 89.23076923076925],
       [53.28, 88.92307692307693],
       [53.52000000000001, 83.69230769230771],
       [53.52000000000001, 79.38461538461539],
       [53.92, 92.30769230769232],
       [53.68000000000001, 92.92307692307695],
       [53.52000000000001, 95.0769230769231],
       [53.60000000000001, 97.23076923076925],
       [53.84, 97.23076923076925],
       [54.32000000000001, 95.0769230769231],
       [54.08, 94.15384615384617],
       [53.84, 94.15384615384617],
       [54.64, 100.30769230769232],
       [55.760000000000005, 96.61538461538464],
       [56.08, 96.61538461538464],
       [56.400000000000006, 100.30769230769232],
       [56.88000000000001, 100.30769230769232],
       [57.040000000000006, 99.69230769230771],
       [57.040000000000006, 100.30769230769232],
       [57.28, 98.46153846153848],
       [57.52000000000001, 100.30769230769232],
       [57.84, 99.69230769230771],
       [58.56, 96.61538461538464],
       [58.88000000000001, 99.69230769230771],
       [59.360000000000014, 98.46153846153848],
       [59.68000000000001, 96.30769230769232],
       [59.84, 97.23076923076925],
       [60, 96.00000000000003],
       [60.56, 100.00000000000003],
       [60.56, 94.15384615384617],
       [61.360000000000014, 95.38461538461542],
       [61.360000000000014, 92.30769230769232],
       [62.32000000000001, 97.23076923076925],
       [57.760000000000005, 91.69230769230771],
       [58.480000000000004, 91.38461538461542],
       [59.360000000000014, 91.38461538461542],
       [61.360000000000014, 88.30769230769232],
       [62.24000000000001, 89.53846153846155],
       [61.360000000000014, 84.61538461538463],
       [61.360000000000014, 79.69230769230771],
       [61.44000000000001, 77.23076923076924],
       [62.56, 80.00000000000001],
       [63.44000000000001, 80.00000000000001],
       [63.360000000000014, 80.92307692307693],
       [63.84, 80.61538461538464],
       [64.48, 82.46153846153848],
       [63.920000000000016, 76.30769230769232],
       [65.12, 76.46153846153847],
       [65.52000000000001, 73.23076923076923],
       [66.08000000000001, 72.61538461538463],
       [63.120000000000005, 86.76923076923079],
       [63.60000000000001, 85.53846153846155],
       [63.60000000000001, 85.84615384615387],
       [63.760000000000005, 87.0769230769231],
       [63.68000000000001, 90.15384615384617],
       [64.48, 87.3846153846154],
       [66.4, 76],
       [65.68, 68.92307692307693],
       [65.80000000000001, 66.92307692307693],
       [65.36000000000001, 64.30769230769232],
       [66.88000000000001, 66.46153846153847],
       [63.360000000000014, 59.69230769230769],
       [63.360000000000014, 52.92307692307692],
       [64.72000000000001, 53.84615384615384],
       [64.80000000000001, 52.30769230769231],
       [65.68, 47.999999999999986],
       [67.12, 63.38461538461539],
       [67.76, 62.769230769230774],
       [68.4, 59.69230769230769],
       [67.52000000000001, 51.69230769230769],
       [66.96000000000001, 42.76923076923076],
       [70.52000000000001, 48.923076923076934],
       [69.36000000000001, 42.153846153846146],
       [69.36000000000001, 41.84615384615384],
       [67.36000000000001, 35.69230769230768],
       [68.48, 27.076923076923066],
       [70.16000000000001, 29.538461538461533],
       [70.4, 30.153846153846132],
       [71.36000000000001, 34.76923076923076],
       [73.44000000000001, 34.46153846153845],
       [72.96000000000001, 25.538461538461533],
       [72.4, 24.307692307692292],
       [73.36000000000001, 18.461538461538453],
       [73.36000000000001, 17.53846153846152],
       [73.36000000000001, 12.615384615384585],
       [74.32000000000001, 12.923076923076906],
       [75.12000000000002, 17.84615384615384],
       [75.44000000000001, 12.615384615384585],
       [76.4, 8.307692307692292]
    ])

    force_max = 100.3077
    length_at_max = 54.64

    force_normalized = data[:, 1]/force_max
    length_normalized = data[:, 0]/length_at_max

    centres = np.arange(0.74, 1.5, .2)
    width = .15
    result = Regression(length_normalized, force_normalized, centres, width, .1, sigmoids=False)
    return result


force_length_regression = get_muscle_force_length_regression()
force_velocity_regression = get_muscle_force_velocity_regression()


def force_length_muscle(lm):
    """
    :param lm: muscle (contractile element) length
    :return: force-length scale factor
    """
    return force_length_regression.eval(lm)


def force_velocity_muscle(vm):
    """
    :param vm: muscle (contractile element) velocity)
    :return: force-velocity scale factor
    """
    return np.maximum(0, force_velocity_regression.eval(vm))

if __name__ == '__main__':
    resting_muscle_length = 0.3
    resting_tendon_length = 0.1
    max_isometric_force = 100
    muscle = HillTypeMuscle(max_isometric_force, resting_muscle_length, resting_tendon_length)

    velocity = get_velocity(1, 1, 1.01)
    print(velocity)

    total_length = resting_muscle_length + resting_tendon_length


    def f(t, lm):
        if (t < 0.5):
            activation = 0
        else:
            activation = 1

        return get_velocity(activation, lm, muscle.norm_tendon_length(total_length, lm))

    sol = solve_ivp(f, [0, 2], [1], max_step=.01, rtol=1e-5, atol=1e-8)

    plt.figure()
    plt.subplot(211)
    plt.plot(sol.t, sol.y.T)
    plt.xlabel('Time (s)')
    plt.ylabel(' CE length')
    plt.subplot(212)
    plt.plot(sol.t, (muscle.get_force(total_length, sol.y.T)))
    plt.xlabel('Time (s)')
    plt.ylabel('Force')
    plt.tight_layout()
    plt.show()
