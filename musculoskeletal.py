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
        print ("making Hill-type stuff in musculo-file")

    def norm_tendon_length(self, muscle_tendon_length, normalized_muscle_length):
        """
        :param muscle_tendon_length: non-normalized length of the full muscle-tendon (0.4)
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
    def f(vm):
        p1 = a*force_length_muscle(lm)*force_velocity_muscle(vm)
        p2 = force_length_parallel(lm) + beta*vm
        p3 = force_length_tendon(lt)
        return ((p1 + p2) - p3)
    return fsolve(f, 0)

    # WRITE CODE HERE TO CALCULATE VELOCITY


def force_length_tendon(lt):
    """
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized tension produced by tendon
    """
    # WRITE CODE HERE
    if isinstance(lt, float) or isinstance(lt, int):
        if lt < 1:
            return 0
        elif lt >= 1:
            return ((3*(lt - 1)**2)/(-0.4+lt))
    

    lenTend = 0*lt
    for i in range(len(lt)):
        if lt[i] < 1:
            lenTend[i] = 0
        if lt[i] >= 1:
            lenTend[i] = (10*(lt[i] - 1)+240*(lt[i] - 1)**2)
    return lenTend


def force_length_parallel(lm):
    """
    :param lm: normalized length of muscle (contractile element)
    :return: normalized force produced by parallel elastic element
    """
    # WRITE CODE HERE
    if isinstance(lm, float) or isinstance(lm, int):
        if lm < 1:
            return 0
        else:
            return ((3*(lm - 1)**2)/(-0.4+lm))
    
    lenPar = lm*0
    for i in range(len(lm)):
        if lm[i] < 1:
            lenPar[i] = 0
        elif lm[i] >= 1:
            lenPar[i] = ((3*(lm[i] - 1)**2)/(-0.4+lm[i]))
    return lenPar

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
    The sampples were taken form the paper with WebPlotDigitizer, and
    cut-and-pasted here.
    WRITE CODE HERE 1) Use WebPlotDigitizer to extract force-length points
    from Winters et al. (2011) Figure 3C, which is on Learn. Click
    "View Data", select all, cut, and paste below. 2) Normalize the data
    so optimal length = 1 and peak = 1. 3) Return a Regression object that
    uses Gaussian basis functions. 
    """
    TAActive = np.array([
        [43.820375335120644, 21.929057537636623],
        [43.37801608579089, 23.469787585069085],
        [42.89544235924933, 23.472262322128273],
        [41.407506702412874, 14.402969684471017],
        [41.407506702412874, 15.479892761394098],
        [40.402144772117964, 17.485048463600748],
        [41.80965147453083, 1.6316766343576035],
        [39.35656836461126, 3.3365642400494835],
        [37.38605898123325, 9.50051557022067],
        [38.39142091152815, 14.418436791090954],
        [39.23592493297588, 24.41410600123737],
        [40.36193029490617, 21.485254691689008],
        [41.407506702412874, 26.710661992163338],
        [41.407506702412874, 31.787585069086404],
        [42.05093833780161, 32.09197772736647],
        [40.402144772117964, 36.56197154052383],
        [43.37801608579089, 34.85440296968446],
        [42.493297587131366, 41.93586306454938],
        [43.41823056300268, 44.546504433903905],
        [44.42359249329759, 45.46442565477419],
        [45.54959785522788, 46.227882037533504],
        [45.54959785522789, 45.9201897298412],
        [45.54959785522788, 43.458651268302745],
        [46.67560321715818, 44.68364611260053],
        [45.388739946380696, 53.61332233450196],
        [45.22788203753351, 54.07568570839348],
        [42.85522788203754, 46.24169931944732],
        [42.85522788203754, 48.24169931944731],
        [43.1367292225201, 50.086409568983285],
        [43.41823056300268, 53.931119818519235],
        [43.69973190348525, 56.852753144978266],
        [46.635388739946336, 75.45308310991928],
        [44.38337801608579, 60.695401113631675],
        [46.394101876675606, 62.68508970921839],
        [47.39946380697051, 62.52608785316559],
        [48.96782841823056, 62.82573726541554],
        [47.721179624664884, 66.67828418230563],
        [45.71045576407507, 67.61167250979584],
        [45.911528150134046, 70.53371829243142],
        [46.394101876675606, 71.60816663229531],
        [46.394101876675606, 73.60816663229531],
        [49.410187667560294, 76.20808414105996],
        [47.35924932975871, 71.60321715817693],
        [47.07774798927614, 80.37389152402547],
        [47.4798927613941, 81.60259847391202],
        [47.56032171581769, 81.6021860177355],
        [43.41823056300268, 34.392658280057745],
        [50.57640750670241, 74.5097958341926],
        [50.817694369973196, 77.8931738502784],
        [50.6970509383378, 77.8937925345432],
        [51.139410187667565, 78.50690864095688],
        [50.93833780160858, 80.04640131985977],
        [50.93833780160858, 80.04640131985977],
        [48.96782841823056, 82.21035265003093],
        [48.96782841823056, 81.28727572695401],
        [48.16353887399464, 83.29140028871932],
        [49.77211796246649, 81.89853578057331],
        [48.88739946380697, 85.28768818313054],
        [49.77211796246649, 85.28315116518868],
        [49.611260053619304, 86.8224376160033],
        [53.552278820375335, 78.80222726335326],
        [53.51206434316354, 83.26397195298],
        [53.270777479892764, 88.64982470612497],
        [52.62734584450402, 89.26850897092183],
        [50.656836461126005, 90.50938337801608],
        [50.25469168900804, 87.58836873582182],
        [50.656836461126005, 87.27861414724686],
        [50.656836461126005, 85.1247679934007],
        [50.6970509383378, 84.66302330377397],
        [51.46112600536193, 89.58218189317384],
        [51.702412868632706, 90.96555990925964],
        [53.9142091152815, 92.0311404413281],
        [53.632707774798924, 92.34027634563827],
        [54.11528150134048, 93.87626314704062],
        [54.316353887399465, 94.64446277583006],
        [53.47184986595175, 94.8026397195298],
        [53.552278820375335, 96.64838110950711],
        [53.793565683646115, 96.95483604866982],
        [54.67828418230563, 99.71952979995875],
        [55.72386058981233, 96.02186017735615],
        [56.1260053619303, 96.3274902041658],
        [56.447721179624665, 99.71045576407506],
        [56.76943699731903, 99.70880593936894],
        [57.252010723860586, 97.8601773561559],
        [57.493297587131366, 99.70509383378015],
        [57.815013404825734, 99.39575170138173],
        [57.01072386058981, 99.70756857083934],
        [57.01072386058981, 99.09218395545473],
        [58.860589812332435, 99.08269746339451],
        [58.538873994638074, 96.31511651886987],
        [59.343163538873995, 97.84945349556608],
        [59.66487935656836, 95.69395751701381],
        [59.82573726541555, 96.61620952773768],
        [59.98659517426274, 95.69230769230768],
        [63.68632707774799, 85.5194885543411],
        [63.12332439678284, 86.13776036296143],
        [63.766756032171585, 86.7498453289338],
        [64.49061662198392, 86.74613322334503],
        [63.68632707774799, 89.51948855434108],
        [63.64611260053619, 85.0581563208909],
        [60.54959785522789, 99.53557434522581],
        [61.35388739946381, 94.76221901422973],
        [62.31903485254691, 96.44957723241905],
        [61.35388739946381, 91.83914209115281],
        [60.54959785522789, 93.53557434522583],
        [59.343163538873995, 91.08022272633532],
        [58.41823056300268, 90.46958135698081],
        [57.815013404825734, 91.08805939368942],
        [62.19839142091153, 89.21942668591461],
        [61.4343163538874, 87.68488348113013],
        [61.27345844504021, 84.30109300886781],
        [61.4343163538874, 79.37719117343781],
        [61.4343163538874, 76.91565271189936],
        [62.56032171581769, 79.67910909465868],
        [63.44504021447721, 79.67457207671684],
        [63.36461126005362, 80.5980614559703],
        [63.847184986595174, 80.13404825737265],
        [64.49061662198392, 81.82305630026809],
        [63.92761394101876, 76.13363580119612],
        [65.09383378016085, 76.12765518663642],
        [65.57640750670241, 72.74056506496184],
        [66.09919571045576, 71.96865333058363],
        [66.38069705093834, 75.50567127242729],
        [65.69705093833781, 68.12456176531244],
        [65.7372654155496, 66.12435553722416],
        [65.37533512064343, 63.97236543617241],
        [66.8632707774799, 66.11858115075273],
        [67.18498659517425, 63.04000824912353],
        [67.78820375335121, 62.42153021241492],
        [68.39142091152814, 59.49535986801402],
        [63.36461126005362, 59.21344607135491],
        [63.36461126005362, 52.751907609816456],
        [64.73190348525469, 53.514126624046185],
        [64.7721179624665, 51.821612703650246],
        [65.69705093833781, 47.81686945762013],
        [67.50670241286863, 51.65374303980202],
        [67.02412868632707, 42.425448546091985],
        [67.38605898123325, 35.50051557022066],
        [70.56300268096516, 48.4073004743246],
        [69.39678284182305, 41.490204165807384],
        [69.43699731903484, 41.64384409156527],
        [73.41823056300268, 34.392658280057745],
        [71.40750670241286, 34.40296968447103],
        [70.20107238605898, 29.48607960404206],
        [70.36193029490616, 29.485254691689008],
        [68.55227882037534, 27.18684264796866],
        [72.41286863270777, 24.243967828418235],
        [72.97587131367293, 25.47184986595174],
        [73.41823056300268, 18.238812126211585],
        [73.41823056300268, 17.469581356980825],
        [73.41823056300267, 12.238812126211585],
        [74.42359249329758, 12.54134873169724],
        [75.3887399463807, 12.382553103732718],
        [75.2278820375335, 17.46030109300888],
        [76.39410187667562, 8.22355124767995],
        [66.38069705093834, 55.65951742627347]
    ])

    TAActive = TAActive/[54.67828418230563, 99.71952979995875]

    lengthA = TAActive[:,0]
    forceA = TAActive[:,1]
    
    centres = np.arange(min(lengthA), max(lengthA), 0.1)
    width = .15
    result = Regression(lengthA, forceA, centres, width, .1, sigmoids=False)
    
    return result



force_length_regression = get_muscle_force_length_regression()
force_velocity_regression = get_muscle_force_velocity_regression()


def force_length_muscle(lm):
    """
    :param lm: muscle (contracile element) length
    :return: force-length scale factor
    """
    return force_length_regression.eval(lm)


def force_velocity_muscle(vm):
    """
    :param vm: muscle (contractile element) velocity)
    :return: force-velocity scale factor
    """
    return np.maximum(0, force_velocity_regression.eval(vm))
lm = np.arange(0, 1.8, 1.8/100)
lt = np.arange(0, 1.07, 1.07/100)

a = np.arange(0, 1, 1/100)
plot_curves()
print (get_velocity(1,1,1.01))  #-0.55451055
myMuscle = HillTypeMuscle(100, .3, .1)

def finding_lm(t, x):
    if t < 0.5:
        return get_velocity(0, x, myMuscle.norm_tendon_length(0.4, x))
    else:
        return get_velocity(1, x, myMuscle.norm_tendon_length(0.4, x))

contractile_length = solve_ivp(finding_lm, [0,2], [1], max_step = 0.01)


plt.figure()
plt.subplot(1,2,1)
plt.plot(contractile_length.t, contractile_length.y.T)

plt.xlabel('Time (s)')
plt.ylabel('Contractile Element Length')
plt.subplot(1,2,2)
plt.plot(contractile_length.t, myMuscle.get_force(0.4, contractile_length.y.T))

plt.xlabel('Time (s)')
plt.ylabel('Force produced by Muscle')
plt.tight_layout()
plt.show()
