#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate quadrature points and weights for Gauss-Lobatto rules.
For disc integration, the range of cos(emission angle) is [0,1], and we choose
to not have quadurature point at 0.
See https://mathworld.wolfram.com/LobattoQuadrature.html
"""
from scipy.special import eval_legendre,legendre
import numpy as np
from functools import partial

def bisect(func, lower, upper, tol=1e-15):
    """Find root of function by bisection.

    Parameters
    ----------
    func : function
        Some pre-defined function.
    lower : real
        Lower bound of the root.
    upper : real
        Upper bound of the root.
    tol : real
        Error tolerance. Defaults to 1e-15.

    Returns
    -------
    root : real
        Root of the function between lower bound and upper bound.
    """
    assert func(lower) * func(upper) < 0
    while upper-lower > tol:
        median = lower + (upper-lower)/2
        f_median = func(median)
        if func(lower) * f_median < 0:
            upper = median
        else:
            lower = median
    root = median
    return root

def calc_legendre_derivative(n, x):
    """
    Calculate the derivative of degree n Legendre polynomial at x.

    Parameters
    ----------
    n : int
        Degree of the Legendre polynomia P_n(x).
    x : real
        Point at which to evaluate the derivative

    Returns
    -------
    result : real
        Derivative of P_n at x, i.e., P'_n(x).
    """
    result = (x*eval_legendre(n, x) - eval_legendre(n-1, x))\
        /((x**2-1)/n)
    return result


def lobatto(n):
    """
    Generate points and weights for Lobatto quadrature.

    Parameters
    ----------
    n : int
        Number of quadrature points apart from the end points -1 and 1.

    Returns
    -------
    points : ndarray
        Lobatto quadrature points.
    weights : ndarray
        Lobatto quadrature weights
    """
    assert n >= 1
    brackets = legendre(n-1).weights[:, 0]
    points = np.zeros(n)
    points[0] = -1
    points[-1] = 1
    for i in range(n-2):
        points[i+1] = bisect(
            partial(calc_legendre_derivative, n-1),
            brackets[i], brackets[i+1])
    points = np.around(points,decimals=14)
    weights = np.zeros(n)
    weight_end_pts = 2 / (n*(n-1))
    weights[0] = weight_end_pts
    weights[-1] = weight_end_pts
    for i in range(1,n-1):
        weights[i] = 2 / ( (n*(n-1)) * eval_legendre(n-1,points[i])**2 )
    return points, weights

def disc_weights(n):
    """
    Generate weights for disc integration in the the emission angle
    direction.

    Parameters
    ----------
    n : int
        Number of emission angles. Minuim 2.

    Returns:
    mu : ndarray
        List of cos(emission angle) for integration.
    wtmu : ndarray
        List of weights for integration.
    """
    assert n >= 2
    points, weights = lobatto(2*n)
    mu = np.zeros(n)
    wtmu = np.zeros(n)
    for i in range(n):
        mu[i] = abs(points[i])
        wtmu[i] = weights[i]
    mu = mu[::-1]
    wtmu = wtmu[::-1]
    return mu,wtmu

# print('lobatto weights')
# for i in range(2,16):
#     mu, wtmu = lobatto(i)
#     print('order ', i)
#     print(list(mu))
#     print(list(wtmu))

# print('disc average')
# for i in range(2,16):
#     mu, wtmu = disc_weights(i)
#     print('order ', i)
#     print(list(mu))
#     print(list(wtmu))

"""
disc average
order  2
[0.44721359549996, 1.0]
[0.8333333333333333, 0.16666666666666666]
order  3
[0.28523151648065, 0.76505532392946, 1.0]
[0.5548583770354862, 0.37847495629784705, 0.06666666666666667]
order  4
[0.20929921790248, 0.59170018143314, 0.87174014850961, 1.0]
[0.41245879465870366, 0.34112269248350424, 0.21070422714350606, 0.03571428571428571]
order  5
[0.16527895766639, 0.47792494981044, 0.7387738651055, 0.91953390816646, 1.0]
[0.32753976118389705, 0.2920426836796838, 0.22488934206312644, 0.1333059908510701, 0.022222222222222223]
order  6
[0.13655293285493, 0.39953094096535, 0.63287615303186, 0.81927932164401, 0.94489927222288, 1.0]
[0.27140524091069596, 0.2512756031992008, 0.21250841776102114, 0.15797470556437015, 0.09168451741319596, 0.015151515151515152]
order  7
[0.1163318688837, 0.34272401334271, 0.55063940292865, 0.72886859909133, 0.86780105383035, 0.95993504526726, 1.0]
[0.2316127944684567, 0.21912625300977043, 0.1948261493734163, 0.16002185176295194, 0.11658665589871156, 0.06683728449768114, 0.01098901098901099]
order  8
[0.10132627352195, 0.29983046890076, 0.48605942188714, 0.65238870288249, 0.79200829186182, 0.89920053309347, 0.96956804627022, 1.0]
[0.20195830817822982, 0.19369002382520328, 0.17749191339170406, 0.15402698080716404, 0.12425538213251425, 0.08939369732593098, 0.0508503610059198, 0.008333333333333333]
order  9
[0.08974909348465, 0.26636265287828, 0.43441503691212, 0.58850483431866, 0.72367932928324, 0.83559353521809, 0.92064918534753, 0.9761055574122, 1.0]
[0.17901586343970305, 0.1732621094894562, 0.16193951723760244, 0.14541196157380204, 0.12421053313296725, 0.09901627171750278, 0.07063716688563368, 0.03997062881091395, 0.006535947712418301]
order  10
[0.08054593723882, 0.23955170592299, 0.39235318371391, 0.53499286403189, 0.66377640229031, 0.77536826095206, 0.86687797808995, 0.93593449881267, 0.98074370489391, 1.0]
[0.1607432863878459, 0.15658010264747568, 0.14836155407091675, 0.1363004823587243, 0.12070922762867468, 0.10199149969945077, 0.08063176399611946, 0.0571818021275667, 0.03223712318848885, 0.005263157894736842]
order  11
[0.0730545400109, 0.21760658515929, 0.35752071013892, 0.4898148751899, 0.61166943828426, 0.7204872399612, 0.81394892761192, 0.8900622901909, 0.94720428399923, 0.98415243845765, 1.0]
[0.14584901944424192, 0.1427404922713613, 0.13658968861374166, 0.12752769665343047, 0.11574764465393902, 0.10150057480164791, 0.08509006039183853, 0.06686560586455312, 0.04721446529374087, 0.026545747682501883, 0.004329004329004329]
order  12
[0.06683799373723, 0.19932125339083, 0.32824761337551, 0.45131637321432, 0.56633135797929, 0.67124010526413, 0.76417048242049, 0.84346407015487, 0.90770567511351, 0.95574822092989, 0.98673055350516, 1.0]
[0.1334768438669862, 0.13109494187360402, 0.12637364202802084, 0.11939719370249148, 0.11029008689296854, 0.09921482768408361, 0.08636902996792918, 0.0719818620552941, 0.056309848724646276, 0.03963168133346778, 0.022236853464711336, 0.0036231884057971015]
order  13
[0.06159641178192, 0.18385549527006, 0.30332751285925, 0.41820138706625, 0.52673574202988, 0.62728529949232, 0.71832581636267, 0.79847718310744, 0.86652432395912, 0.92143554681756, 0.96237787476772, 0.98872741231148, 1.0]
[0.12303696380008272, 0.12117184628844326, 0.11746988409380896, 0.11198719411986023, 0.10480688623073697, 0.09603780235390127, 0.08581286398000451, 0.07428705012229121, 0.06163502514254753, 0.04804839908118083, 0.033732303685955645, 0.018896858024263646, 0.003076923076923077]
order  14
[0.05711712169351, 0.170606755308, 0.2818722666216, 0.38946313757636, 0.49197675393158, 0.58807668983718, 0.67651012892957, 0.75612419400557, 0.82588097005634, 0.8848710172113, 0.93232516712156, 0.96762428585713, 0.99030540261845, 1.0]
[0.11410997967262779, 0.11262238007723906, 0.10966657379597669, 0.10528109376105582, 0.09952311041249591, 0.09246768599771213, 0.08420679512151033, 0.07484812350970772, 0.06451365808035459, 0.05333807704732744, 0.041466915243006726, 0.029054220677979196, 0.01625588395750441, 0.0026455026455026454]
order  15
[0.05324511048549, 0.15913204262585, 0.26321594371957, 0.36431750042245, 0.46129119016824, 0.55303826009505, 0.63851917580756, 0.71676539863708, 0.78689035723755, 0.8480994871802, 0.89969921819928, 0.94110478095106, 0.97184660316627, 0.9915739428405, 1.0]
[0.10638955872366786, 0.10518412159645452, 0.102786905307235, 0.09922507100429963, 0.09453897519386084, 0.08878171231976516, 0.08201851283340691, 0.07432600332471831, 0.06579133639779007, 0.056511197923080486, 0.046590694533142844, 0.03614209419940861, 0.025283166740551526, 0.014131799327905363, 0.0022988505747126436]
"""

"""
lobatto weights
order  2
[-1.0, 1.0]
[1.0, 1.0]
order  3
[-1.0, 0.57735026918962, 1.0]
[0.3333333333333333, 3.3386682475146785e+27, 0.3333333333333333]
order  4
[-1.0, -0.44721359549996, 0.44721359549996, 1.0]
[0.16666666666666666, 0.8333333333333333, 0.8333333333333334, 0.16666666666666666]
order  5
[-1.0, -0.65465367070798, 0.33998104358486, 0.65465367070798, 1.0]
[0.1, 0.5444444444444448, 2.0440826005191908e+27, 0.5444444444444445, 0.1]
order  6
[-1.0, -0.76505532392946, -0.28523151648065, 0.28523151648065, 0.76505532392946, 1.0]
[0.06666666666666667, 0.37847495629784705, 0.5548583770354862, 0.5548583770354868, 0.37847495629784705, 0.06666666666666667]
order  7
[-1.0, -0.83022389627857, -0.46884879347071, 0.2386191860832, 0.46884879347071, 0.83022389627857, 1.0]
[0.047619047619047616, 0.27682604736156585, 0.43174538120986283, 1.109829392410592e+27, 0.43174538120986283, 0.27682604736156596, 0.047619047619047616]
order  8
[-1.0, -0.87174014850961, -0.59170018143314, -0.20929921790248, 0.20929921790248, 0.59170018143314, 0.87174014850961, 1.0]
[0.03571428571428571, 0.21070422714350606, 0.34112269248350424, 0.41245879465870366, 0.4124587946587041, 0.34112269248350435, 0.21070422714350598, 0.03571428571428571]
order  9
[-1.0, -0.89975799541146, -0.67718627951074, -0.36311746382618, 0.18343464249565, 0.36311746382618, 0.67718627951074, 0.89975799541146, 1.0]
[0.027777777777777776, 0.16549536156080588, 0.2745387125001619, 0.34642851097304606, 1.4085006669202548e+29, 0.34642851097304617, 0.2745387125001618, 0.16549536156080552, 0.027777777777777776]
order  10
[-1.0, -0.91953390816646, -0.7387738651055, -0.47792494981044, -0.16527895766639, 0.16527895766639, 0.47792494981044, 0.7387738651055, 0.91953390816646, 1.0]
[0.022222222222222223, 0.1333059908510701, 0.22488934206312644, 0.2920426836796838, 0.32753976118389705, 0.3275397611838977, 0.2920426836796838, 0.22488934206312652, 0.1333059908510701, 0.022222222222222223]
order  11
[-1.0, -0.93400143040806, -0.78448347366314, -0.5652353269962, -0.29575813558694, 0.14887433898163, 0.29575813558694, 0.5652353269962, 0.78448347366314, 0.93400143040806, 1.0]
[0.01818181818181818, 0.10961227326699495, 0.18716988178030547, 0.2480481042640284, 0.2868791247790081, 1.9505247402459908e+27, 0.2868791247790081, 0.2480481042640284, 0.1871698817803052, 0.10961227326699494, 0.01818181818181818]
order  12
[-1.0, -0.94489927222288, -0.81927932164401, -0.63287615303186, -0.39953094096535, -0.13655293285493, 0.13655293285493, 0.39953094096535, 0.63287615303186, 0.81927932164401, 0.94489927222288, 1.0]
[0.015151515151515152, 0.09168451741319596, 0.15797470556437015, 0.21250841776102114, 0.2512756031992008, 0.27140524091069596, 0.2714052409106964, 0.2512756031992014, 0.21250841776102114, 0.15797470556437015, 0.09168451741319614, 0.015151515151515152]
order  13
[-1.0, -0.95330984664216, -0.84634756465187, -0.68618846908176, -0.48290982109134, -0.24928693010624, 0.12523340851147, 0.24928693010624, 0.48290982109134, 0.68618846908176, 0.84634756465187, 0.95330984664216, 1.0]
[0.01282051282051282, 0.0778016867468189, 0.1349819266896083, 0.18364686520355022, 0.22076779356610987, 0.2440157903066763, 1.3033109179018366e+27, 0.24401579030667658, 0.2207677935661102, 0.18364686520355017, 0.13498192668960834, 0.07780168674681895, 0.01282051282051282]
order  14
[-1.0, -0.95993504526726, -0.86780105383035, -0.72886859909133, -0.55063940292865, -0.34272401334271, -0.1163318688837, 0.1163318688837, 0.34272401334271, 0.55063940292865, 0.72886859909133, 0.86780105383035, 0.95993504526726, 1.0]
[0.01098901098901099, 0.06683728449768114, 0.11658665589871156, 0.16002185176295194, 0.1948261493734163, 0.21912625300977043, 0.2316127944684567, 0.23161279446845717, 0.21912625300977076, 0.19482614937341605, 0.16002185176295222, 0.1165866558987116, 0.06683728449768128, 0.01098901098901099]
order  15
[-1.0, -0.96524592650384, -0.88508204422298, -0.76351968995181, -0.60625320546985, -0.42063805471367, -0.21535395536379, 0.10805494870734, 0.21535395536379, 0.42063805471367, 0.60625320546985, 0.76351968995181, 0.88508204422298, 0.96524592650384, 1.0]
[0.009523809523809525, 0.0580298930286012, 0.10166007032571801, 0.14051169980242792, 0.17278964725360094, 0.19698723596461337, 0.21197358592682083, 7.574387050252467e+25, 0.21197358592682092, 0.19698723596461337, 0.17278964725360088, 0.1405116998024281, 0.10166007032571808, 0.05802989302860128, 0.009523809523809525]
"""