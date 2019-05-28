"""
This code is modification of vincenty_inverse function from
https://github.com/maurycyp/vincenty

Changes were made to make possible to run it on numba
"""

import math
from numba import njit, prange, guvectorize

# WGS 84
a = 6378137  # meters
f = 1 / 298.257223563
b = 6356752.314245  # meters; b = (1 - f)a


@njit(parallel=True)
def vincenty(point1, point2):
    MAX_ITERATIONS = 200
    CONVERGENCE_THRESHOLD = 1e-12  # .000,000,000,001

    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0.0

    U1 = math.atan((1 - f) * math.tan(math.radians(point1[1])))
    U2 = math.atan((1 - f) * math.tan(math.radians(point2[1])))
    L = math.radians((point2[0] - point1[0]))
    Lambda = L

    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)
    sinU2 = math.sin(U2)
    cosU2 = math.cos(U2)

    for iteration in prange(MAX_ITERATIONS):
        sinLambda = math.sin(Lambda)
        cosLambda = math.cos(Lambda)
        sinSigma = math.sqrt((cosU2 * sinLambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
        if sinSigma == 0:
            return 0.0  # coincident points

        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2

        if cosSqAlpha == 0:
            cos2SigmaM = 0
        else:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha

        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM ** 2)))
        if abs(Lambda - LambdaPrev) < CONVERGENCE_THRESHOLD:
            break  # successful convergence
    else:
        return 0.0  # failure to converge

    uSq = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (
        cosSigma *
        (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM *
        (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)
    ))

    return b * A * (sigma - deltaSigma)


@guvectorize('void(int64, float32[:,:], float32[:])', '(),(n, m) -> ()',
             target='parallel', cache=True)
def get_min_distances(idx, points, result):
    l = len(points)
    min_dist = -1

    for i in prange(l):
        if i != idx:
            distance = vincenty(points[idx], points[i])

            if (min_dist == -1) or (distance < min_dist):
                min_dist = distance

    result[0] = min_dist