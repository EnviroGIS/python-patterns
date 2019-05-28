"""
This code is modification of vincenty_inverse function from
https://github.com/maurycyp/vincenty

Changes were made to make possible to run it on numba
"""

import math
import numba
from numba import cuda
import numpy as np

# WGS 84
a = 6378137  # meters
f = 1 / 298.257223563
b = 6356752.314245  # meters; b = (1 - f)a

MAX_ITERATIONS = 200
CONVERGENCE_THRESHOLD = 1e-12  # .000,000,000,001


@numba.njit(parallel=True)
def vincenty(point1, point2):
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0.0

    U1 = math.atan((1 - f) * math.tan(point1[1] * math.pi / 180))
    U2 = math.atan((1 - f) * math.tan(point2[1] * math.pi / 180))
    L = (point2[0] - point1[0]) * math.pi / 180
    Lambda = L

    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)
    sinU2 = math.sin(U2)
    cosU2 = math.cos(U2)

    for iteration in numba.prange(MAX_ITERATIONS):
        sinLambda = math.sin(Lambda)
        cosLambda = math.cos(Lambda)
        sinSigma = math.sqrt((cosU2 * sinLambda)**2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda)**2)
        if sinSigma == 0:
            return 0.0  # coincident points

        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha**2

        if cosSqAlpha == 0:
            cos2SigmaM = 0
        else:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha

        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM**2)))
        if abs(Lambda - LambdaPrev) < CONVERGENCE_THRESHOLD:
            break  # successful convergence
    else:
        return 0.0  # failure to converge

    uSq = cosSqAlpha * (a**2 - b**2) / (b**2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (
        cos2SigmaM + B / 4 *
        (cosSigma * (-1 + 2 * cos2SigmaM**2) - B / 6 * cos2SigmaM *
         (-3 + 4 * sinSigma**2) * (-3 + 4 * cos2SigmaM**2)))

    return b * A * (sigma - deltaSigma)


#@cuda.jit('float32(float32, float32, float32, float32)', device=True)
def vincenty_inverse(lng0, lat0, lng1, lat1):
    if lng0 == lng1 and lat0 == lat1:
        return 0.0

    U1 = math.atan((1 - f) * math.tan(lat0 * math.pi / 180))
    U2 = math.atan((1 - f) * math.tan(lat1 * math.pi / 180))
    L = (lng1 - lng0) * math.pi / 180
    Lambda = L

    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)
    sinU2 = math.sin(U2)
    cosU2 = math.cos(U2)

    for iteration in numba.prange(MAX_ITERATIONS):
        sinLambda = math.sin(Lambda)
        cosLambda = math.cos(Lambda)
        sinSigma = math.sqrt((cosU2 * sinLambda)**2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda)**2)
        if sinSigma == 0:
            return 0.0  # coincident points

        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha**2

        if cosSqAlpha == 0:
            cos2SigmaM = 0
        else:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha

        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM**2)))
        if abs(Lambda - LambdaPrev) < CONVERGENCE_THRESHOLD:
            break  # successful convergence
    else:
        return 0.0  # failure to converge

    uSq = cosSqAlpha * (a**2 - b**2) / (b**2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (
        cos2SigmaM + B / 4 *
        (cosSigma * (-1 + 2 * cos2SigmaM**2) - B / 6 * cos2SigmaM *
         (-3 + 4 * sinSigma**2) * (-3 + 4 * cos2SigmaM**2)))

    return b * A * (sigma - deltaSigma)


#@cuda.jit('void(float32[:,:], float32[:], int32[:], float32, int32[:])')
def fast(array, res, idx, step, res2):
    x = cuda.grid(1)

    if x < res.shape[0]:
        idx_start = math.floor((math.ceil((x + 1) / step) - 1) * step)
        idx_end = math.floor((math.ceil((x + 1) / step)) * step)

        for y in range(idx_start, idx_end):
            if y != x:
                p1 = array[idx[x]]
                p2 = array[idx[y]]

                dist = vincenty_inverse(p1[0], p1[1], p2[0], p2[1])
                if res[idx[x]] > dist:
                    res[idx[x]] = dist
                    res2[idx[x]] = idx[y]


def get_min_distances(array, idx, step):
    n = len(array)

    result = np.zeros(n, dtype=np.float32)
    result2 = np.zeros(n, dtype=np.int32)
    result[:] = np.inf
    # Configure the blocks
    # threadsperblock = (16, 16)
    # blockspergrid_x = int(math.ceil(n / threadsperblock[1]))
    # blockspergrid_y = int(math.ceil(n / threadsperblock[0]))
    # blockspergrid = (blockspergrid_x, blockspergrid_y)

    device_array = cuda.to_device(array)
    device_result = cuda.to_device(result)
    device_result2 = cuda.to_device(result2)
    device_idx = cuda.to_device(idx)

    # Start the kernel
    fast[int(math.ceil(n / 64)), 64](
        device_array, device_result, device_idx, step, device_result2
    )
    return device_result.copy_to_host(), device_result2.copy_to_host()


#@cuda.jit('void(float32[:,:], float32[:])')
def find_all(arr, res):
    x, y = cuda.grid(2)

    if x > y and (x < res.shape[0]) and (y < res.shape[0]):
        dist = vincenty_inverse(
                arr[x, 0], arr[x, 1],
                arr[y, 0], arr[y, 1])

        cuda.atomic.min(res, x, dist)
        cuda.atomic.min(res, y, dist)


def brute_min(array):
    n = len(array)

    result = np.zeros(n, dtype=np.float32)
    result[:] = np.inf

    device_array = cuda.to_device(array)
    device_result = cuda.to_device(result)

    # Configure the blocks
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(n / threadsperblock[1])
    blockspergrid_y = math.ceil(n / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    find_all[blockspergrid, threadsperblock](device_array, device_result)

    return device_result.copy_to_host()
