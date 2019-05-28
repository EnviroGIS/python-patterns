from time import time
import pyproj
import geopandas as gpd
import numba
import math
import numpy as np
from sklearn.neighbors import BallTree as skBallTree
from lib.myvincenty import vincenty, get_min_distances, brute_min
# from shapely.geometry import Point
from functools import partial
from shapely.ops import transform

LEVELS_UP = 2

# ----------------------------------------------------------------------
# Distance computations


@numba.njit
def distance_to_node(node_centroids, node_radius, i_node, X, j):
    d = vincenty(node_centroids[i_node], X[j])
    return max(0, d - node_radius[i_node])


# ----------------------------------------------------------------------
# Heap for distances and neighbors


@numba.njit
def heap_create(N, k):
    distances = np.full((N, k), np.finfo(np.float32).max)
    indices = np.zeros((N, k), dtype=np.int32)
    return distances, indices


def heap_sort(distances, indices):
    i = np.arange(len(distances), dtype=np.int32)[:, None]
    j = np.argsort(distances, 1)
    return distances[i, j], indices[i, j]


@numba.njit
def heap_push(row, val, i_val, distances, indices):
    size = distances.shape[1]

    # check if val should be in heap
    if val > distances[row, 0]:
        return

    # insert val at position zero
    distances[row, 0] = val
    indices[row, 0] = i_val

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= size:
            break
        elif ic2 >= size:
            if distances[row, ic1] > val:
                i_swap = ic1
            else:
                break
        elif distances[row, ic1] >= distances[row, ic2]:
            if val < distances[row, ic1]:
                i_swap = ic1
            else:
                break
        else:
            if val < distances[row, ic2]:
                i_swap = ic2
            else:
                break

        distances[row, i] = distances[row, i_swap]
        indices[row, i] = indices[row, i_swap]

        i = i_swap

    distances[row, i] = val
    indices[row, i] = i_val


# ----------------------------------------------------------------------
# Tools for building the tree

@numba.njit
def node_id_to_range(node_id, n):
    level = math.floor(math.log(node_id + 1) / math.log(2))
    step = n / (2 ** level)
    pos = node_id - 2 ** level + 1
    idx_start = math.floor(pos * step)
    idx_end = math.floor((pos + 1) * step)

    return idx_start, idx_end


@numba.njit
def _recursive_build(i_node, data, node_centroids,
                     node_radius, idx_array, node_idx, n_nodes, leaf_size):

    idx_start, idx_end = node_id_to_range(i_node, data.shape[0])

    # determine Node centroid
    for j in range(data.shape[1]):
        node_centroids[i_node, j] = 0
        for i in range(idx_start, idx_end):
            node_centroids[i_node, j] += data[idx_array[i], j]
        node_centroids[i_node, j] /= (idx_end - idx_start)

    # determine Node radius
    radius = 0.0
    for i in range(idx_start, idx_end):
        dist = vincenty(node_centroids[i_node], data[idx_array[i]])
        if dist > radius:
            radius = dist

    # set node properties
    node_radius[i_node] = radius
    node_idx[i_node] = idx_start, idx_end

    i_child = 2 * i_node + 1

    # recursively create subnodes
    if i_child + 1 >= n_nodes:
        if idx_end - idx_start > 2 * leaf_size:
            print("Memory layout is flawed: not enough nodes allocated")

    elif idx_end - idx_start < 2:
            print("Memory layout is flawed: not enough nodes allocated")

    else:
        _, n_mid = node_id_to_range(i_child, data.shape[0])

        _partition_indices(data, idx_array, idx_start, idx_end, n_mid)

        _recursive_build(i_child, data, node_centroids,
                         node_radius, idx_array, node_idx, n_nodes, leaf_size)
        _recursive_build(i_child + 1, data, node_centroids,
                         node_radius, idx_array, node_idx, n_nodes, leaf_size)


@numba.njit
def _partition_indices(data, idx_array, idx_start, idx_end, split_index):
    # Find the split dimension
    n_features = data.shape[1]

    split_dim = 0
    max_spread = 0

    for j in range(n_features):
        max_val = -np.inf
        min_val = np.inf
        for i in range(idx_start, idx_end):
            val = data[idx_array[i], j]
            max_val = max(max_val, val)
            min_val = min(min_val, val)
        if max_val - min_val > max_spread:
            max_spread = max_val - min_val
            split_dim = j

    # Partition using the split dimension
    left = idx_start
    right = idx_end - 1

    while True:
        midindex = left
        for i in range(left, right):
            d1 = data[idx_array[i], split_dim]
            d2 = data[idx_array[right], split_dim]
            if d1 < d2:
                tmp = idx_array[i]
                idx_array[i] = idx_array[midindex]
                idx_array[midindex] = tmp
                midindex += 1
        tmp = idx_array[midindex]
        idx_array[midindex] = idx_array[right]
        idx_array[right] = tmp
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1


# ----------------------------------------------------------------------
# Tools for querying the tree
@numba.njit
def _query_recursive(i_node, X, i_pt, heap_distances, heap_indices,
                     dist_to_node, data, idx_array, node_centroids,
                     node_radius, node_idx, use_gpu):
    i1 = 2 * i_node + 1
    i2 = i1 + 1
    # ------------------------------------------------------------
    # Case 1: query point is outside node radius:
    #         trim it from the query
    if dist_to_node > heap_distances[idx_array[i_pt], 0]:
        pass

    # ------------------------------------------------------------
    # Case 2: this is a leaf node.  Update set of nearby points
    elif i2 >= node_centroids.shape[0]:
        go_deeper = True
        if use_gpu:
            step = len(X) / ((node_centroids.shape[0] + 1) // (2 ** LEVELS_UP))
            idx_start = math.floor((math.ceil((i_pt + 1) / step) - 1) * step)
            idx_end = math.floor((math.ceil((i_pt + 1) / step)) * step)

            if not (
                idx_start != node_idx[i_node, 0] and
                idx_end != node_idx[i_node, 1]
            ):
                go_deeper = False

        if go_deeper:
            for i in range(node_idx[i_node, 0],
                           node_idx[i_node, 1]):
                dist_pt = vincenty(data[idx_array[i]], X[idx_array[i_pt]])
                if dist_pt < heap_distances[idx_array[i_pt], 0]:
                    heap_push(idx_array[i_pt], dist_pt, idx_array[i],
                              heap_distances, heap_indices)

    # ------------------------------------------------------------
    # Case 3: Node is not a leaf.  Recursively query subnodes
    #         starting with the closest
    else:
        dist1 = distance_to_node(
            node_centroids, node_radius, i1, X, idx_array[i_pt])

        dist2 = distance_to_node(
            node_centroids, node_radius, i2, X, idx_array[i_pt])

        # recursively query subnodes
        if dist1 > dist2:
            i1, i2 = i2, i1
            dist1, dist2 = dist2, dist1

        _query_recursive(i1, X, i_pt, heap_distances, heap_indices,
                         dist1, data, idx_array, node_centroids,
                         node_radius, node_idx, use_gpu)

        _query_recursive(i2, X, i_pt, heap_distances, heap_indices,
                         dist2, data, idx_array, node_centroids,
                         node_radius, node_idx, use_gpu)


@numba.njit(parallel=True)
def _query_parallel(i_node, X, heap_distances, heap_indices, data, idx_array,
                    node_centroids, node_radius, node_idx, use_gpu):
    for i in numba.prange(len(idx_array)):
        i_pt = idx_array[i]
        sq_dist_LB = distance_to_node(
                node_centroids, node_radius, i_node, X, idx_array[i_pt]
        )
        _query_recursive(i_node, X, i_pt, heap_distances, heap_indices,
                         sq_dist_LB, data, idx_array, node_centroids,
                         node_radius, node_idx, use_gpu)


class BallTree:
    def __init__(self, data, leaf_size=40):
        self.data = data
        self.leaf_size = leaf_size

        # validate data
        if self.data.size == 0:
            raise ValueError("X is an empty array")

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")

        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]

        # determine number of levels in the tree, and from this
        # the number of nodes in the tree.  This results in leaf nodes
        # with numbers of points betweeen leaf_size and 2 * leaf_size
        self.n_levels = int(
                1 + np.log2(max(1, ((self.n_samples - 1) // self.leaf_size)))
        )
        self.n_nodes = int(2 ** self.n_levels) - 1

        # allocate arrays for storage
        self.idx_array = np.arange(self.n_samples, dtype=np.int32)
        self.node_radius = np.zeros(self.n_nodes, dtype=np.float32)
        self.node_idx = np.zeros((self.n_nodes, 2), dtype=np.int32)
        self.node_centroids = np.zeros((self.n_nodes, self.n_features),
                                       dtype=np.float32)

        # Allocate tree-specific data from TreeBase
        _recursive_build(0, self.data, self.node_centroids,
                         self.node_radius, self.idx_array, self.node_idx,
                         self.n_nodes, self.leaf_size)

    def pre_calculate_with_gpu(self, X, heap_distances, heap_indices):
        min_distances, min_indexes = get_min_distances(
                X,
                self.idx_array,
                np.float32(len(X) / ((self.n_nodes + 1) // (2 ** LEVELS_UP)))
        )

        heap_distances[:, 0] = min_distances
        heap_distances[:, 1] = 0

        heap_indices[:, 0] = min_indexes
        heap_indices[:, 1] = np.arange(self.n_samples)

    def query(self, X, k=1, sort_results=True, use_gpu=True):
        X = np.asarray(X, dtype=np.float32)

        if X.shape[-1] != self.n_features:
            raise ValueError("query data dimension must "
                             "match training data dimension")

        if self.data.shape[0] < k:
            raise ValueError("k must be less than or equal "
                             "to the number of training points")

        # flatten X, and save original shape information
        Xshape = X.shape
        X = X.reshape((-1, self.data.shape[1]))

        # initialize heap for neighbors
        heap_distances, heap_indices = heap_create(X.shape[0], k)

        if use_gpu:
            self.pre_calculate_with_gpu(X, heap_distances, heap_indices)

        _query_parallel(0, X, heap_distances, heap_indices, self.data,
                        self.idx_array, self.node_centroids, self.node_radius,
                        self.node_idx, use_gpu)

        distances, indices = heap_sort(heap_distances, heap_indices)

        # deflatten results
        return (distances.reshape(Xshape[:-1] + (k,)),
                indices.reshape(Xshape[:-1] + (k,)))


# ----------------------------------------------------------------------
# Testing function

def point_to_circle(row):
    point = row['geometry']
    local_azimuthal_projection = \
        f"+proj=aeqd +R=6371000 +units=m +lat_0={point.y} +lon_0={point.x}"

    wgs84_to_aeqd = partial(
        pyproj.transform,
        pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
        pyproj.Proj(local_azimuthal_projection),
    )

    aeqd_to_wgs84 = partial(
        pyproj.transform,
        pyproj.Proj(local_azimuthal_projection),
        pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
    )

    point_transformed = transform(wgs84_to_aeqd, point)

    circle = point_transformed.buffer(row['radius'])

    return transform(aeqd_to_wgs84, circle)


def test_tree(K=2, LS=3):
    df = gpd.read_file('../datasets/POINT/UK.geojson')
    X = np.stack(df['geometry']).astype(np.float32)

    print("-------------------------------------------------------")
    print("{0} neighbors of {1} points".format(K, len(X)))

    # pre-run to jit compile the code
    BallTree(X, leaf_size=LS).query(X, K)

    t0 = time()
    bt1 = skBallTree(X, leaf_size=LS, metric=vincenty)
    t1 = time()
    dist1, ind1 = bt1.query(X, K)
    t2 = time()

    bt2 = BallTree(X, leaf_size=LS)
    t3 = time()

#     geometry = gpd.GeoSeries(map(Point, bt2.node_centroids))
#     df = gpd.GeoDataFrame(geometry=geometry, crs={'init': 'epsg:4326'})
#     df['node_idx_start'] = bt2.node_idx[:, 0]
#     df['node_idx_end'] = bt2.node_idx[:, 1]
#     df['idx'] = np.arange(bt2.n_nodes)
#     df['radius'] = bt2.node_radius
#     df['geometry'] = df.apply(point_to_circle, axis=1)
#     df.to_file('nodes4_.geojson', driver='GeoJSON')

    dist2, ind2 = bt2.query(X, K)
    t4 = time()

    t5 = time()

    brute_dist = brute_min(X)

    t6 = time()
    print('Brute dist = sklearn:', np.allclose(dist1[:, 1], brute_dist))
    print('Brute dist = my dist:', np.allclose(dist2[:, 1], brute_dist))
    print()

    print('My dist = sklearn:', np.allclose(dist1, dist2))
    print('My index = sklearn', np.allclose(ind1, ind2, rtol=0))
    print()
    print("sklearn build: {0:.3g} sec".format(t1 - t0))
    print("numba build  : {0:.3g} sec".format(t3 - t2))
    print()
    print("sklearn query: {0:.3g} sec".format(t2 - t1))
    print("numba query  : {0:.3g} sec".format(t4 - t3))
    print("brute query  : {0:.3g} sec".format(t6 - t5))


if __name__ == '__main__':
    test_tree()
