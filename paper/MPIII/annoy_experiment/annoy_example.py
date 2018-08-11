from annoy import AnnoyIndex
import random
import numpy as np
import time

def exact_1nn(t, n, query):
    min_dist = np.inf
    min_idx = -1
    for i in range(n):
        item = np.array(t.get_item_vector(i))
        dist = np.linalg.norm(query - item)
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    return min_idx, min_dist


if __name__ == '__main__':
    f = 400
    t = AnnoyIndex(f, metric='euclidean')  # Length of item vector that will be indexed
    for i in range(1000):
        v = [random.gauss(0, 1) for z in range(f)]
        t.add_item(i, v)

    t.build(100)  # 10 trees

    for _ in range(50):
        query = np.random.random(f)
        start = time.time()
        exact_nn_idx, exact_dist = exact_1nn(t, 1000, query)
        exact_time = time.time() - start
        start = time.time()
        ann = t.get_nns_by_vector(query, 1)[0]
        approximate_time = time.time() - start

        aitem = np.array(t.get_item_vector(ann))
        adist = np.linalg.norm(aitem- query)

        print('Exact NN: {}, in {}, dist {}, Approximate NN: {}, in {}, dist {}'.format(exact_nn_idx, exact_time, exact_dist, ann, approximate_time, adist))



