from annoy import AnnoyIndex
import time
import numpy as np


def exact_nn(query, ds, size=10000):
    min_idx, min_dist = 0, np.inf
    for i in range(size):
        dist = np.linalg.norm(query - ds[i])
        if dist < min_dist:
            min_idx = i

    return min_idx


if __name__ == '__main__':
    m = [100, 300, 500, 700]
    trees = [1, 10, 50, 100, 1000]

    stats = {1: [], 10: [], 50: [], 100: [], 1000: []}
    for di in m:
        ds = np.random.random_integers(100, size=(10000, di))

        query = np.random.random_integers(100, size=(100, di))

        ann_indexer = AnnoyIndex(di, metric='euclidean')

        for t in trees:
            build = time.time()
            for i in range(10000):
                ann_indexer.add_item(i, ds[i])
            ann_indexer.build(t)
            build_finish = time.time() - build

            print('Dimensions: {}, Trees: {}, Build Time: {}'.format(di, t, build_finish))

            ann_anwser = []
            ann_query_start = time.time()
            for i in range(100):
                ann_anwser.append(ann_indexer.get_nns_by_vector(query[i], 1)[0])
            ann_query_end = time.time() - ann_query_start

            exact_anwser = []
            exact_query_start = time.time()
            for i in range(100):
                exact_anwser.append(exact_nn(query[i], ds))
            exact_query_end = time.time() - exact_query_start

            ann_dist = []
            exact_dist = []

            for i in range(100):
                ann_dist.append(np.linalg.norm(query[i] - ann_anwser[i]))
                exact_dist.append(np.linalg.norm(query[i] - exact_anwser[i]))

            ann_dist = np.array(ann_dist)
            exact_dist = np.array(exact_dist)
            acc = np.sum(np.abs(ann_dist - exact_dist))

            stats[t].append([di, build_finish])
            stats[t].append([di, ann_query_end])
            stats[t].append([di, exact_query_end])
            stats[t].append([di, acc])
            print('Dimensions: {}, Trees: {}, Ann Query Time: {}'.format(di, t, ann_query_end))
            print('Dimensions: {}, Trees: {}, Exact Query Time: {}'.format(di, t, exact_query_end))
            print('Dimensions: {}, Trees: {}, Acc: {}'.format(di, t, acc))





