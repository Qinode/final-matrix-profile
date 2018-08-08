from paper.MPIII.util import discretization, rdl
import numpy as np

class Cluster(object):

    # patterns.shape = (window_size x n)
    # center.shape = (n, 1)
    # patterns_idx.shape = (n, )

    def __init__(self, size, center, patterns, patterns_idx):
        self.size = size
        self.center = center.squeeze()
        self.patterns = patterns
        self.patterns_idx = patterns_idx

    def description_length(self, t_min, t_max, bits):
        dl_h = self.center.shape[0] * bits

        max_c_h = -np.inf
        sum_c_h = 0

        d_center = discretization(self.center, t_min, t_max, bits)
        for i in range(self.size):
            d_pattern = discretization(self.patterns[:, i], t_min, t_max, bits)

            dl_c_h = rdl(d_pattern, d_center, bits)
            max_c_h = max(max_c_h, dl_c_h)
            sum_c_h += dl_c_h

        return dl_h - max_c_h + sum_c_h

    @staticmethod
    def add_pattern(a_cluster, a_pattern, a_pattern_idx):
        new_center = ((a_cluster.center * a_cluster.size + a_pattern.squeeze())/(a_cluster.size + 1)).squeeze()
        new_size = a_cluster.size + 1
        new_patterns = np.c_[a_cluster.patterns, a_pattern]
        new_patterns_idx = np.r_[a_cluster.patterns_idx, a_pattern_idx]

        return Cluster(new_size, new_center.squeeze(), new_patterns, new_patterns_idx)

    @staticmethod
    def merge_cluster(c1, c2):
        new_size = c1.size + c2.size
        new_patterns = np.c_[c1.patterns, c2.patterns]
        new_center = (c1.center * c1.size + c2.center * c2.size)/new_size
        new_patterns_idx = np.r_[c1.patterns_idx, c2.patterns_idx]

        return Cluster(new_size, new_center.squeeze(), new_patterns, new_patterns_idx)
