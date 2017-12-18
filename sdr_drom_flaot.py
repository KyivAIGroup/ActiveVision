import numpy as np
import os


class ScalarEncoder(object):
    def __init__(self, size=100, sparsity=0.1, bins=100, similarity=0.8):
        """
        Sparse Distributed Representation scalar encoder.
        :param size: sdr output vector dimensionality
        :param sparsity: sdr output vector sparsity
        :param bins: number of bins (1/resolution)
        :param similarity: how similar two adjacent encodings (bins)
        """
        self.size = size
        self.sparsity = sparsity
        self.bins = bins
        self.similarity = similarity

        data_dir = "data"
        self.data_path = "size={}_sparse={}_bins={}_similar={}.npy".format(size, sparsity, bins, similarity)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.data_path = os.path.join(data_dir, self.data_path)
        if not os.path.exists(self.data_path):
            self.generate_sdr()
        self.sdr = np.load(self.data_path)

    def generate_sdr(self):
        self.sdr = np.zeros((self.bins, self.size), dtype=bool)
        n_active_total = max(int(self.size * self.sparsity), 1)
        n_active_stay = int(n_active_total * self.similarity)
        n_active_new = n_active_total - n_active_stay
        mask_active = np.random.choice(self.size, size=n_active_total, replace=False)
        self.sdr[0][mask_active] = 1
        dim_arange = np.arange(self.size)
        for bin_id in range(1, self.bins):
            active_prev = np.nonzero(self.sdr[bin_id - 1])[0]
            active_stay = np.random.choice(active_prev, size=n_active_stay, replace=False)
            non_active = np.delete(dim_arange, active_prev)
            active_new = np.random.choice(non_active, size=n_active_new, replace=False)
            self.sdr[bin_id][active_stay] = 1
            self.sdr[bin_id][active_new] = 1
        for sdr_bin in self.sdr:
            assert len(np.nonzero(sdr_bin)[0]) == n_active_total
        np.save(self.data_path, self.sdr)

    def encode(self, scalar):
        """
        :param scalar: float value in range [0, 1) 
        :return: sparse distributed representation vector of `scalar`
        """
        bin_active = int(scalar * self.bins)
        return self.sdr[bin_active]


if __name__ == '__main__':
    encoder = ScalarEncoder()
    sdr1 = encoder.encode(0.47)
    sdr2 = encoder.encode(0.5)
    overlap = np.count_nonzero(sdr1*sdr2)
    overlap_expected = int(encoder.bins * encoder.sparsity * encoder.similarity)
    print(sdr1.astype(int))
    print("Similarity overlap={} (expected={})".format(overlap, overlap_expected))
