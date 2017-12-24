import numpy as np
import os
from tqdm import trange

try:
    from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
except ImportError:
    from nupic_stub import EncoderStub as RandomDistributedScalarEncoder


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

        data_dir = os.path.join("data", "encoder", type(self).__name__)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.data_path = "size={}_sparse={}_bins={}_similar={}.npy".format(size, sparsity, bins, similarity)
        self.data_path = os.path.join(data_dir, self.data_path)
        if not os.path.exists(self.data_path):
            self.generate_sdr()
        self.sdr = np.load(self.data_path)

    def generate_sdr(self):
        self.sdr = np.zeros((self.bins+1, self.size), dtype=bool)  # 1 more bin to handle upper value 1.0
        n_active_total = max(int(self.size * self.sparsity), 1)
        n_active_stay = int(n_active_total * self.similarity)
        n_active_new = n_active_total - n_active_stay
        mask_active = np.random.choice(self.size, size=n_active_total, replace=False)
        self.sdr[0][mask_active] = 1
        dim_arange = np.arange(self.size)
        for bin_id in trange(1, len(self.sdr), desc="Generating ScalarEncoder SDR bins"):
            active_prev = np.nonzero(self.sdr[bin_id - 1])[0]
            active_stay = np.random.choice(active_prev, size=n_active_stay, replace=False)
            non_active = np.delete(dim_arange, active_prev)
            active_new = np.random.choice(non_active, size=n_active_new, replace=False)
            self.sdr[bin_id][active_stay] = 1
            self.sdr[bin_id][active_new] = 1
        for sdr_bin in self.sdr:
            assert len(np.nonzero(sdr_bin)[0]) == n_active_total
        np.save(self.data_path, self.sdr)

    def _to_bin(self, scalar):
        """
        :param scalar: input value
        :return: corresponding SDR bin index
        """
        raise NotImplementedError("Implement in subclass")

    def _to_scalar(self, bin_index):
        """
        :param bin_index: SDR bin index
        :return: corresponding scalar
        """
        raise NotImplementedError("Implement in subclass")

    def encode(self, scalar):
        """
        :param scalar: input value
        :return: sparse distributed representation vector of `scalar`
        """
        bin_active = self._to_bin(scalar)
        return self.sdr[bin_active].astype(np.int32)

    def decode(self, sdr):
        """
        :param sdr: input sdr, encoded as a scalar
        :return: decoded scalar value (best fit)
        """
        score_table = self.sdr.dot(sdr)
        bin_matched = np.argmax(score_table)
        return self._to_scalar(bin_matched)


class FloatEncoder(ScalarEncoder):

    def _to_bin(self, scalar):
        """
        :param scalar: float value in range [0, 1]
        :return: corresponding SDR bin index
        """
        assert 0 <= scalar <= 1, "Illegal value: {}".format(scalar)
        return int(scalar * self.bins)

    def _to_scalar(self, bin_index):
        """
        :param bin_index: SDR bin index
        :return: corresponding scalar
        """
        return float(bin_index) / self.bins


class IntEncoder(ScalarEncoder):

    def _to_bin(self, scalar):
        """
        :param scalar: int value in range [0, #bins]
        :return: corresponding SDR bin index
        """
        assert isinstance(scalar, int)
        assert 0 <= scalar <= self.bins, "Illegal value: {}".format(scalar)
        return scalar

    def _to_scalar(self, bin_index):
        """
        :param bin_index: SDR bin index
        :return: corresponding scalar
        """
        return bin_index


class LocationEncoder(object):
    def __init__(self, max_amplitude):
        self.max_amplitude = float(max_amplitude)
        self.scalar_encoder = FloatEncoder(size=100, sparsity=0.1, bins=100, similarity=0.8)
        # self.scalar_encoder = RandomDistributedScalarEncoder(resolution=0.01, w=11, n=100)

    def encode_amplitude(self, vector):
        ampl = np.linalg.norm(vector)
        ampl = ampl / self.max_amplitude
        ampl_encoded = self.scalar_encoder.encode(ampl)
        return ampl_encoded

    def encode_phase(self, vector):
        x, y = vector
        phase = np.arctan2(y, x)
        # transform [-pi, pi] --> [0, 1]
        phase = (phase / np.pi + 1.) / 2.
        phase_encoded = self.scalar_encoder.encode(phase)
        return phase_encoded


if __name__ == '__main__':
    encoder = FloatEncoder()
    sdr1 = encoder.encode(0.47)
    sdr2 = encoder.encode(0.48)
    overlap = np.count_nonzero(sdr1*sdr2)
    overlap_expected = int(encoder.bins * encoder.sparsity * encoder.similarity)
    print(sdr1.astype(int))
    print("Similarity overlap={} (expected={})".format(overlap, overlap_expected))
