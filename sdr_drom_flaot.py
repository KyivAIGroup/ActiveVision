import numpy as np
import os
import pickle


class NumberEncoder(object):
    def __init__(self, sdr_size=100, sparsity=0.1, discretization=100, similarity=0.8):
        self.sdr_size = sdr_size
        self.sparsity = sparsity
        self.discretization = discretization # shows how many sdr should be
        self.similarity = similarity # shows how similar two adjacent encodings

        self.datafile = 'data/number_data_num_size_'+str(sdr_size) + '_' +str(discretization)+'_'+str(sparsity)+'.p'
        if not os.path.exists('data/'):
            os.makedirs('data/')
        try:
            with open(self.datafile) as file:
                data = pickle.load(file)
                self.sdr = data
                print 'here'
        except:
            self.generate_sdr()
            with open(self.datafile, 'wb') as file:
                pickle.dump(self.sdr, file)

    def generate_sdr(self):
        self.sdr = np.zeros((self.discretization, self.sdr_size), dtype=bool)
        self.sdr[0][np.random.choice(self.sdr_size, size=int(self.sdr_size * self.sparsity), replace=False)] = 1
        for i in range(1, self.discretization):
            old = np.random.choice(np.nonzero(self.sdr[i-1])[0], size=int(self.sdr_size * self.sparsity * self.similarity), replace=False)
            fresh_inds = np.delete(np.arange(self.sdr_size), np.nonzero(self.sdr[i-1])[0])
            new = np.random.choice(fresh_inds, size=int(self.sdr_size * self.sparsity * (1 - self.similarity)), replace=False)
            self.sdr[i][old] = 1
            self.sdr[i][new] = 1

    def encode(self, number):
        # number should be in range [0, 1)
        return self.sdr[int(np.rint(number*self.discretization))]


if __name__ == '__main__':
    num_encoder = NumberEncoder()

    sdr1 = num_encoder.encode(0.47)
    sdr2 = num_encoder.encode(0.5)
    print sdr1.astype(int)
    print np.count_nonzero(sdr1*sdr2)




