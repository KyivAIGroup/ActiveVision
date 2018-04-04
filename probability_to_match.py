# What is the probability that for an object with N features, and only l stored, that if I look r times on random
# I will hit the correct combination


import math
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

def C(b, s):
    # number of combinations
    if s == 0 or s == b:
        return 1
    return reduce(lambda x, y: x*y, range(b-s+1,b+1))/math.factorial(s)

def overlap(n, a, s, theta):
    return Decimal(C(s, theta)) * C(n-s, a-theta) / C(n, a)


def prob(N, l, r):
    L = N * (N - 1)
    p1 = 1 - float(C(L-l, r)) / C(L, r)
    p2 = 1 - (1 - float(l) / L) ** r
    print p1, p2


prob(4, 10, 5)