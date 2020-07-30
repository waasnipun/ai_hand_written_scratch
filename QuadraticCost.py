import numpy as np
class QuadraticCost(object):
    @staticmethod
    def fn(a,y):
        return 0.5*np.linalg.norm(a-y)**2
    @staticmethod
    def delta(z,a,y):
        return (a-y)*sigmoid_prime(z)

