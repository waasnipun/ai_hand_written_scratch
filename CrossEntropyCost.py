import numpy as np
class CrossEntropyCost(object):
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    def delta(z,a,y):
        return (a-y)
