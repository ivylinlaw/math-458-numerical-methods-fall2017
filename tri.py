import numpy as np

def tri(n, a, d, c, b):
	# define x as numpy array with len n
	x = np.empty(n)
	# copy numpy arrays a, d, c, b
	a_, b_, c_, d_ = map(np.asfarray, (a, b, c, d))
	for i in xrange(1, n):
		xmult = float(a_[i - 1]) / d_[i - 1]
		d_[i] = d_[i] - xmult * c_[i - 1]
		b_[i] = b_[i] - xmult * b_[i - 1]
	# set x's last element
	x[-1] = float(b_[-1]) / d_[-1]
	for i in xrange(n - 2, -1, -1):
		x[i] = (b_[i] - c_[i] * x[i + 1]) / d_[i]
	# TODO: delete copy arrays for memory?
	return x
