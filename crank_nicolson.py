import numpy as np
from tri import tri

def crank_nicolson():
	s = h ^ 2 / k
	r = 2 + s
	# copy numpy arrays a, d, c, b
	c_, d_ = map(np.asfarray, (c, d))
	for i in xrange(n - 1):
		d_[i] = r
		c_[i] = -1
		u[i] = np.sin(np.pi * i * h)
	for j in xrange(m):
		for i in xrange(n - 1):
			d_[i] = r
			v[i] = s * u[i]
		v = tri(n - 1, c, d, c, v)
		t = j * k
		for i in xrange(n - 1):
			u[i] = np.exp(-np.pi ^ 2 * t) * np.sin(np.pi * i * h) - v[i]
