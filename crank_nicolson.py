import numpy as np
from scipy.integrate import quad
from decimal import Decimal
from tri import tri

def crank_nicolson(h, k, n, m):
	vec = []

	s = np.power(h, 2) / k
	r = 2 + s

	d = np.full(n - 1, 0.0)
	c = np.full(n - 1, 0.0)
	u = np.full(n - 1, 0.0)
	v = np.full(n - 1, 0.0)

	for i in range(n - 1):
		d[i] = r
		c[i] = -1
		u[i] = np.sin(np.pi * (i + 1) * h)

	vec.append(['u'] + map(Decimal, u))

	for j in range(m):
		for i in range(n - 1):
			d[i] = r
			v[i] = s * u[i]
		v = tri(n - 1, c, d, c, v)

		vec.append(['v'] + map(Decimal, v))

		t = (j + 1) * k
		for i in range(n - 1):
			u[i] = np.exp(-np.power(np.pi, 2) * t) * np.sin(np.pi * (i + 1) * h) 
			u[i] -= v[i]

		# print('diff: %s' % u)
		vec.append(['diff'] + map(Decimal, u))

		for i in range(n - 1):
			u[i] = v[i]

	vec_t = [['t']]
	vec_t[0] += [i + 1 for i in range(len(vec[0]) - 1)]
	return vec_t + vec

if __name__ == '__main__':
	crank_nicolson(np.power(2.0, -4), np.power(2.0, -10), 16, 13)

