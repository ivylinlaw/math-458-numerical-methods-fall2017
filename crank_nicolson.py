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

def crank_nicolson_heat_eq(K, r, sigma, h, k, n, m):
	alpha = (sigma * sigma - 2 * r) / (2 * sigma * sigma)
	beta = (sigma * sigma + 2 * r) / (2 * sigma * sigma)
	beta *= -beta

	def init(x):
		# return u(x, 0)
		return np.exp(-alpha * x) * max(np.exp(x) - K , 0)

	def integrand(E, x, t):
		# return crank_nicolson_heat_eq_init(E) * np.exp(-np.power(x - E) / (4 * t))
		return init(E) * np.exp(-(x - E) * (x - E) / (4 * t))

	s = np.power(h, 2) / k
	r = 2 + s

	d = np.full(n - 1, 0.0)
	c = np.full(n - 1, 0.0)
	u = np.full(n - 1, 0.0)
	v = np.full(n - 1, 0.0)

	for i in range(n - 1):
		d[i] = r
		c[i] = -1
		u[i] = init((i + 1) * h) #np.sin(np.pi * (i + 1) * h)

	print('u: %s' % u)

	for j in range(m):
		for i in range(n - 1):
			d[i] = r
			v[i] = s * u[i]
		v = tri(n - 1, c, d, c, v)

		print('v: %s' % v) #

		t = (j + 1) * k
		for i in range(n - 1):
			# args=(x, t, K, alpha)
			u[i] = (1 / np.sqrt(4 * np.pi * (i + 1) * h)) * quad(integrand, -np.inf, np.inf, args=(i, (i + 1) * h))[0] #np.exp(-np.power(np.pi, 2) * t) * np.sin(np.pi * (i + 1) * h) 
			u[i] -= v[i]

		print('diff: %s' % u)

		for i in range(n - 1):
			u[i] = v[i]

if __name__ == '__main__':
	crank_nicolson_heat_eq(50.0, 0.10, 0.4, np.power(2.0, -4), np.power(2.0, -10), 16, 13)

