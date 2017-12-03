import numpy as np
from tri import tri

def put_option(flag, t, S): # flag as 'american' or 'european' (default)
	K = 50.0 	# strick price
	r = 0.10 	# risk-free interest rate
	sigma = 0.4 	# volatility
	T = 5.0 / 12	# life of option (unit as year)
	N = 10	# number of time intervals
	S_max = 100.0		# stock price where f(S_max, t) = 0
	M = 20 	# number of stock prices intervals

	delta_t = T / N 	# length of each equally spaced time interval
	delta_s = S_max / M 	# length of each equally spaced stock price interval

	a = np.full(M + 1, 0.0)	# size of M + 1
	d = np.full(M + 1, 1.0)	# size of M + 1
	c = np.full(M + 1, 0.0)	# size of M + 1

	for j in range(1, M + 1):
		a[j] = 0.5 * delta_t * (r * j - np.square(sigma) * np.square(j))
		d[j] = 1 + delta_t * (r + np.square(sigma) * np.square(j))
		c[j] = 0.5 * delta_t * (-r * j - np.square(sigma) * np.square(j))

	b = np.full([N + 1, M + 1], 0.0)

	for j in range(M + 1):
		b[N][j] = max(K - j * delta_s, 0)
	for i in range(N + 1):
		b[i][0] = K

	# construct tridiagonal matrix A
	a_, d_, c_ = map(np.asfarray, (a, d, c))
	a_ = a_[2 : -1]
	c_ = c_[1 : -2]
	d_ = d_[1 : -1]
	A = np.diag(a_, -1) + np.diag(d_, 0) + np.diag(c_, 1)

	for i in range(N - 1, -1, -1):
		y = np.array(b[i + 1, :])
		y = y[1 : M]

		y[0] = y[0] - a[1] * K	# len(y) = M - 1

		x = np.linalg.solve(A, y)
		# x = tri(len(y), a[1 : -1], d_, c[1: -1], y)

		for j in range(1, M):
			b[i][j] = x[j - 1]

		if(flag is 'american'):
			for j in range(M + 1):
				b[i][j] = max(K - j * delta_s, b[i][j])

	# print(b)

	return b[t][int(S/delta_s)] # return put price given t and S

if __name__ == '__main__':
    print('American put option price when t = 0 and S = 50: $%s' % put_option('american', 0, 50))
    print('European put option price when t = 0 and S = 50: $%s' % put_option('european', 0, 50))