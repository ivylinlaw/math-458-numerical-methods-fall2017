import numpy as np
from tri import tri

def put_option_american():	# TODO: parameterize variables
	K = 50.0 	# strick price
	r = 0.10 	# risk-free interest rate
	sigma = 0.5 #0.4 	# volatility
	T = 1.0 #5.0	# life of option (unit as month)
	N = 10	# number of time intervals
	S_max = 150.0 #100.0		# stock price where f(S_max, t) = 0
	M = 150 #20 	# number of stock prices intervals

	delta_t = T / N 	# length of each equally spaced time interval
	delta_s = S_max / M 	# length of each equally spaced stock price interval

	a = np.full(M + 1, 0.0)	# size of M + 1
	d = np.full(M + 1, 1.0)	# size of M + 1
	c = np.full(M + 1, 0.0)	# size of M + 1

	for j in range(1, M + 1):
		a[j] = 0.5 * delta_t * (r * j - np.square(sigma) * np.square(j))
		d[j] = 1 + delta_t * (r + np.square(sigma) * np.square(j))
		c[j] = 0.5 * delta_t * (-r * j + np.square(sigma) * np.square(j))

	b = np.empty([N + 1, M + 1])

	for j in range(M + 1):
		b[N][j] = max(K - j * delta_s, 0)
	for i in range(N + 1):
		b[i][0] = K

	for i in range(N - 1, -1, -1):
		y = np.array(b[i + 1, :])
		y = y[1 : M]
		y[0] = y[0] - a[0] * K
		x = tri(len(y), a, d, c, y)
		for j in range(1, M):
			b[i][j] = x[j - 1]

		for j in range(M + 1):
			b[i][j] = max(K - j * delta_s, b[i][j])

if __name__ == '__main__':
    put_option_american()