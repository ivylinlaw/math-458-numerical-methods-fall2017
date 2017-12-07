import numpy as np
from scipy.stats import norm
from tri import tri

class BlackScholes:
	def __init__(self, strikeprice, rate, sigma, totaltime, timeslice, maxprice, priceslice):
		self.K = strikeprice 	# strick price
		self.r = rate 	# risk-free interest rate
		self.sigma = sigma 	# volatility
		self.T = totaltime 	# life of option (unit as year)
		self.N = timeslice 	# number of time intervals
		self.S_max = maxprice 	# stock price where f(S_max, t) = 0
		self.M = priceslice 	# number of stock prices intervals
		self.prices = np.full([self.N + 1, self.M + 1], 0.0) 	# an (N+1)x(M+1) matrix of optoin prices solved

	# return list of time within totaltime divided by timeslice
	def get_t_intervals(self):
		return [i * self.T / self.N for i in range(self.N)]

	def get_price_matrix(self):
		return self.prices

	# calculate option price by Black-Scholes model
	def calculate_bs_model(self, cpfalg, S):	# cpfalg as 'call' or 'put'
		d = 0 #???
		d1 = (np.log(float(S) / self.K) + ((self.r - d) + np.square(self.sigma) / 2.0) * self.T) / (self.sigma * np.sqrt(self.T))
		d2 = d1 - self.sigma * np.sqrt(self.T)
		if(cpfalg is 'call'):
			return S * np.exp(-d * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
		else: # cpfalg is 'put'
			return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - S * np.exp(-d * self.T) * norm.cdf(-d1)

	# calculate option price using implicit finite differences method to solve the Black-Scholes PDE
	def calculate_bs_implicit_fd(self, cpflag, flag, t, S): # flag as 'american' or 'european' (default)
		delta_t = self.T / self.N 	# length of each equally spaced time interval
		delta_s = self.S_max / self.M 	# length of each equally spaced stock price interval

		a = np.full(self.M + 1, 0.0)	# size of M + 1
		d = np.full(self.M + 1, 1.0)	# size of M + 1
		c = np.full(self.M + 1, 0.0)	# size of M + 1

		for j in range(1, self.M + 1):
			a[j] = 0.5 * delta_t * (self.r * j - np.square(self.sigma) * np.square(j))
			d[j] = 1 + delta_t * (self.r + np.square(self.sigma) * np.square(j))
			c[j] = 0.5 * delta_t * (-self.r * j - np.square(self.sigma) * np.square(j))

		self.prices = np.full([self.N + 1, self.M + 1], 0.0)

		for j in range(self.M + 1):
			if(cpflag is 'call'):
				self.prices[self.N][j] = max(j * delta_s - self.K, 0)
			else:
				self.prices[self.N][j] = max(self.K - j * delta_s, 0)
		for i in range(self.N + 1):
			self.prices[i][0] = self.K

		# construct tridiagonal matrix A
		a_, d_, c_ = map(np.asfarray, (a, d, c))
		a_ = a_[2 : -1]
		c_ = c_[1 : -2]
		d_ = d_[1 : -1]
		A = np.diag(a_, -1) + np.diag(d_, 0) + np.diag(c_, 1)

		for i in range(self.N - 1, -1, -1):
			y = np.array(self.prices[i + 1, :])
			y = y[1 : self.M]

			y[0] = y[0] - a[1] * self.K	# len(y) = M - 1

			x = np.linalg.solve(A, y)

			for j in range(1, self.M):
				self.prices[i][j] = x[j - 1]

			if(flag is 'american'):
				for j in range(self.M + 1):
					if(cpflag is 'call'):
						self.prices[i][j] = max(j * delta_s - self.K, self.prices[i][j])
					else:
						self.prices[i][j] = max(self.K - j * delta_s, self.prices[i][j])

		return self.prices[t][int(S/delta_s)] # return put price given t and S

	def get_parity(self, price, S):
		p = price + self.K / np.power((1+self.r), self.T) - S
		return p

if __name__ == '__main__':
	BS = BlackScholes(50.0, 0.10, 0.4, 5.0 / 12, 10, 100.0, 20)
	print('American put option price when t = 0 and S = 50: $%s' % BS.put_option('put', 'american', 0, 50))
	print('European put option price when t = 0 and S = 50: $%s' % BS.put_option('put', 'european', 0, 50))