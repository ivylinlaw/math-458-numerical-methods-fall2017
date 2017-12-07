import numpy as np

class Binomial:
	def __init__(self, strikeprice, rate, sigma, totaltime, timeslice):
		self.K = strikeprice 	# strick price
		self.r = rate 	# risk-free interest rate
		self.sigma = sigma 	# volatility
		self.T = totaltime 	# life of option (unit as year)
		self.N = timeslice 	# number of time intervals
		self.delta_t = self.T / self.N if self.N != 0 else 0.0 	# length of each equally spaced time interval
		self.u = np.exp(self.sigma * np.sqrt(self.delta_t))	# up factor
		self.d = 1 / self.u 	# down factor
		self.p = (np.exp(self.r * self.delta_t) - self.d) / (self.u - self.d)

	def get_price_tree(self, S):	# S as initial stock price
		S_tree = np.full([self.N + 1, self.N + 1], 0.0)

		S_tree[0, 0] = S
		for i in range(1, self.N + 1):
			for j in range(i + 1):
				if(j == 0):
					S_tree[i, j] = S_tree[i - 1, j] * self.u
				else:
					S_tree[i, j] = S_tree[i - 1, j - 1] * self.d

		return S_tree

	def get_value_tree(self, cpflag, flag, S_tree):	# cpfalg as 'call' or 'put'
		C_tree = np.full([self.N + 1, self.N + 1], 0.0)

		for i in range(self.N, -1, -1):
			for j in range(i + 1):
				if(i == self.N):
					if(cpflag is 'call'):
						C_tree[i , j] = max(S_tree[i , j] - self.K, 0.0)
					else:
						C_tree[i , j] = max(self.K - S_tree[i , j], 0.0)
				else:
					C_tree[i , j] = np.exp(-self.r * self.delta_t) * (self.p * C_tree[i + 1, j] + ((1 - self.p) * C_tree[i + 1, j + 1]))
					# C_tree[i , j] = (self.p * C_tree[i + 1, j] + ((1 - self.p) * C_tree[i + 1, j + 1])) / (1 + self.r)
					if(flag is 'american'):
						if(cpflag is 'call'):
							C_tree[i, j] = max(C_tree[i, j], max(S_tree[i , j] - self.K, 0.0))
						else:
							C_tree[i, j] = max(C_tree[i, j], max(self.K - S_tree[i , j], 0.0))

		return C_tree

if __name__ == '__main__':
	BT = Binomial(50.0, 0.10, 0.4, 5.0 / 12, 10)
	# BT = Binomial(12.0, 0.05, 0.4, 1.5, 3)
	S_tree = BT.get_price_tree(50.0)
	# print('%s' % S_tree)
	# print('\n\n\n')
	# V_tree = BT.get_value_tree('put', 'european', S_tree)
	# print('%s' % V_tree)
	# print('\n\n\n')
	V_tree = BT.get_value_tree('put', 'american', S_tree)
	print('%s' % V_tree[0,0])