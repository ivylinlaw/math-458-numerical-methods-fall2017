import numpy as np
from texttable import Texttable

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

	def get_price_tree(self, S, ifPrintTree=False):	# S as initial stock price
		S_tree = np.full([self.N + 1, self.N + 1], 0.0)

		S_tree[0, 0] = S
		for i in range(1, self.N + 1):
			for j in range(i + 1):
				if(j == 0):
					S_tree[i, j] = S_tree[i - 1, j] * self.u
				else:
					S_tree[i, j] = S_tree[i - 1, j - 1] * self.d

		# print S_tree as table
		if(ifPrintTree):
			table = Texttable()
			table.add_rows(S_tree.transpose())
			table.set_cols_width([5 for i in range(len(S_tree[0]))])
			print(table.draw())

		return S_tree

	def get_value_tree(self, cpflag, flag, S_tree, ifPrintTree=False):	# cpfalg as 'call' or 'put'
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
					if(flag is 'american'):
						if(cpflag is 'call'):
							C_tree[i, j] = max(C_tree[i, j], max(S_tree[i , j] - self.K, 0.0))
						else:
							C_tree[i, j] = max(C_tree[i, j], max(self.K - S_tree[i , j], 0.0))

		# print C_tree as table
		if(ifPrintTree):
			table = Texttable()
			table.add_rows(C_tree.transpose())
			table.set_cols_width([5 for i in range(len(C_tree[0]))])
			print(table.draw())

		return C_tree

	def get_parity(self, C_tree, S):
		c = C_tree[0,0]
		p = c + self.K / np.power((1+self.r), self.T) - S
		return p

if __name__ == '__main__':
	BT = Binomial(320.0, 0.02, 0.4, 5.0 / 12, 10)
	S_tree = BT.get_price_tree(326.0)
	V_tree = BT.get_value_tree('call', 'european', S_tree, True)
	print('%s' % V_tree[0,0])