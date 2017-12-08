import sys
import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from texttable import Texttable
from black_scholes import BlackScholes
from binomial import Binomial
from crank_nicolson import crank_nicolson

# def print_table():
# 	#

def main():
	# write to file
	sys.stdout = open('output.txt', 'w')

	BS = BlackScholes(50.0, 0.10, 0.4, 5.0 / 12, 10, 100.0, 20)
	# 1. (a)
	print('(a)')
	price = BS.calculate_bs_implicit_fd('put', 'american', 0, 50)
	BS.print_price_matrix()
	BS.graph_price_matrix('American put option price when t = 0 and S = 50')
	print('American put option price when t = 0 and S = 50 (Implicit Finite Differences): $%s' % price)
	
	# 1. (b)
	print('(b)')
	price = BS.calculate_bs_implicit_fd('put', 'european', 0, 50)
	BS.print_price_matrix()
	BS.graph_price_matrix('European put option price when t = 0 and S = 50')
	print('European put option price when t = 0 and S = 50 (Implicit Finite Differences): $%s' % price)
	print('European put option price when t = 0 and S = 50  (Black-Scholes): $%s' % BS.calculate_bs_model('put', 50))

	# plot error difference of implicit fd and bs model with matplotlib
	# df_err = pd.DataFrame({'x': BS.get_t_intervals(), 'y': [r[int(50/(100.0/20))] - BS.calculate_bs_model('put', 50) for r in BS.get_price_matrix()]})
	# sns.set()
	# plt.plot('x', 'y', data=df_err, marker='.')
	# plt.show()


	# 1. (c)
	print('(c)')
	print('Solve the heat equation (Crank-Nicolson)')
	vec = crank_nicolson(np.power(2.0, -4), np.power(2.0, -10), 16, 13)
	# print table
	t_c = Texttable()
	t_c.add_rows(vec)
	t_c.set_cols_width([10 for i in range(len(vec[0]))])
	print(t_c.draw())

	# 1. (d)
	# will be done in MATLAB

	# 1. (e)
	print('(e)')
	print('Estimate the price of a 5 month American put option when the stock price S0 = $50, the strike price K = $50, the risk-free interest rate r = 0.10 and the volatility sigma = 0.4 (Binomial Tree)')
	BT = Binomial(50.0, 0.10, 0.4, 5.0 / 12, 10)
	S_tree = BT.get_price_tree(50.0, True)
	V_tree = BT.get_value_tree('put', 'american', S_tree, True)
	print('$%s' % V_tree[0, 0])

	# 1. (f)
	# print('(f)')
	pt_cnt = 100
	pt_t = [i for i in range(1, pt_cnt + 1)]
	# prices by BlackScholes
	bs = BlackScholes(50.0, 0.10, 0.4, 5.0 / 12, 10, 100.0, 20).calculate_bs_model('put', 50.0)
	pt_bs = np.full(pt_cnt, bs)
	# prices by Binomial
	pt_bt = []
	for i in range(len(pt_t)):
		bt = Binomial(50.0, 0.10, 0.4, 5.0 / 12, pt_t[i])
		pt_bt.append(bt.get_value_tree('put', 'american', bt.get_price_tree(50.0))[0,0])

	df_bt = pd.DataFrame({'x': pt_t, 'y': pt_bt})
	df_bs = pd.DataFrame({'x': pt_t, 'y': pt_bs})
	# plot with matplotlib
	sns.set()
	plt.gca().set_prop_cycle(cycler('color', ['mediumvioletred', 'blue']))

	plt.plot('x', 'y', data=df_bt, marker='.')
	plt.plot('x', 'y', data=df_bs, marker='.')

	x1, x2, y1, y2 = plt.axis()
	plt.axis((x1, x2, 0, 10))
	plt.show()


	# 1. (g)
	print('(g)')
	print('Estimate the price of an February 2015 European call option when the stock price S0 = $326, the strike price K = $320, the risk-free interest rate r = 0.02 and the volatility sigma = 0.4 (Binomial Tree)')
	BT = Binomial(320.0, 0.02, 0.4, 5.0 / 12, 10)
	S_tree = BT.get_price_tree(326.0, True)
	V_tree = BT.get_value_tree('call', 'european', S_tree, True)
	print('$%s' % V_tree[0, 0])

	# 1. (h)
	print('(h)')
	print('Estimate the price of an February 2015 European call option when the stock price S0 = $326, the strike price K = $320, the risk-free interest rate r = 0.02 and the volatility sigma = 0.4 (Black-Scholes)')
	BS = BlackScholes(320.0, 0.02, 0.4, 5.0 / 12, 10, 100.0, 20)
	bs_price = BS.calculate_bs_model('call', 326.0)
	print('$%s' % bs_price)
	print('Estimate the price of its corresponding put option price')
	print('$%s' % BS.get_parity(bs_price, 326.0))

if __name__ == '__main__':
    main()