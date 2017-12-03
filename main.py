from black_scholes import BlackScholes

def main():
	BS = BlackScholes(50.0, 0.10, 0.4, 5.0 / 12, 10, 100.0, 20)
	# # 1. (a)
	print('American put option price when t = 0 and S = 50: $%s' % BS.put_option('put', 'american', 0, 50))
	# # 1. (b)
	print('European put option price when t = 0 and S = 50: $%s' % BS.put_option('put', 'european', 0, 50))
	print('European put option price when t = 0 and S = 50 (by B-S model): $%s' % BS.calculate_bs_model('put', 50))


if __name__ == '__main__':
    main()