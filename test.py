import numpy as np
from tri import tri

def test_tri():
	n = 50
	a = np.full(n, -1)
	d = np.full(n, 5)
	c = np.full(n, -1)
	b = np.arange(1, 51)
	x = tri(n, a, d, c, b)
	print(x)

if __name__ == '__main__':
    test_tri()