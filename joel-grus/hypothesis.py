from __future__ import division 
import math, random


def normal_cdf(x, mu=0, sigma=1):
	return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


if __name__ == '__main__': 
	print 'hi'
