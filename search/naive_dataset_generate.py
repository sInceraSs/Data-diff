#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import sys, json, re, os
import statsmodels.api as sm
import scipy.stats.mstats
import scipy.stats
from time import clock
reload(sys)
sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt

rg_idx_4_2 = [[0, 4], [1, 6], [2, 7], [3, 8], [5, 9]]
rg_idx_4_1 = [[0, 4], [1, 5], [2, 7], [3, 6]]
rg_idx_3_1 = [[0, 3], [1, 4], [2, 5]]
rg_idx_3_2 = [[0, 3], [2, 6], [1, 5], [4, 7]]
rg_idx_3_3 = [[0, 3], [2, 6], [1, 5], [4, 9], [7, 10], [8, 11]]
f = open("dataset", "r")
for line in f:
	data = json.loads(line)
	print data
	exit()

for i in range(10):
	data = [5, 4, 2]
	cur_x = np.random.randint(0, 100, 60)
	dst_x = np.array(cur_x)
	rg = sorted(np.random.randint(0, 60, 12))
	add = np.random.randint(-100, 101, 5)
	op, model = [], rg_idx_3_1
	for j in range( len(model) ):
		dst_x[ rg[ model[j][0] ]:rg[ model[j][1]] ] += add[j]
		op.append( [ add[j], rg[ model[j][0] ], rg[ model[j][1]] ] )
	data.append( cur_x.tolist() ), data.append( dst_x.tolist() ), data.append( op )
	print data
	f.write( json.dumps(data) )
	f.write("\n")
f.close()


