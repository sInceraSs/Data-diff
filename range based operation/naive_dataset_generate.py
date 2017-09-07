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

def AB_add_case(filename):
	rg_idx_4_2 = [[0, 4], [1, 6], [2, 7], [3, 8], [5, 9]]
	rg_idx_4_1 = [[0, 4], [1, 5], [2, 7], [3, 6]]
	rg_idx_3_1 = [[0, 3], [1, 4], [2, 5]]
	rg_idx_3_2 = [[0, 3], [2, 6], [1, 5], [4, 7]]
	rg_idx_3_3 = [[0, 3], [2, 6], [1, 5], [4, 9], [7, 10], [8, 11]]
	f = open(filename, "r")
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

def check(num, candi, op=False):
	if op and num in candi:
		return True
	for pk_num in candi:
		if np.abs(num*1.0 - pk_num*1.0) < 2:
			return False
	return True

def AB_lt_exclude_case(filename):
	f = open(filename, "w")
	op_nums = [i for i in range(3, 11)]
	for op_num in op_nums:
		for i in range(6):
			pk_op_cs = []
			for c_idx in range(op_num):
				pk_op_c = np.random.randint(-5, 6) * 1.0
				while not pk_op_c:
					pk_op_c = np.random.randint(-5, 6) * 1.0
				pk_op_cs.append(pk_op_c)
			pk_op_is = np.random.randint(-100, 101, op_num) * 1.0

			pk_range, op = [], []
			for j in range(op_num):
				first = np.random.randint(0, 100)
				while not check(first, pk_range, False) or first == 1:
					first = np.random.randint(0, 100)
				second = np.random.randint(0, 100)
				while (not check(second, pk_range, False)) or np.abs(second*1.0 - first*1.0) < 2 or second == 1:
				# while (not check(second, pk_range, not first in pk_range)) or np.abs(second*1.0 - first*1.0) < 2 or second == 1:
					second = np.random.randint(0, 100)
				pk_range.append(min(first, second))
				pk_range.append(max(first, second))

			ori_x = np.random.randint(0, 100, 100)
			dst_x = np.array(ori_x)
			for j in range(op_num):
				op.append( [ pk_op_cs[j] , pk_op_is[j], pk_range[j*2], pk_range[j*2+1] ]  )
			for j in range(op_num):
				dst_x[ op[j][2]:op[j][3] ] = dst_x[  op[j][2]:op[j][3] ] * op[j][0] + op[j][1]
			data = [ op_num ]
			data.append( ori_x.tolist() ), data.append( dst_x.tolist() ), data.append( op )
			print data
			f.write( json.dumps(data) )
			f.write("\n")
			print sorted(pk_range)
	f.close()

def AB_lt_general_case(filename):
	f = open(filename, "w")
	op_nums = [i for i in range(3, 11)]
	for op_num in op_nums:
		for i in range(6):
			pk_op_cs = []
			for c_idx in range(op_num):
				pk_op_c = np.random.randint(-5, 6) * 1.0
				while not pk_op_c:
					pk_op_c = np.random.randint(-5, 6) * 1.0
				pk_op_cs.append(pk_op_c)
			pk_op_is = np.random.randint(-100, 101, op_num) * 1.0

			pk_range, op = [], []
			for j in range(op_num):
				first = np.random.randint(0, 100)
				# while not check(first, pk_range, False) or first == 1:
					# first = np.random.randint(0, 100)
				second = np.random.randint(0, 100)
				while second == first:
				# while (not check(second, pk_range, False)) or np.abs(second*1.0 - first*1.0) < 2 or second == 1:
				# while (not check(second, pk_range, not first in pk_range)) or np.abs(second*1.0 - first*1.0) < 2 or second == 1:
					second = np.random.randint(0, 100)
				pk_range.append(min(first, second))
				pk_range.append(max(first, second))

			ori_x = np.random.randint(0, 100, 100)
			dst_x = np.array(ori_x)
			for j in range(op_num):
				op.append( [ pk_op_cs[j] , pk_op_is[j], pk_range[j*2], pk_range[j*2+1] ]  )
			for j in range(op_num):
				dst_x[ op[j][2]:op[j][3] ] = dst_x[  op[j][2]:op[j][3] ] * op[j][0] + op[j][1]
			data = [ op_num ]
			data.append( ori_x.tolist() ), data.append( dst_x.tolist() ), data.append( op )
			print data
			f.write( json.dumps(data) )
			f.write("\n")
			print sorted(pk_range)
	f.close()

def AAB_add_case(filename):
	f = open(filename, "w")
	for op_num in range(3, 7):
		for i in range(6):
			x_x = np.random.randint(-50, 50, 100)
			x_y = np.random.randint(-50, 50, 100)
			ori_delta = np.zeros(shape=100)
			ops, data, adds = [], [op_num], []
			while len(adds) < op_num:
				add = np.random.randint(-50, 50)
				if add != 0 and add not in adds:
					adds.append(add)
			for j in range(op_num):
				pk_range = []
				while len(pk_range) < 4:
					add = np.random.randint(-50, 50)
					if add not in pk_range:
						pk_range.append(add)
				ops.append( (adds[j], min(pk_range[0], pk_range[1]), max(pk_range[0], pk_range[1]), min(pk_range[2], pk_range[3]), max(pk_range[2], pk_range[3])) )
			for add, x_min, x_max, y_min, y_max in ops:
				for x_x_idx, x_x_v in enumerate(x_x):
					if (x_x_v > x_min and x_x_v <= x_max) and (x_y[x_x_idx] > x_min and x_y[x_x_idx] <= x_max):
						ori_delta[x_x_idx] += add
			ori_delta.reshape(-1, 1)
			x_x, x_y = np.array(x_x).reshape(-1, 1),  np.array(x_y).reshape(-1, 1)
			ori_x = np.concatenate((x_x, x_y), axis=1)
			data.append( ori_x.tolist() ), data.append( ori_delta.tolist() ), data.append( ops )
			f.write( json.dumps(data) )
			f.write("\n")
	f.close()

# AAB_add_case("dataset_aab")
AB_lt_general_case("dataset_general")