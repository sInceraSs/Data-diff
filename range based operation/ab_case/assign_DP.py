#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

def DP(l, r, theta):
	"""
	Test each k in [l, r] as split point to minimize (l, r, theta).
	The cost of assigning tuples in [l, r] with tuples from B_old to B_new is 
	just the same as # of operations about assigning the same value theta of tuples in B_new
	"""
	global dst
	# dst is actually the col B which sorted by col A
	min_, min_flag = float('inf'), 0
	if l == r:
		if theta not in dp[l][r]:
			if dst[l] == theta:
				dp[l][r][ theta ], ans_op[l][r][ theta ] = 0, (l, r, float('nan') )
			else:
				dp[l][r][ theta ], ans_op[l][r][ theta ] = 1, (l, r, theta)
		return dp[l][r][ theta ]

	if theta == theta and theta == dst[l]:
		if theta not in dp[l+1][r]:
			dp[l+1][r][theta] = DP(l+1, r, theta)
		if min_ > dp[l+1][r][theta]:
			min_, min_flag = dp[l+1][r][theta], (l+1, r, theta)
	if theta == theta and theta == dst[r]:
		if theta not in dp[l][r-1]:
			dp[l][r-1][theta] = DP(l, r-1, theta)
		if min_ > dp[l][r-1][theta]:
			min_, min_flag = dp[l][r-1][theta], (l, r-1, theta)

	if dst[l] not in dp[l+1][r]:
		dp[l+1][r][ dst[l] ] = DP(l+1, r, dst[l])
	if min_ > dp[l+1][r][ dst[l] ] + 1:
		min_, min_flag = dp[l+1][r][dst[l]] + 1, (l+1, r, dst[l])

	if dst[r] not in dp[l][r-1]:
		dp[l][r-1][dst[r]] = DP(l, r-1, dst[r])
	if min_ > dp[l][r-1][dst[r]] + 1:
		min_, min_flag = dp[l][r-1][dst[r]] + 1, (l, r-1, dst[r])
	
	for k in range(l,r):
		if theta not in dp[l][k]:
			dp[l][k][theta] = DP(l, k, theta)
		if theta not in dp[k+1][r]:
			dp[k+1][r][theta] = DP(k+1, r, theta)
		if min_ > dp[l][k][theta] + dp[k+1][r][theta]:
			min_, min_flag = dp[l][k][theta] + dp[k+1][r][theta], (l, k, r, theta)

	dp[l][r][theta], ans_op[l][r][theta] = min_, min_flag
	# print dp[l][r][theta], "op", ans_op[l][r][theta], "range", l, r, theta
	return min_

dst = [4, 4, 5, 6, 4, 4, 5]
ans_op, dp = {}, {}
op_res = []

for i in range( len(dst) ):
	dp[i], ans_op[i] = {}, {}
	for j in range( len(dst) ):
		dp[i][j], ans_op[i][j] = {}, {}
# l, r, theta means set all tuples from l to r to theta
print DP(0, 5, float('nan'))
# print ans_op[0][  len(dst) - 1 ]
