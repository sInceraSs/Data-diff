#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from time import clock
import heapq as hp
from sklearn import linear_model
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus, json
from time import clock

dataset = []
with open("dataset", 'r') as infile:
    for line in infile:
        dataset.append(json.loads(line))
data_choice = [3, 4, 15, 19, 21, 33, 41, 43, 53, 55]
#np.random.shuffle(data_choice)
bactch_idx = 9
ori_x = np.array(dataset[data_choice[bactch_idx]][3])
dst_x = np.array(dataset[data_choice[bactch_idx]][4])
min_op = dataset[data_choice[bactch_idx]][0] * 2 - 1
globla_hp, n = [], 60

start_c = 0
tmp_s = clock()
tmp_x, base_o = np.array(ori_x), 0
for i in reversed(range(len(dst_x))):
	cur_a = dst_x[i] - tmp_x[i]
	tmp_x[:i+1] += cur_a
	if cur_a != 0:
		base_o += 1
print clock() - tmp_s, base_o
#exit()
def store(ans):
        f = open("3_11", "w")
        f.write("\n")
        f.write(json.dumps(ans))
        print  len(ans)
start_t = clock()
ori_x, dst_x = np.array(ori_x), np.array(dst_x)
# ( weigh, depth, [pre_op]  )
# pre_op : (coef, intercept, range)
hp.heapify(globla_hp)
cur_op, pre_add = [], float('nan')
for i in range(len(ori_x)):
	cur_add = dst_x[i] - ori_x[i]
	if cur_add == 0:
		continue
	if cur_add == pre_add:
		cur_op[-1][2] = i
	else:
		cur_op.append( [cur_add, i, i] )
		pre_add = cur_add

_max, _min = len(cur_op), float('inf')
def weigh(cur_x, depth):
	global _min
	dist = 1
	tmp = []
	for i in range( len(cur_x) ):
		dis = cur_x[i] - dst_x[i]
		if abs(dis*1.0) > 1e-6 and dis not in tmp:
			tmp.append(dis)
	dist += len(tmp)
			
	dist_1 = 0
	cur_op, pre_add = [], float('nan')
	for i in range(len(cur_x)):
		cur_add = dst_x[i] - cur_x[i]
		if cur_add == 0:
			continue
		if cur_add == pre_add:
			cur_op[-1][2] = i
		else:
			cur_op.append( [cur_add, i, i] )
			pre_add = cur_add
	dist_1 = len(cur_op)
	if dist_1 == 0:
		return depth, dist_1
	#weight = dist * dist_1 * 10
	#weight =  np.log2(dist*dist_1) * (100)
	weight = np.log2(dist*depth)*dist_1 * (100)
	return (weight, dist_1)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def stateTrans(state):
	if isinstance(state, list):
		s = ""
		for pk_op_i, start, end in state:
			s = s + str(pk_op_i) + "," + str(start) + "," + str(end) + " "
		return s
	else:
		s = []
		ops = state.split(' ')
		for op in ops:
			if len(op) == 0:
				continue
			op_i = op.split(",")
			s.append([int(op_i[0]), int(op_i[1]), int(op_i[2])])
		return s
def cal_op(s):
    global dst_x
    pre_coef, pre_add = float('nan'), float('nan')
    cur_op = []
    for i in range( 0, len(s) ):
        o_tuple, d_tuple = s[i], dst_x[i]
	cur_add = dst_x[i] - s[i]
        if cur_add == 0:
		continue
        if cur_add == pre_add:
               cur_op[-1][3] = i+1
        else:
               cur_op.append( [1, cur_add, i, i+1] )
               pre_add = cur_add
    return cur_op

def cal_state(s_op, filter_, option=False):
    global ori_op, ori_x
    ss, candidate_op, true_op = [], [], []
    tmp_r, tmp_f = [], []
    if len(filter_) > 0:
         #print filter_
         for op in filter_:
             if len(op) == 0:
                 continue
             op = stateTrans(op)[0]
             tmp_f.append((op[0], op[1], op[2]))
    s_ = np.array(ori_x)
    for i,( pk_op_i, start_id, end_id) in enumerate(s_op):
        s_[start_id:end_id] = s_[start_id:end_id] + pk_op_i
	tmp_r.append( (start_id, end_id)  )
    cur_op = cal_op(s_)
    for pk_op_idx in range( len(cur_op) ):
        pk_op_c, pk_op_i = cur_op[ pk_op_idx ][:2]
        if pk_op_c == 1 and pk_op_i == 0:
	        print cur_op
	        exit()
        for start in range( pk_op_idx + 1 ):
            for end in range( pk_op_idx, len(cur_op) ):
                start_id, end_id = cur_op[ start ][2], cur_op[ end ][3]
                if (pk_op_i, start_id, end_id) in tmp_f:
                    continue
                tmp_x = np.array(list( s_ ))
                tmp_x[start: end] = tmp_x[start: end] * pk_op_c + pk_op_i
                candidate_op.append( [ pk_op_i, start_id, end_id] )
    if option:
    	return len(candidate_op), candidate_op
    return len(candidate_op)

ans, stateT = {}, {}
weight_t = 0
iterations, batch = 0, 0
h_stack = []

def heuristic_search(pre_op, depth, cur_weight):
	global iterations, ori_x, start_c, min_op, bactch_idx
	size = cal_state([], [])
	if "" not in stateT:
		stateT[ "" ] = [ [], size ]
	batch = 0
	# range and operations
	while True:
		s0 = stateTrans(pre_op)
		filter_ = []
		# print s0
		if s0 in stateT:
			filter_ = stateT[s0][0]
		size, c_op = cal_state(pre_op, [], True)
		if size <= 1:
			batch += 1
			if size == 1:
				pre_op.append(c_op[0])
			cur_x = np.array(ori_x)
			for i in range(len(pre_op)):
				add_op, min_, max_ = pre_op[i]
				cur_x[min_:max_] = (cur_x[min_:max_] + add_op)
			if size == 0:
				print cur_x == dst_x
				exit(0)
			if np.linalg.norm(cur_x - dst_x) == 0:
				# print len(pre_op), pre_op, "here"
				ans[bactch_idx].append( (len(pre_op), batch, list(pre_op)) )
			pre_op.pop()
			if np.linalg.norm(cur_x - dst_x) == 0:
				min_op = min( min_op, len(pre_op)+1)
				stateT[s0] = [[], -1]
				if len(pre_op):
					pre_op.pop()
				s = stateTrans(pre_op)
				if not s0[len(s):] in stateT[s][0]:
					stateT[s][0].append(s0[len(s):])
				ans[bactch_idx].sort()
				# print ans[bactch_idx][0]
				continue

		if s0 not in stateT:
			stateT[s0] = [[], size]
		if (s0 in stateT and stateT[s0][1] <= 0) or len(stateT[s0][0]) >= stateT[s0][1]:
			if s0 == "":
			    break
			pre_op.pop()
			# print "skyfall", stateT[s0], s0
			s = stateTrans(pre_op)
			if s0[len(s):] not in stateT[s][0]:
			    stateT[s][0].append( s0[len(s):]  )
			batch += 1
			continue

		end_t = clock()
		if end_t - start_t > 30:
			ans[bactch_idx].sort()
			print len(stateT.keys())
			print ans, dataset[ bactch_idx ][5], end_t - start_t, batch
			f = open("sb_sd_", "w")
			f.write(json.dumps(dataset[ bactch_idx ]))
			f.write("\n")
			f.write(json.dumps(ans[bactch_idx]))
			f.write("\n\n")
			f.close()
			return ans[bactch_idx][0], dataset[bactch_idx][5]

		cur_x = np.array(ori_x)
		for i in range(len(pre_op)):
			add_op, min_, max_ = pre_op[i]
			cur_x[min_:max_] = (cur_x[min_:max_] + add_op)

		# reach target
		if np.linalg.norm(cur_x - dst_x) == 0:
			global _min
			depth = len(pre_op)
			if (depth, pre_op) not in ans:
				ans[bactch_idx].append( (depth, batch, pre_op)  )
				# print depth, pre_op, "there"
				ans[bactch_idx].sort()
			min_op = min(depth, min_op)
			batch += 1
			s = stateTrans(pre_op)
			stateT[s] = [[], -1]
			if len(pre_op):
				pre_op.pop()
			s = stateTrans(pre_op)
			stateT[s] = [[], -1]
			if len(pre_op):
				pre_op.pop()
			s0 = stateTrans(pre_op)
			action = ""
			continue

		# backtracing
		if len(globla_hp):
			hp_top = globla_hp[0]
			depth = len(pre_op)
			if (hp_top[0] < cur_weight and depth > hp_top[1]) or depth + 1 >= min_op or len(pre_op) >= min_op:
				# ans[bactch_idx].append( (depth+10, batch, pre_op)  )
				tmp_op = list(pre_op)
				batch += 1
				hp.heappop(globla_hp)
				tmp_op = hp_top[2]
				cur_weight = hp_top[0]
				if len(pre_op) >= min_op: 
					s = stateTrans(pre_op)
					stateT[s] = [[], -1]
					if len(pre_op):
						pre_op.pop()
					s0 = stateTrans(pre_op)
					if not s[len(s0):] in stateT[s0][0]:
						stateT[s0][0].append(s[len(s0):])
				pre_op = list(tmp_op)
				del tmp_op
				continue

		cur_op, pre_add = [], float('nan')
		for i in range(len(cur_x)):
			cur_add = dst_x[i] - cur_x[i]
			# if cur_add == 0:
			# 	continue
			if cur_add == pre_add:
				cur_op[-1][2] = i + 1
			else:
				cur_op.append( [cur_add, i, i + 1] )
				pre_add = cur_add

		if len(cur_op) == 1 and cur_add == 0:
			ans.sort()
			ans[bactch_idx].append( (depth, batch, pre_op)  )
			s = stateTrans(pre_op)
			stateT[s] = [[], -1]
			if len(pre_op):
				pre_op.pop()
			s = stateTrans(pre_op)
			stateT[s] = [[], -1]
			if len(pre_op):
				pre_op.pop()
			s0 = stateTrans(pre_op)
			action = ""
			if not s[len(s0):] in stateT[s0][0]:
				stateT[s0][0].append(s[len(s0):])
			hp_top = globla_hp[0]
			batch += 1
			pre_op = hp_top[2]
			cur_weight = hp_top[0]
			hp.heappop(globla_hp)
			continue


		tmp_f, tmp_r = [], []
		filter_ = []
		if s0 in stateT:
			filter_ = stateT[s0][0]
		if len(filter_) > 0:
         #print filter_
			for op in filter_:
				if len(op) == 0:
					continue
				op = stateTrans(op)[0]
				tmp_f.append((op[0], op[1], op[2]))
		for pk_op_i, start_id, end_id in pre_op:
			tmp_r.append( (start_id, end_id)  )
		# expand to the next state
		for pk_op_idx in range( len(cur_op) ):
			pk_op = cur_op[ pk_op_idx ][0]
			if pk_op == 0:
				continue
			for start in range( pk_op_idx + 1 ):
				for end in range( pk_op_idx, len(cur_op) ):
					start_id, end_id = cur_op[ start ][1], cur_op[ end ][2]
					if (start_id, end_id) in tmp_r or ( pk_op,  start_id, end_id) in tmp_f:
						continue
					tmp_x, tmp_op = np.array(list( cur_x )), list(pre_op)
					tmp_x[ start_id:end_id ] += pk_op
					tmp_op.append(   [pk_op, start_id, end_id] )
					try:
						weight, dist_1 = weigh( tmp_x,depth+1 )
					except:
						print weigh(tmp_x, depth+1)
						print tmp_x
						exit()
					# print tmp_x, tmp_op, weight
					if len(cur_op) <= dist_1:
						continue
					item = (weight, depth+1, tmp_op)
					if item not in globla_hp:
						hp.heappush( globla_hp, item )
					del tmp_x, tmp_op, weight
		# random select
		#for rs_idx in range( len(globla_hp) ):
		# 	weight_t.append( globla_hp[rs_idx][0] )
		# weight_t = np.array(weight_t)
		# weight_t = max(weight_t) * 1.0 / weight_t
		# weight_t = softmax(weight_t) * 10
		# weight_to = np.sum(weight_t)
		# print weight_t, weight_to

		weight_t, _max = 0, 0
		if len(globla_hp) == 0:
			ans[bactch_idx].sort()
			print len(stateT.keys())
			print ans, dataset[ bactch_idx ][5], end_t - start_t, batch
			f = open("dataset_supervise", "a")
			f.write(json.dumps(dataset[ bactch_idx ]))
			f.write("\n")
			f.write(json.dumps(ans[bactch_idx]))
			f.write("\n\n")
			f.close()
			return ans[bactch_idx][0], dataset[bactch_idx][5]

		for rs_idx in range(len(globla_hp)):
		 	_max = max(_max, globla_hp[rs_idx][0])
		for rs_idx in range(len(globla_hp)):
		 	weight_t += _max / globla_hp[rs_idx][0]

		rs = np.random.randint(weight_t)
		tmp_rs = 0
		for rs_idx in range(len(globla_hp)):
		 	tmp_rs += _max / globla_hp[rs_idx][0]
		 	if tmp_rs > rs:
		 		rs = rs_idx
		 		break
		hp_top = globla_hp[ rs ]
		globla_hp.pop( rs )
		choice = np.random.choice
		# hp_top = hp.heappop(globla_hp)
		if np.random.choice(2, 1, p=np.array([0.5, 0.5]))[0]:
			hp.heappush( globla_hp, hp_top )
			hp_top = hp.heappop(globla_hp)
		# print hp_top

		add_op, min_, max_ = hp_top[2][-1]
		cur_weight = hp_top[0]
		pre_op = hp_top[2]
		depth = hp_top[1]
		s = stateTrans(pre_op)
		filter_ = []
		if s in stateT:
			filter_ = stateT[s][0]
		size = cal_state(pre_op, filter_)
		if not s in stateT:
			stateT[s] = [[], size]
		del cur_x
		continue
	# return heuristic_search( hp_top[2], hp_top[1], hp_top[0])

# bactch_idx = 50
for bactch_idx in range(60):
	stateT = {}
	ori_x = np.array(dataset[bactch_idx][3])
	dst_x = np.array(dataset[bactch_idx][4])
	min_op = dataset[bactch_idx][0] * 2 - 1
	globla_hp, n = [], 60
	start_c = 0
	start_t = clock()
	ori_x, dst_x = np.array(ori_x), np.array(dst_x)
	hp.heapify(globla_hp)
	ans[bactch_idx] = []
	print heuristic_search([], 0, float('inf'))
	exit()