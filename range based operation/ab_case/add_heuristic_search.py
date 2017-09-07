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
	diff_delta = 1
	tmp = []
	for i in range( len(cur_x) ):
		dis = cur_x[i] - dst_x[i]
		if abs(dis*1.0) > 1e-6 and dis not in tmp:
			tmp.append(dis)
	diff_delta += len(tmp)
	diff_blocks = 0
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
	diff_blocks = len(cur_op)
	if diff_blocks == 0:
		return depth, diff_blocks
	#weight = dist * dist_1 * 10
	#weight =  np.log2(dist*dist_1) * (100)
	weight = np.log2(diff_delta*depth)*diff_blocks * (100)
	return (weight, diff_blocks)

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
			try:
				s.append([float(op_i[0]), int(op_i[1]), int(op_i[2])])
			except:
				print ops, s, op_i
				exit()
		return s

def cal_op(s):
    global dst_x
    pre_coef, pre_add = float('nan'), float('nan')
    cur_op = []
    for i in range( 0, len(s) ):
        o_tuple, d_tuple = s[i], dst_x[i]
	cur_add = dst_x[i] - s[i]
  #       if cur_add == 0:
		# continue
        if cur_add == pre_add:
               cur_op[-1][2] = i+1
        else:
               cur_op.append( [ cur_add, i, i+1] )
               pre_add = cur_add
    return cur_op

def check_zero(zero_idx, start_id, end_id):
	for tmp_c, z_s, z_e in zero_idx:
		if start_id <= z_s and end_id >= z_e:
			return True
	return False

def cal_state(s_op, filter_, option=False):
    global ori_op, ori_x
    ss, candidate_op, true_op = [], [], []
    tmp_r, tmp_f = [], []
    if len(filter_) > 0:
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
    zero_idx = [item for item in cur_op if item[0] == 0.0]
    for pk_op_idx in range( len(cur_op) ):
        pk_op_i = cur_op[ pk_op_idx ][0]
        if pk_op_i == 0:
	        continue
        for start in range( pk_op_idx + 1 ):
            for end in range( pk_op_idx, len(cur_op) ):
                start_id, end_id = cur_op[ start ][1], cur_op[ end ][2]
                if check_zero(zero_idx, start_id, end_id):
                	continue
                # if ( start_id, end_id ) in tmp_r or (pk_op_i, start_id, end_id) in tmp_f:
                    #print "cao", s_op,"sb", cur_op
                if (pk_op_i, start_id, end_id) in tmp_f:
                    continue
                tmp_x = np.array(list( s_ ))
                tmp_x[start: end] = tmp_x[start: end] + pk_op_i
                candidate_op.append( [ pk_op_i, start_id, end_id] )
    if option:
    	return len(candidate_op), candidate_op
    return len(candidate_op)

ans, stateT = {}, {}
weight_t = 0
iterations, batch = 0, 0
h_stack = []

def heuristic_search(pre_op, depth, cur_weight):
	global iterations, ori_x, start_c, min_op, bactch_idx, start_t
	size = cal_state([], [])
	if "" not in stateT:
		stateT[ "" ] = [ [], size ]
	batch = 0
	# range and operations
	tmp_op = cal_op(ori_x)
	tmp_op = [item for item in tmp_op if item[0] != 0.0]
	ans = [(len(tmp_op), 0, list(tmp_op), 0)]
	min_op = len(tmp_op)
	print min_op, clock() - start_t
	while True:
		s0 = stateTrans(pre_op)
		filter_ = []
		# print s0
		cur_x = np.array(ori_x)
		for add, start, end in pre_op:
			cur_x[start:end] += add
		if np.linalg.norm(cur_x - dst_x) == 0:
			print (len(pre_op), batch, list(pre_op), clock() - start_t)
			min_op = min(min_op, len(pre_op))
			ans.append( (len(pre_op), batch, list(pre_op), clock() - start_t) )
			stateT[s0] = [[], -1]
			if len(pre_op):
				pre_op.pop()
			s0 = stateTrans(pre_op)
			stateT[s0] = [[], -1]
			tmp_op = list(pre_op)
			for rs_idx, rs in enumerate(global_hp):
	 			if rs[2] == pre_op:
	 				global_hp.pop(rs_idx)
	 				break
			if len(pre_op):
				pre_op.pop()
			s = stateTrans(pre_op)
			if not s0[len(s):] in stateT[s][0]:
				stateT[s][0].append(s0[len(s):])
			batch += 1
			if len(global_hp):
				hp_top = global_hp[0]
				pre_op = hp_top[2]
				if np.random.choice(2, 1, p=np.array([0.1, 0.9]))[0]:
					pre_op = []
			continue
		filter_ = []
		if s0 in stateT:
			filter_ =  stateT[s0][0]
		size, candidate_op = cal_state(pre_op, filter_, True)
		if s0 not in stateT:
			stateT[s0] = [[], len(candidate_op)]
		elif (s0 in stateT and stateT[s0][1] <= 0) or len(stateT[s0][0]) >= stateT[s0][1]:
			if s0 == "":
				break
			pre_op.pop()
			# print "skyfail", stateT[s0], s0
			s = stateTrans(pre_op)
			if s0[len(s):] not in stateT[s][0]:
				stateT[s][0].append( s0[len(s):]  )
			batch += 1
			continue

		if clock() - start_t > 60:
			ans.sort()
			print len(stateT.keys())
			print ans[0], clock() - start_t, batch
			return ans

		if len(global_hp):
			hp_top = global_hp[0]
			depth = len(pre_op)
			if depth+1 >= min_op or len(pre_op) >= min_op:
				batch += 1
				tmp_op = hp_top[2]
				cur_weight = hp_top[0]
				if len(pre_op)+1 >= min_op:
					s = stateTrans(pre_op)
					for rs_idx, rs in enumerate(global_hp):
			 			if rs[2] == pre_op:
			 				global_hp.pop(rs_idx)
			 				break
					stateT[s] = [[], -1]
					if len(pre_op):
						pre_op.pop()
					s0 = stateTrans(pre_op)
					if not s[len(s0):] in stateT[s0][0]:
						stateT[s0][0].append(s[len(s0):])
				pre_op = list(tmp_op)
				if np.random.choice(2, 1, p=np.array([0.1, 0.9]))[0]:
					pre_op = []
				del tmp_op
				continue

		depth, next_op = len(pre_op), []
		for add, start, end in candidate_op:
			tmp_x, tmp_op = np.array(list( cur_x )), list(pre_op)
			tmp_x[start: end] += add
			tmp_op.append(  [add,  start, end] )
			weight, dist_1 = weigh( tmp_x,depth+1 )
			if len(cur_op) <= dist_1:
				continue
			item = (weight, depth+1, tmp_op)
			if item not in global_hp:
				hp.heappush( global_hp, item )
			next_op.append( item )
			del tmp_x, tmp_op, weight

		if len(global_hp) == 0:
			print len(stateT.keys())
			ans.sort()
			print ans[0], clock() - start_t, batch
			return ans
		next_op.sort()
		next_proba, weight_t, _max, tmp_rs = [], 0, 0, 0
		# print pre_op, "next", next_op
		# exit()
		# print pre_op,"candy", candidate_op
		for it in next_op:
			_max = max(_max, it[0]*5.0)
		for it in next_op:
		 	weight_t += _max / it[0]
 		try:
			rs = np.random.randint(weight_t)
		except:
			# print next_op, candidate_op, pre_op
			s = stateTrans(pre_op)
			stateT[s] = [[], -1]
			for rs_idx, rs in enumerate(global_hp):
	 			if rs[2] == pre_op:
	 				global_hp.pop(rs_idx)
	 				break
	 		if len(pre_op):
				pre_op.pop()
			s0 = stateTrans(pre_op)
			if not s[len(s0):] in stateT[s0][0]:
				stateT[s0][0].append(s[len(s0):])
			continue
		for rs_idx, it in enumerate(next_op):
		 	tmp_rs += _max / it[0]
		 	if tmp_rs > rs:
		 		rs = rs_idx
		 		break
		hp_top = next_op[rs]
		if np.random.choice(2, 1, p=np.array([0.2, 0.8]))[0]:
			hp_top = next_op[0]
		
		cur_weight = hp_top[0]
		pre_op = hp_top[2]
		depth = hp_top[1]
		s = stateTrans(pre_op)
		filter_ = []
		if s in stateT:
			filter_ = stateT[s][0]
		filter_ = []
		if s0 in stateT:
			filter_ =  stateT[s0][0]
		size = cal_state(pre_op, filter_)
		if not s in stateT:
			stateT[s] = [[], size]
		del cur_x
		continue
	ans.sort()
	return ans

dataset = []
with open("search/dataset_aab", 'r') as infile:
	for line in infile:
		dataset.append(json.loads(line))

error, ans_v = {}, {}
f = open("search/ans_aab_col1", "w")
for dataset_idx, dataset_data in enumerate(dataset):
	# if dataset_idx < 46:
	# 	continue
	# if dataset_idx == 47:
	# 	break
	print dataset_data[0], dataset_data[3]
	ori_x = np.array(dataset_data[1])[:,1] * 1.0
	dst_x = np.array(dataset_data[2]) * 1.0
	ori_x, dst_x = ori_x.reshape(-1, 1), dst_x.reshape(-1, 1)
	tmp_x = np.concatenate((ori_x, dst_x), axis=1)
	tmp_x = np.array(sorted(tmp_x,  key=lambda x:x[0]))
	ori_x, dst_x = np.zeros(shape=len(tmp_x)), tmp_x[:,1]

	stateT = {}
	global_hp = []
	hp.heapify(global_hp)
	start_t = clock()
	ans = heuristic_search([], 0, float('inf'))
	# render_state_tree(stateT)
	f.write(json.dumps(ans))
	# f.write(json.dumps(stateT))
	f.write("\n")
	ans.sort()
	print ""