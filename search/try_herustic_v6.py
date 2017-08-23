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
import graphviz as gv

dataset = []
with open("dataset_lt", 'r') as infile:
	for line in infile:
		dataset.append(json.loads(line))
# ori_x = np.random.randint(0, 100, 100)
# dst_x = np.array(ori_x)
# dst_x[10:50] = dst_x[10:50] * 3 + 2
# dst_x[20:60] = dst_x[20:60] * -2 + 1
# dst_x[0:70] = dst_x[0:70] * 6 + 3
# dst_x[67:85] = dst_x[67:85] * -11 + 13
# dst_x[75:80] = dst_x[75:80] * -9 + 10
# dst_x[85:90] = dst_x[85:90] * -10 + 100
#print dst_x
# start_c = 0
# tmp_s = clock()
# tmp_x, base_o = np.array(ori_x), 0
# for i in reversed(range(len(dst_x))):
# 	cur_a = dst_x[i] - tmp_x[i]
# 	tmp_x[:i+1] += cur_a
# 	if cur_a != 0:
# 		base_o += 1
# print clock() - tmp_s, base_o

def cal_op(s, ori_x):
	# global ori_x
	pre_coef = 0
	cur_op = [ [ 0, dst_x[0], 0, 1] ]
	for i in range( 1, len(s) ):
		o_tuple, d_tuple = ori_x[i], s[i]
		try:
			coef = float(d_tuple - s[i-1]) / (o_tuple - ori_x[i-1])
		except:
			if d_tuple == s[i-1] and o_tuple == ori_x[i-1]:
				cur_op[-1][3] = i+1
				continue
			else:
				if i != len(s) - 1:
					cur_op.append( [ 0, d_tuple, i, i+1] )
					pre_coef = 0
				else:
					cur_op.append( [ 1.0, 0.0, i, i+1] )
				continue
			#pre_coef = float('nan')
		intercept = float(s[i]) - o_tuple*coef
		if coef == pre_coef or pre_coef == 0:
			cur_op[-1][:2] = coef, intercept
			cur_op[-1][3] = i+1
			pre_coef = coef
		else:
			if i != len(s) - 1:
				cur_op.append( [ 0, d_tuple, i, i+1] )
				pre_coef = 0
			else:
				cur_op.append( [ 1.0, 0.0, i, i+1] )
	return cur_op

def cal_true_para(cur, pre):
	cur_c, cur_i = cur
	pre_c, pre_i = pre
	true_c = cur_c / pre_c
	true_i = cur_i - true_c * pre_i
	return true_c, true_i

def check_op(s_op, filter_):
	global dst_xs, ori_x
	tmp_r = [(item[2], item[3]) for item in s_op]
	tmp_f = []
	if len(filter_) > 0:
		for op in filter_:
			if len(op) == 0:
				continue
			op = stateTrans(op)[0]
			tmp_f.append((op[2], op[3]))
	cur_x = np.array(dst_x)
	for pk_op_c, pk_op_i, min_, max_ in s_op:
		cur_x[min_:max_] = (cur_x[min_:max_] - pk_op_i) / pk_op_c
	cur_op = cal_op(cur_x, ori_x)
	candidate_range = []
	for start_idx, s_op in enumerate(cur_op):
		if s_op[:2] == [1.0, 0.0]:
			continue
		if s_op[0] == 0.0:
			print cur_op
			exit()
		left_neigh, right_neigh = True, True
		if start_idx == 0 or cur_op[ start_idx-1 ][:2] == [1.0, 0.0]:
			left_neigh = False
		for end_idx in range(start_idx, len(cur_op)):
			e_op = cur_op[end_idx]
			if e_op[:2] == [1.0, 0.0]:
				continue
			if end_idx == len(cur_op)-1 or cur_op[ end_idx+1 ][:2] == [1.0, 0.0]:
				right_neigh = False
			candidate_range.append([ start_idx, end_idx, left_neigh, right_neigh ])
	candidate_op = []
	for start_idx, end_idx, left_neigh, right_neigh in candidate_range:
		# one block
		if end_idx - start_idx == 0:
			if (left_neigh and right_neigh) and (cur_op[start_idx-1][:2] == cur_op[start_idx+1][:2]):
				true_c, true_i = cal_true_para( cur_op[start_idx][:2], cur_op[start_idx-1][:2] )
				candidate_op.append( [ true_c, true_i, cur_op[start_idx][2], cur_op[start_idx][3] ] )
			elif not (left_neigh or right_neigh):
				candidate_op.append( cur_op[start_idx] )
			elif (not left_neigh) and right_neigh:
				true_c, true_i = cal_true_para( cur_op[start_idx][:2], cur_op[end_idx+1][:2] )
				candidate_op.append( [ true_c, true_i, cur_op[start_idx][2], cur_op[start_idx][3] ] )
			elif (not right_neigh) and left_neigh:
				true_c, true_i = cal_true_para( cur_op[start_idx][:2], cur_op[start_idx-1][:2] )
				candidate_op.append( [ true_c, true_i, cur_op[start_idx][2], cur_op[start_idx][3] ] )
		# more than one block
		else:
			if cur_op[start_idx][:2] == cur_op[end_idx][:2] and end_idx - start_idx > 1:
				true_c, true_i = cur_op[start_idx][:2]
				candidate_op.append( [ true_c, true_i, cur_op[start_idx][2], cur_op[end_idx][3] ] )	
			elif left_neigh and right_neigh:
				true_c_0, true_i_0 = cal_true_para( cur_op[start_idx][:2], cur_op[start_idx-1][:2] )
				true_c_1, true_i_1 = cal_true_para( cur_op[end_idx][:2], cur_op[end_idx+1][:2] )
				# if true_c_1 == true_c_0 and true_i_1 == true_i_0:
				candidate_op.append( [ true_c_0, true_i_0, cur_op[start_idx][2], cur_op[end_idx][3] ] )
				candidate_op.append( [ true_c_1, true_i_1, cur_op[start_idx][2], cur_op[end_idx][3] ] )				
			elif (not left_neigh) and right_neigh:
				true_c_0, true_i_0 = cur_op[start_idx][:2]
				# true_c_0, true_i_0 = cal_true_para( cur_op[end_idx][:2], cur_op[end_idx+1][:2] )
				# if true_c == true_c_0 and true_i == true_i_0:
				candidate_op.append( [ true_c_0, true_i_0, cur_op[start_idx][2], cur_op[end_idx][3] ] )			
			elif (not right_neigh) and left_neigh:
				true_c_0, true_i_0 = cur_op[end_idx][:2]
				# true_c_0, true_i_0 = cal_true_para( cur_op[start_idx][:2], cur_op[start_idx-1][:2] )
				# if true_c == true_c_0 and true_i == true_i_0:
				candidate_op.append( [ true_c_0, true_i_0, cur_op[start_idx][2], cur_op[end_idx][3] ] )
			elif end_idx - start_idx > 1:
				true_c, true_i = cur_op[start_idx][:2]
				candidate_op.append( [ true_c, true_i, cur_op[start_idx][2], cur_op[end_idx][3] ] )
				true_c_0, true_i_0 = cur_op[end_idx][:2]
				if not (true_c == true_c_0 and true_i == true_i_0):
					candidate_op.append( [ true_c_0, true_i_0, cur_op[start_idx][2], cur_op[end_idx][3] ] )
	if not len(candidate_op):
		candidate_op = [item for item in cur_op if item[:2] != [1.0, 0.0]]
	# print "candy", candidate_op, tmp_f, filter_
	candidate_op = [item for item in candidate_op if (item[2], item[3]) not in tmp_f and (item[2], item[3]) not in tmp_r ]
	return candidate_op

def weigh(cur_x, depth):
	global _min, ori_x
	cur_op = cal_op(cur_x, ori_x)
	tmp_op = []
	assign = False
	for op in cur_op:
		tmp_ = (round(op[0], 3), round(op[1], 3))
		if tmp_ != (1.0, 0.0) and tmp_ not in tmp_op:
			tmp_op.append( tmp_ )
		if tmp_[0] == 0.0:
			assign = True
	cur_op = [item for item in cur_op if item[:2] != [1.0, 0.0]]
	dist, dist_1 = len(tmp_op) + 1, len(cur_op)
	if dist_1 == 0:
		return depth, dist_1, assign
	#weight = dist * dist_1 * 10
	#weight =  np.log2(dist*dist_1) * (100)
	weight = np.log2(dist*depth) * (100)
	return (weight, dist_1, assign)

def stateTrans(state):
	if isinstance(state, list):
		s = ""
		for pk_op_c, pk_op_i, start, end in state:
			s = s + str(pk_op_c) + "," + str(pk_op_i) + "," + str(start) + "," + str(end) + " "
		return s
	else:
		s = []
		ops = state.split(' ')
		for op in ops:
			if len(op) == 0:
				continue
			op_i = op.split(",")
			s.append([float(op_i[0]), float(op_i[1]), int(op_i[2]), int(op_i[3])])
		return s

def heuristic_search(pre_op, depth, cur_weight, test_idx):
	global stateT, global_hp, min_op, start_t, ori_x, dst_x, error, ori_x
	if test_idx not in error:
		error[ test_idx ] = []
	candidate_op = check_op([], [])
	stateT[ "" ] = [ [], len(candidate_op) ]
	cur_op = cal_op( dst_x, ori_x )
	tmp_op = [ item for item in cur_op if item[:2] != [1.0, 0.0] ]
	state, ans = [], [(len(tmp_op), 0, list(tmp_op), 0)]
	print cur_op
	print "candy", candidate_op
	min_op = len(cur_op)
	batch, rs = 1, 0
	while True:
		s0 = stateTrans(pre_op)
		if s0 in stateT:
			print pre_op, stateT[s0], len(global_hp)
		cur_x = np.array(dst_x)
		c_op = check_op(pre_op, [])
		for pk_op_c, pk_op_i, min_, max_ in pre_op:
			cur_x[min_:max_] = (cur_x[min_:max_] - pk_op_i) / pk_op_c
		if np.allclose(cur_x, ori_x):
			# print pre_op
			tmp_x = np.array(ori_x)
			for pk_op_c, pk_op_i, min_, max_ in reversed(pre_op):
				tmp_x[min_:max_] = (tmp_x[min_:max_] * pk_op_c) + pk_op_i
			if not np.allclose(tmp_x, dst_x):
				_t = 0
				for i in range(100):
					_t += np.abs(tmp_x[i]-dst_x[i])
				tmp_op = cal_op(tmp_x, dst_x)
				tmp_op = [item+[True] for item in tmp_op if item[:2] != [1.0, 0.0]]
				for item in pre_op:
					tmp_op.append(item)
				error[ test_idx ].append( len(tmp_op)-len(pre_op) )
				ans.append( (len(pre_op), batch, list(tmp_op), clock() - start_t) )
				print "wrong bye", _t, tmp_op
				min_op = min( min_op, len(pre_op))
				# print tmp_x == dst_x
				# exit()
			else:
				ans.append( (len(pre_op), batch, list(pre_op), clock() - start_t) )
				print (len(pre_op), batch, list(pre_op), clock() - start_t), len(ans)
				min_op = min( min_op, len(pre_op))

			for rs_idx, rs in enumerate(global_hp):
	 			if rs[2] == pre_op:
	 				global_hp.pop(rs_idx)
	 				break
			stateT[s0] = [[], -1]
			if len(pre_op):
				pre_op.pop()
			for rs_idx, rs in enumerate(global_hp):
	 			if rs[2] == pre_op:
	 				global_hp.pop(rs_idx)
	 				break
			s0 = stateTrans(pre_op)
			stateT[s0] = [[], -1]
			# tmp_op = list(pre_op)
			# while len(tmp_op) > len(pre_op) / 2:
			# 	for rs_idx, rs in enumerate(global_hp):
	 	# 			if rs[2] == tmp_op:
	 	# 				tmp_hp = list(global_hp[rs_idx])
	 	# 				tmp_hp[0] += len(tmp_op)*10
	 	# 				global_hp[rs_idx] = tuple(tmp_hp)
	 	# 				break
	 	# 		tmp_op.pop()
			if len(pre_op):
				pre_op.pop()
			s = stateTrans(pre_op)
			if not s0[len(s):] in stateT[s][0]:
				stateT[s][0].append(s0[len(s):])
			batch += 1
			if len(global_hp):
				hp_top = global_hp[0]
				pre_op = hp_top[2]
				# cur_weight = hp_top[0]
				# hp.heappop(global_hp)
			continue

		if s0 not in stateT:
			stateT[s0] = [[], len(c_op)]
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
		
		candidate_op = check_op(pre_op, stateT[s0][0])
		# print candidate_op
		end_t = clock()
		if end_t - start_t > 60:
			ans.sort()
			print len(stateT.keys())
			print ans[0], end_t - start_t, batch
			return ans

		# backtrace
		if len(global_hp):
			hp_top = global_hp[0]
			depth = len(pre_op)
			# tmp_choices, tmp_choice = [0], 0
			# for tmp_idx in range(len(global_hp)):
			# 	if global_hp[tmp_idx][0] <= hp_top[0]:
			# 		tmp_choices.append(tmp_idx)
			# 	else:
			# 		break
			# tmp_choice = tmp_choices[ np.random.randint(len(tmp_choices)) ]
			# if np.random.choice(2, 1, p=np.array([0.1, 0.9]))[0]:
			# 	hp_top = global_hp[tmp_choice]
			if depth+1 >= min_op or len(pre_op) >= min_op:
			# if (hp_top[0] < cur_weight and depth > hp_top[1]) or depth+1 >= min_op or len(pre_op) >= min_op:
				batch += 1
				# hp.heappop(global_hp)
				tmp_op = hp_top[2]
				cur_weight = hp_top[0]
				if len(pre_op)+1 >= min_op:
					s = stateTrans(pre_op)
					for rs_idx, rs in enumerate(global_hp):
			 			if rs[2] == pre_op:
			 				global_hp.pop(rs_idx)
			 				break
					# print s, "bi", pre_op
					stateT[s] = [[], -1]
					if len(pre_op):
						pre_op.pop()
					s0 = stateTrans(pre_op)
					if not s[len(s0):] in stateT[s0][0]:
						stateT[s0][0].append(s[len(s0):])
				# tmp_op_ = list(pre_op)
				# while len(tmp_op_) > len(pre_op) / 2:
				# 	for rs_idx, rs in enumerate(global_hp):
		 	# 			if rs[2] == tmp_op_:
		 	# 				tmp_hp = list(global_hp[rs_idx])
		 	# 				tmp_hp[0] += len(tmp_op_)*10
		 	# 				global_hp[rs_idx] = tuple(tmp_hp)
		 	# 				break
		 	# 		tmp_op_.pop()
				pre_op = list(tmp_op)
				# if np.random.choice(2, 1, p=np.array([0.2, 0.8]))[0]:
				# 	pre_op = []
				del tmp_op
				continue

		depth = len(pre_op)
		cur_op = cal_op(cur_x, ori_x)
		next_op = []
		# print candidate_op
		for pk_op_c, pk_op_i, start_id, end_id in candidate_op:
			tmp_x, tmp_op = np.array(list( cur_x )), list(pre_op)
			tmp_x[ start_id:end_id ] = (tmp_x[ start_id:end_id ] - pk_op_i) / pk_op_c
			tmp_op.append(  [pk_op_c, pk_op_i, start_id, end_id] )
			weight, dist_1, assign = weigh( tmp_x, depth+1 )
			if len(cur_op) <= dist_1 or assign:
				continue
			item = (weight, depth+1, tmp_op)
			flag, flag_item = False, item
			for rs_idx, rs in enumerate(global_hp):
		 		if rs[2] == tmp_op:
		 			flag, flag_item = True, rs
		 	if not flag:
		 		hp.heappush( global_hp, item )
			next_op.append( flag_item )
			del tmp_x, tmp_op, weight

		if len(global_hp) == 0:
			print len(stateT.keys())
			ans.sort()
			print ans[0], end_t - start_t, batch
			return ans

		next_op.sort()
		next_proba, weight_t, _max, tmp_rs = [], 0, 0, 0
		# print pre_op, "next", next_op
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
		if np.random.choice(2, 1, p=np.array([0.5, 0.5]))[0]:
			hp_top = next_op[0]
		# weight_t, _max = 0, 0
		# for rs_idx in range(len(global_hp)):
		#  	_max = max(_max, global_hp[rs_idx][0]*5)
		# for rs_idx in range(len(global_hp)):
		#  	weight_t += _max / global_hp[rs_idx][0]
		# try:
		# 	rs = np.random.randint(weight_t)
		# except:
		# 	print global_hp
		# 	exit()
		# tmp_rs = 0
		# for rs_idx in range(len(global_hp)):
		#  	tmp_rs += _max / global_hp[rs_idx][0]
		#  	if tmp_rs > rs:
		#  		rs = rs_idx
		#  		break
		# hp_top = global_hp[ rs ]
		# # global_hp.pop( rs )
		# # hp_top = hp.heappop(global_hp)
		# if np.random.choice(2, 1, p=np.array([0.5, 0.5]))[0]:
		# 	# hp.heappush( global_hp, hp_top )
		# 	# hp_top = hp.heappop(global_hp)
		# 	hp_top = global_hp[0]

		# print hp_top
		cur_weight = hp_top[0]
		pre_op = hp_top[2]
		depth = hp_top[1]
		s = stateTrans(pre_op)
		filter_ = []
		if s in stateT:
			filter_ = stateT[s][0]
		candidate_op = check_op(pre_op, filter_)
		size = len(candidate_op)
		if not s in stateT:
			stateT[s] = [[], size]
		del cur_x
		continue
	return ans

def render_state_tree(stateT):
	g1 = gv.Graph(format='pdf')
	nodes = []
	for node in stateT:
		ops = stateTrans(node)
		if len(ops) == 0:
			g1.node(str(ops))
			continue
		if ops[0] not in nodes:
			g1.node(str(ops[0]))
			nodes.append(ops[0])
		for op_idx in range(1, len(ops)):
			if ops[op_idx] not in nodes:
				g1.node(str(ops[op_idx]))
				nodes.append(ops[op_idx])
			g1.edge(str(ops[op_idx-1]), str(ops[op_idx]))
	fn = g1.render(filename='search\wc_md\stateT_2')

error = {}
f = open("lt_ans_w", "a")
for dataset_idx, dataset_data in enumerate(dataset):
	if dataset_idx < 42:
		continue
	if dataset_idx == 48:
		break
	print dataset_data[0], dataset_data[3]
	ori_x = dataset_data[1]
	dst_x = dataset_data[2]
	stateT = {}
	global_hp = []
	hp.heapify(global_hp)
	start_t = clock()
	ans = heuristic_search([], 0, float('inf'), dataset_data[0])
	render_state_tree(stateT)
	f.write(json.dumps(ans))
	f.write("\n")
	ans.sort()
	print ans[0], dataset_data[0]
	# for item in ans:
	# 	op = item[2]
	# 	cur_x = np.array(ori_x)
	# 	for pk_op_c, pk_op_i, min_, max_ in reversed(op):
	# 		cur_x[min_:max_] = (cur_x[min_:max_] * pk_op_c) + pk_op_i
	# 	if np.linalg.norm(cur_x - dst_x) != 0:
	# 		print "wrong bye"
			# print cur_x == dst_x
# f.close()
open("error_lt", "w").write(json.dumps( error ))