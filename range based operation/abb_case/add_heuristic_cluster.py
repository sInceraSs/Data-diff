#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from time import clock
import heapq as hp
from sklearn import linear_model
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus, json, sys
from time import clock
reload(sys)
sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('white')

def parse_node(cur_delta):
	global ori_x
	clf = tree.DecisionTreeClassifier(max_depth=None)
	res = clf.fit(ori_x, cur_delta)
	yy, Dtree = res.classes_, res.tree_
	n_nodes = Dtree.node_count
	children_left = Dtree.children_left
	children_right = Dtree.children_right
	feature = Dtree.feature
	threshold = Dtree.threshold
	values = Dtree.value
	parent_node, decision_depth = {}, { 0:{ float('inf'):1,  float('-inf'):1 }, 1:{ float('inf'):1,  float('-inf'):1 } }
	is_leaves, node_depth = np.zeros(shape=n_nodes, dtype=bool), np.zeros(shape=n_nodes, dtype=np.int64)
	stack = [(0, -1)]  # seed is the root node id and its parent depth
	while len(stack) > 0:
	    node_id, parent_depth = stack.pop()
	    node_depth[node_id] = parent_depth + 1
	    # If we have a test node
	    parent_node[ children_left[node_id] ] = (node_id, 1)
	    parent_node[ children_right[node_id] ] = (node_id, 0)
	    if (children_left[node_id] != children_right[node_id]):
	    	decision_depth[ feature[node_id] ][ threshold[node_id] ] = parent_depth + 1
	        stack.append((children_left[node_id], parent_depth + 1))
	        stack.append((children_right[node_id], parent_depth + 1))
	    else:
	        is_leaves[node_id] = True
	parent_node[0] = [-1, -1]
	blobs, ori_blobs = {}, []
	for i in range(n_nodes):
		if is_leaves[i]:
			tmp_node, decision_path = i, [ float('-inf'), float('inf'), float('-inf'), float('inf') ]
			while tmp_node != -1:
				p_node = parent_node[tmp_node][0]
				if p_node == -1:
					break
				data = feature[p_node], threshold[p_node]
				d_idx = data[0] * 2 + parent_node[tmp_node][1]
				if parent_node[tmp_node][1]:
					decision_path[d_idx] = min(threshold[p_node], decision_path[d_idx])
				else:
					decision_path[d_idx] = max(threshold[p_node], decision_path[d_idx])
				tmp_node = p_node
			label = yy[np.argmax(values[i])]
			if label not in blobs:
				blobs[label] = []
			blobs[label].append( decision_path )
			ori_blobs.append( [label] + decision_path )
	return blobs, ori_blobs, decision_depth

def check_valid(filter_blobs, blob):
	test_x_min, test_x_max, test_y_min, test_y_max = blob
	for x_min, x_max, y_min, y_max in filter_blobs:
		t_x_min, t_x_max = test_x_min > x_min and test_x_min < x_max, test_x_max > x_min and test_x_max <= x_max
		t_y_min, t_y_max = test_y_min > y_min and test_y_min < y_max, test_y_max > y_min and test_y_max <= y_max
		if (t_x_min or t_x_max) and (t_y_min or t_y_max):
			return False
	return True

def candidate_blobs(blobs, ori_blobs, filter_):
	candidate_blobs, filter_blobs, tmp_f, dus = [], [], [], []
	if len(filter_) > 0:
		for op in filter_:
			if len(op) == 0:
				continue
			op = stateTrans(op)[0]
			tmp_f.append((op[1], op[2], op[3], op[4]))
	if 0 in blobs:
		filter_blobs = blobs[0]
	x_candidate, y_candidate = [], []
	for item in ori_blobs:
		if item[0] != 0:
			x_candidate.append(item[1]), x_candidate.append(item[2])
			y_candidate.append(item[3]), y_candidate.append(item[4])
	x_candidate, y_candidate = np.unique(x_candidate), np.unique(y_candidate)
	x_ranges, y_ranges = [], []
	for x_min_id, x_min in enumerate(x_candidate):
		for x_max_id, x_max in enumerate(x_candidate[x_min_id:]):
			x_ranges.append((x_min, x_max))
	for y_min_id, y_min in enumerate(y_candidate):
		for y_max_id, y_max in enumerate(y_candidate[y_min_id:]):
			y_ranges.append((y_min, y_max))
	for x_min, x_max in x_ranges:
		for y_min, y_max in y_ranges:
			deltas, flag, duplicate = [], True, []
			# check for already done points
			if check_valid(filter_blobs, (x_min, x_max, y_min, y_max)):
				# check for each blobs
				for add, t_x_min, t_x_max, t_y_min, t_y_max in ori_blobs:
					if add == 0:
						continue
					total_cover = t_x_min >= x_min and t_x_max <= x_max and t_y_min >= y_min and t_y_max <= y_max
					total_seperate = t_x_min >= x_max or t_x_max <= x_min or t_y_min >= y_max or t_y_max <= y_min
					if not (total_cover or total_seperate):
						flag = False
						break
					elif total_cover:
						duplicate.append( str(t_x_min)+str(t_x_max)+str(t_y_min)+str(t_y_max) )
						deltas.append(add)
				if flag:
					duplicate.sort()
					for delta in deltas:
						tmp_blob = (delta, x_min, x_max, y_min, y_max)
						if tmp_blob not in candidate_blobs:# and duplicate not in dus:
							candidate_blobs.append(tmp_blob), dus.append(duplicate)
	candidate_blobs = [ item for item in candidate_blobs if (item[1], item[2], item[3], item[4]) not in tmp_f ]
	# print len(candidate_blobs)
	return candidate_blobs
	# for  start_blob in ori_blobs:
	# 	if start_blob[0] == 0:
	# 		continue
	# 	for end_blob in ori_blobs:
	# 		if end_blob[0] == 0:
	# 			continue
	# 		x_min, y_min = min(start_blob[1], end_blob[1]), min(start_blob[3], end_blob[3])
	# 		x_max, y_max = max(start_blob[2], end_blob[2]), max(start_blob[4], end_blob[4])
	# 		tmp_blob = (start_blob[0], x_min, x_max, y_min, y_max)
	# 		if check_valid(filter_blobs, tmp_blob) and tmp_blob not in candidate_blobs:
	# 			candidate_blobs.append(tmp_blob)
	# candidate_blobs = [ item for item in candidate_blobs if (item[1], item[2], item[3], item[4]) not in tmp_f ]
	# return candidate_blobs

def stateTrans(state):
	if isinstance(state, list):
		s = ""
		for label, x_min, x_max, y_min, y_max in state:
			s = s+ str(label) + ","  + str(x_min) + "," + str(x_max) + "," + str(y_min) + "," + str(y_max) + " "
		return s
	else:
		s = []
		ops = state.split(' ')
		for op in ops:
			if len(op) == 0:
				continue
			op_i = op.split(",")
			s.append([float(op_i[0]), float(op_i[1]), float(op_i[2]), float(op_i[3]), float(op_i[4])])
		return s

def weigh(cur_delta, depth, tmp_decision_depth):
	global _min, ori_x
	blobs, ori_blobs, decision_depth = parse_node(cur_delta)
	candidate_op = candidate_blobs(blobs, ori_blobs, "")
	
	tmp_op = []
	assign = False
	for op in ori_blobs:
		if op[0] != 0.0 and op not in tmp_op:
			tmp_op.append( op )
	diff_add, diff_blobs = len(np.unique(cur_delta)), len(tmp_op)
	if diff_blobs == 0:
		return depth, diff_blobs + 1, 0
	# weight = dist * depth * 10
	#weight =  np.log2(dist*dist_1) * (100)
	weight = diff_blobs*diff_add * (100) * np.log2(tmp_decision_depth)*depth
	# weight = diff_blobs*diff_add * (100) * np.log2(tmp_decision_depth)*depth
	return (weight, diff_blobs, len(candidate_op))

def heuristic_search(pre_op, depth):
	global stateT, min_op, start_t, ori_delta, ori_x
	blobs, ori_blobs, decision_depth = parse_node(ori_delta)
	candidate_op = candidate_blobs(blobs, ori_blobs, "")
	stateT, global_hp = {}, []
	hp.heapify(global_hp)
	start_t = clock()
	stateT[ "" ] = [ [], len(candidate_op) ]
	tmp_blobs = [item for item in ori_blobs if item[0] != 0]
	min_op = len(tmp_blobs)
	state, ans, batch = [], [(len(tmp_blobs), 0, list(tmp_blobs), 0)], 1
	print "ori_blobs",tmp_blobs
	while True:
		s0 = stateTrans(pre_op)
		# if s0 in stateT:
			# print pre_op, stateT[s0], len(global_hp)
		cur_delta = np.array(ori_delta)
		for add, x_min, x_max, y_min, y_max in pre_op:
			for idx, (point_x, point_y) in enumerate( ori_x ):
				if (point_x > x_min and point_x <= x_max) and (point_y > y_min and point_y <= y_max):
					cur_delta[idx] -= add
		tmp_delta = np.unique(cur_delta)
		if len(tmp_delta) == 1 and tmp_delta[0] == 0:
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

		blobs, ori_blobs, decision_depth = parse_node(cur_delta)
		filter_ = []
		if s0 in stateT:
			filter_ =  stateT[s0][0]
		candidate_op = candidate_blobs(blobs, ori_blobs, filter_)
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
		for add, x_min, x_max, y_min, y_max in candidate_op:
			tmp_delta, tmp_op = np.array(list( cur_delta )), list(pre_op)
			for idx, (point_x, point_y) in enumerate( ori_x ):
				if (point_x > x_min and point_x <= x_max) and (point_y > y_min and point_y <= y_max):
					tmp_delta[idx] -= add
			tmp_op.append(  [add, x_min, x_max, y_min, y_max] )
			tmp_decision_depth = decision_depth[0][x_min] + decision_depth[0][x_max] + decision_depth[1][y_min] + decision_depth[1][y_max]
			weight, diff_blobs, size = weigh( tmp_delta, depth+1, tmp_decision_depth )
			if len(ori_blobs) <= diff_blobs:
				continue
			item = (weight, depth+1, tmp_op, size)
			if item not in global_hp:
				hp.heappush( global_hp, item )
			next_op.append( item )
			del tmp_delta, tmp_op, weight

		if len(global_hp) == 0:
			print len(stateT.keys())
			ans.sort()
			print ans[0], end_t - start_t, batch
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

		size = hp_top[3]
		cur_weight = hp_top[0]
		pre_op = hp_top[2]
		depth = hp_top[1]
		s = stateTrans(pre_op)
		filter_ = []
		if s in stateT:
			filter_ = stateT[s][0]
		
		# candidate_op = check_op(pre_op, filter_)
		size = len(candidate_op)
		if not s in stateT:
			stateT[s] = [[], size]
		del cur_delta
		continue
	ans.sort()
	return ans
# x_x = np.random.randint(-50, 50, 100)
# x_y = np.random.randint(-50, 50, 100)
# ori_delta = np.zeros(shape=100)
# pk_range = [10, 20, 30, 40]
# for i in range(4):
# 	add = np.random.randint(-50, 50)
# 	for x_x_idx, x_x_v in enumerate(x_x):
# 		if (x_x_v > -1*pk_range[i] and x_x_v < pk_range[i]) and (x_y[x_x_idx] > -1*pk_range[i] and x_y[x_x_idx] < pk_range[i]):
# 			ori_delta[x_x_idx] += add
# print list(ori_delta)
# ori_delta.reshape(-1, 1)
# print np.unique(ori_delta)
# x_x = list(np.random.randint(-5, -1, 5)) + list(np.random.randint(2, 6, 10)) + list(np.random.randint(6, 10, 12)) + list(np.random.randint(2, 10, 6)) + list(np.random.randint(0, 2, 6)) + list(np.random.randint(0, 6, 6))
# x_y = list(np.random.randint(-2, 3, 5)) + list(np.random.randint(4, 10, 10)) + list(np.random.randint(2, 15, 12)) + list(np.random.randint(2, 4, 6)) + list(np.random.randint(2, 10, 6)) + list(np.random.randint(10, 12, 6))
# ori_delta = np.array([1]*5+[3]*10+[2]*30)
# x_x, x_y = np.array(x_x).reshape(-1, 1),  np.array(x_y).reshape(-1, 1)
# ori_x = np.concatenate((x_x, x_y), axis=1)
dataset = []
with open("search/dataset_aab", 'r') as infile:
	for line in infile:
		dataset.append(json.loads(line))

error = {}
f = open("search/ans_aab", "w")
for dataset_idx, dataset_data in enumerate(dataset):
	# if dataset_idx < 22:
		# continue
	# if dataset_idx == 1:
	# 	break
	print dataset_data[0], dataset_data[3]
	ori_x = np.array(dataset_data[1])
	ori_delta = np.array(dataset_data[2])
	start_t = clock()
	ans = heuristic_search([], 0)
	# render_state_tree(stateT)
	f.write(json.dumps(ans))
	f.write("\n")
	ans.sort()
	print ans[0], dataset_data[0]
	print ""

# start_t = clock()
# blobs, ori_blobs, decision_depth = parse_node(ori_delta)
# print blobs
# print ori_blobs
# print candidate_blobs(blobs, ori_blobs, "")
# print clock() - start_t
# # exit()
# ans = heuristic_search([], 0)
# print ans