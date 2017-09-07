# coding: utf-8

# import gym
import numpy as np
from keras.models import Model
from keras.layers import Dense, Reshape, Flatten, Input, Dropout, Activation
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from time import clock
import json, math

dataset = []
with open("dataset_lt", 'r') as infile:
	for line in infile:
		dataset.append(json.loads(line))

ori_x = np.array(dataset[47][1])
dst_x = np.array(dataset[47][2])
print dataset[47][3]
ori_dis = np.linalg.norm(ori_x - dst_x)
print ori_dis

def cal_op(s, ori_x, op=False):
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
    if op:
        cur_op = [item for item in cur_op if item[:2] != [1.0, 0.0]]  
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
                if not (true_c_0 == true_c and true_i_0 == true_i):
                    candidate_op.append( [ true_c_0, true_i_0, cur_op[start_idx][2], cur_op[end_idx][3] ] )
    if not len(candidate_op):
        candidate_op = [item for item in cur_op if item[:2] != [1.0, 0.0]]
    # print "candy", candidate_op, tmp_f, filter_
    candidate_op = [item for item in candidate_op if (item[2], item[3]) not in tmp_f and (item[2], item[3]) not in tmp_r ]
    return candidate_op

ori_op = len(cal_op(dst_x, ori_x))
min_op = min(ori_op, float('inf'))

print len(check_op([], [])), min_op, ori_op
print check_op([], [])
def cal_dis(s, cur_op):
    global ori_x, ori_dis, ori_op
    tmp = []
    num_op, num_diff, dis = 0, 0, 0
    for pk_op_c, pk_op_i, start, end in cur_op:
        tmp_ = (round(pk_op_c, 3), round(pk_op_i, 3))
        if tmp_ != (1.0, 0.0) and tmp_ not in tmp:
            tmp.append( tmp_ )
    num_op = len(tmp)+1
    dis = np.linalg.norm(s - ori_x)
    for i in range( len(s) ):
        if abs(s[i] - ori_x[i]) > 1e-6:
            num_diff += 1
    del tmp
    return  num_op*1.0/(ori_op*1.5),  num_diff*1.0/ len(dst_x), np.abs(dis)*1.0/ (np.abs(ori_dis) + np.abs(dis))

def cal_state(s_op, depth, filter_):
    global ori_op, dst_x, ori_x
    ss, candidate_op, true_op, next_op = [], [], [], []
    candidate_op =  check_op(s_op, filter_)
    s_ = np.array(dst_x)
    for i,(pk_op_c, pk_op_i, start_id, end_id) in enumerate(s_op):
        s_[start_id:end_id] = (s_[start_id:end_id]-pk_op_i) / pk_op_c
    c_op =  cal_op(s_, ori_x, True)
    for op_idx, op in enumerate(candidate_op):
        pk_op_c, pk_op_i, start, end = op
        tmp_x = np.array(list( s_ ))
        tmp_x[start: end] = (tmp_x[start: end] - pk_op_i) / pk_op_c
        tmp_op = cal_op(tmp_x, ori_x, True)
        if len(tmp_op) >= len(c_op):
            continue
        num_op, num_diff, dis = cal_dis( tmp_x, tmp_op )
        true_op.append(op)
        next_op.append( [np.log2(num_op*1.5*ori_op*(depth+1)), op] )
        ss += [ np.array([[  num_op, num_diff, dis, (depth+1)*1.0/min_op ]]) ]
    return ss, true_op, len(true_op), next_op

def select(size, rate):
    out = []
    for i in range(int(size * rate)):
        r = np.random.randint(size)
        if r not in out:
            out += [r]
    out.sort()
    return out

def pump(l, ind):
    out = []
    cnt = 0
    for i,j in enumerate(l):
        if cnt >= len(ind):
            out += [j]
            continue
        if i != ind[cnt]:
            out += [j]
        else:
            cnt += 1
    return out

def stateTrans(state):
	if isinstance(state, list):
		s = ""
		for pk_op_c, pk_op_i, start, end in state:
			s = s +str(pk_op_c)+","+ str(pk_op_i) + "," + str(start) + "," + str(end) + " "
		return s
	else:
		s = []
		ops = state.split(' ')
		for op in ops:
			if len(op) == 0:
				continue
			op_i = op.split(",")
			s.append([op_i[0], op_i[1], int(op_i[2]),int(op_i[3])])
		return s

	
class PGAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = 0.75
        self.learning_rate = 0.25
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.labels = []
        self.model = self._build_model()
        self.model.summary()
        self.replay_x = []
        self.replay_y = []
        for i in range(state_shape[0]):
            self.states += [[]]
            self.replay_x += [[]]
        
    def _build_model(self):
        # net1 = Dense(1, name='rank_layer', init='he_uniform')
        net1 = Dense(8, name='rank_layer1', init='he_uniform')
        net2 = Dense(4, name='rank_layer2', init='he_uniform')
        net3 = Dense(1, name='rank_layer3', init='he_uniform')
        outs = []
        inps = []
        for i in range(self.state_shape[0]):
            inp = Input(shape=[self.state_shape[1]])
            inps += [inp]
            rank = net2(net1(inp))
            rank = net3(Dropout(0.3)(rank))
#           rank = net1(inp)
            outs += [rank]
        out = Concatenate()(outs)
        prob = Activation(activation='softmax')(out)
        #prob = Dense(self.action_size, activation='softmax', init='he_uniform')(out)
        opt = Adam(lr = 0.005)
        model = Model(inputs=inps, outputs=prob)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        print(model.summary())
        return model

    def remember(self, state, action, prob, reward, aprob, label=[]):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.probs.append(aprob)
        self.gradients.append(np.array(y).astype('float32') - prob)
        for i,j in zip(self.states, state):
            i.append(j[0])
        self.rewards.append(reward)

    def act(self, state, lenth):
		_aprob, label = 0.0, []
		aprob = self.model.predict(state, batch_size=1).flatten()
		for i,item in enumerate(state):
			if np.sum(item[0]) != 0:
				_aprob += aprob[i]
			else:
				aprob[i] = 0
		if _aprob != 0.0:
			aprob[:] /= _aprob
		#self.probs.append(aprob)
		if np.sum(aprob) == 0:
			prob = np.array([0.05 for i in range(20)])
			return -1, prob, aprob
		else:
			prob = aprob / np.sum(aprob)
		action = np.random.choice(self.action_size, 1, p=prob)[0]
		return action, prob, aprob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        choice = select(len(self.replay_y), 0.3)
        self.replay_y = pump(self.replay_y, choice)
        self.replay_x = [pump(xx, choice) for xx in self.replay_x]
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / (1 + np.std(rewards - np.mean(rewards)))
        gradients *= rewards
        X = [np.array(i) for i in self.states]
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        for i, j in zip(self.replay_x, self.states):
            for jj in j:
                i.append(list(jj))
        self.replay_y += list(Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []
        self.model.train_on_batch([np.array(i) for i in self.replay_x], np.array(self.replay_y))
        for i in range(state_shape[0]):
            self.states += [[]]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class Maze(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.col = np.array(dst_x)
        self.op = []
        self.over = False
        self.step = 0
        return self.op

    def take_action(self, ss, candidate_op, next_op):
		global agent
		while True:
			action_choices, true_op, g_op = [], [], []
			if len(ss) <= 20:
				action_choices, true_op = list(ss), list(candidate_op)
				g_op = [(item[0], true_op.index(item[1])) for item in next_op if item[1] in true_op]
				for left_idx in range(len(action_choices), 20):
					action_choices += [ np.array([[ 0, 0, 0, 0 ]]) ]
				g_op.sort()
				action_id, prob, aprob = agent.act(action_choices, len(true_op))
				return action_id, prob, aprob, action_choices, true_op, g_op
			first_layer = int(math.ceil(len(ss)/20.0))
			each_layer, left_layer, flag_layer = len(ss) / first_layer, len(ss) % first_layer, 0
			for layer_idx in range(first_layer):
				start_id, end_id = layer_idx*each_layer+flag_layer, (layer_idx+1)*each_layer+flag_layer
				select_num = each_layer
				if flag_layer != left_layer:
					select_num = select_num+1
					flag_layer += 1
				tmp_choice, tmp_op = ss[ start_id:end_id ], candidate_op[start_id:end_id]
				for left_idx in range(len(tmp_choice), 20):
					tmp_choice += [ np.array([[ 0, 0, 0, 0 ]]) ]
				action_id, prob, aprob = agent.act(tmp_choice, len(true_op))
				action_ids = np.random.choice(20, 1, p=prob)[0:select_num]
				for item_id, item in enumerate(action_ids):
					action_choices.append( tmp_choice[item] ), true_op.append( tmp_op[item] )
			ss, candidate_op  = list(action_choices), list(true_op)
	
    def trace(self, episode):
        global stateT, cnt, curve, best, min_op, global_hp
        mx_op, mx_para = [], []
        mx_score = float('-inf')
        batch = 0
        para_s, score = [[], [], [], [], [], []], 0
        state = list(self.reset())
        filter_ = []
        if "" in stateT:
            filter_ = stateT[""][0]
            if (stateT[""][1] <= 0) or len(stateT[""][0]) >= stateT[""][1]:
                return -2, -1000
        x, true_op, size, next_op = cal_state(state, len(self.op), filter_)
        if "" not in stateT:
            stateT[ "" ] = [ [], size ]
        #print stateT
        while True:
            #if batch % 5 == 0:
            #    print batch
            if batch >= 1:
                break
            s0 = stateTrans(state)
            if (s0 in stateT) and (stateT[s0][1] <= 0 or len(stateT[s0][0]) >= stateT[s0][1]): 
                if s0 == "":
                    break
                #print stateT[s0], "s0 xie"
                state.pop()
                s = stateTrans(state)
                if s0[len(s):] not in stateT[s][0]:
                    stateT[s][0].append( s0[len(s):]  )
                x, true_op, size, next_op = cal_state(state, len(state), stateT[s][0])
                para_s, score = [[], [], [], [], [], []], 0
                batch += 1
                continue
            #print  len(true_op) == len(next_op)
            action_id, prob, aprob, action_choices, true_op, next_op = self.take_action(x, true_op, next_op)
            #print len(true_op) == len(next_op), next_op ,len(true_op), action_id, batch#, state, len(state), s0
            if action_id == -1 or len(true_op) == 0:
                stateT[s0][1] = -1
                state.pop()
                s = stateTrans(state)
                if not s0[len(s):] in stateT[s][0]:
                    stateT[s][0].append(s0[len(s):])
                x, true_op, size, next_op = cal_state(state, len(self.op), stateT[s][0])
                para_s, score = [[], [], [], [], [], []], 0
                continue
            if episode < 100:
                random_action = np.random.choice(2, 1, p=np.array([0.4, 0.6]))[0]
                if random_action and len(next_op):
                    action_id = int(next_op[0][1])
            else:
                random_action = np.random.choice(2, 1, p=np.array([0.9, 0.1]))[0]
                if random_action and len(next_op):
                    action_id = np.random.randint(0, len(true_op))
            #print state, s0, len(true_op), action_id, next_op
            action = true_op[action_id]
            state.append(action)
            s = stateTrans(state)
            #print "wt",s, "s", self.op, s0
            if (s in stateT and stateT[s][1] <= 0):
                print "a", s, s0, stateT[s0]
                if not s[len(s0):] in stateT[s0][0]:
                    stateT[s0][0].append(s[len(s0):])
                state.pop()
                x, true_op, size, next_op = cal_state(state, len(self.op), stateT[s0][0])
                para_s, score = [[], [], [], [], [], []], 0
                #batch += 1
                continue
            state, reward, done = self.step_(state, action_id, true_op)
            filter_ = []
            if s in stateT:
                filter_ = stateT[s][0]
            x, true_op, size, next_op = cal_state(state, len(state), filter_)
            #print stateT, s0, true_op[action_id], s
            
            if not s in stateT:
                stateT[s] = [[], size]
            if done:
                para_s[0].append(action_choices), para_s[1].append(action_id), para_s[2].append(prob), para_s[3].append((reward+score)), para_s[4].append(aprob)
                if reward + score > mx_score:
                    mx_score = reward + score
                    mx_para = para_s
                cnt += 1
                curve += (cnt, min_op, best) 
                if reward != -1000 and len(self.op) < best[0]:
                    best[0] = len(self.op)
                    best[1] = list(self.op)
                    mx_op = list(self.op)
                    best[2] = cnt
                elif np.random.choice(2, 1, p=np.array([0.1, 0.9]))[0]:
                    break
                #    cnt += 1
                 #   curve += (cnt, min_op)
                s = stateTrans(state)
                stateT[s] = [[], -1]
                if len(state):
                    state.pop()
                s = stateTrans(state)
                stateT[s] = [[], -1]
                if len(state):
                    state.pop()
                s0 = stateTrans(state)
                #print s, s0
                if not s[len(s0)] in stateT[s0][0]:
                    stateT[s0][0].append(s[len(s0):])
                batch += 1
                x, true_op, size, next_op = cal_state(state, len(state), stateT[s0][0])
                para_s, score = [[], [], [], [], [], []], 0
                continue
                #node.end = True
                #break
            else:
                 para_s[0].append(action_choices), para_s[1].append(action_id), para_s[2].append(prob), para_s[3].append((reward-score)), para_s[4].append(aprob)
                 score = reward
            #del para_s
        if len(mx_para) == 0:
            return -1, mx_score
        #print min_op, mx_op
        #exit()
        for idx in range( len(mx_para[0]) ):
            x, action_id, prob, score, aprob = mx_para[0][idx], mx_para[1][idx], mx_para[2][idx], mx_para[3][idx], mx_para[4][idx]
            if len(x) != 20 or len(prob) != 20 or len(aprob) != 20 or action_id > 19:
                 print len(x), len(prob), len(aprob), action_id
                 exit()
            agent.remember(x, action_id, prob, score, aprob)
        return mx_op, mx_score

    def step_(self,state, action_id, true_op):
		global min_op, start_clock, ori_x
		self.op = state
		if len(self.op)+1 >= min_op:
			#self.over = True
			#print self.op, "done"
			return list(self.op), -2000, True
		if action_id >= len(true_op):
			self.op.append([])
			exit()
			return list(self.op), -1500, False
		action, tmp_op = true_op[ action_id ], []
		self.col[ action[2]:action[3] ] = (self.col[ action[2]:action[3] ] - action[1]) / action[0]
		if np.allclose(self.col, ori_x):
			min_op = min(min_op, len(self.op))
			print min_op, len(self.op), self.op, clock() - start_clock
			return list(self.op), (2000.0*ori_op) / len(self.op), True
		cur_op = cal_op( self.col, ori_x )
		for op in cur_op:
			tmp_ = (round(op[0], 3), round(op[1], 3))
			if tmp_ != (1.0, 0.0) and tmp_ not in tmp_op:
				tmp_op.append( tmp_ )
		cur_op = [item for item in cur_op if item[:2] != [1.0, 0.0]]
		dist, dist_1 = len(tmp_op) + 1, len(cur_op)
		return list(self.op), (1000.0*ori_op) / (len(self.op)+dist_1), False
		return list(self.op), (100.0*ori_op) / np.log2(len(self.op)*dist), False

if_log = False
env = Maze()
state = env.reset()
prev_x = None
score = 0
episode = 0
state_shape = [20,4]
action_size = 20
agent = PGAgent(state_shape, action_size)
mx_score = -1000
cnt, batch = 0, 0
curve, curves = [], {}
start_clock = clock()
best = [float('inf'), [], 0]
stateT = dict()
#agent.load("rl_w0")
while True:
    if episode == 1000:
        break
    op, score = env.trace(episode)
    if op == -1:
        env.reset()
        continue
    #exit()
    """
    x, true_op, size = cal_state(state, env.ms)
    action_id, prob, aprob = agent.act(x, len(true_op))
    random_action = np.random.choice(2, 1, p=np.array([0.9, 0.1]))[0]
    if random_action:
        action_id = np.random.randint(0, len(true_op)-1)
    if if_log:
        print true_op, action_id, len(true_op)
    state, reward, done = env.step_(action_id, true_op)
    if done:
#         print(env.action[action], prob[action], reward + score)
        agent.remember(x, action_id, prob, (reward + score), aprob)
	if best[0] >= (env.ms) and reward != -1000:
		best[0] = env.ms
		best[1] = env.op
		best[2] = episode
        score = reward + score
    else:
#         print(env.action[action], prob[action], reward - score)
        agent.remember(x, action_id, prob, (reward - score), aprob)
        score = reward
    if score > mx_score:
        mx_score = score
    """
    if episode < 1000  or cnt == 5:
        episode += 1
	#"""
        if op == -2 or episode % 1500 == 0:
            exit()
            batch += 1
            stateT = {}
            cnt = 0
            ori_op = dataset[ data_choice[batch] ][0] * 2 - 1
            min_op = dataset[ data_choice[batch] ][0] * 2 - 1
            cur_x = np.array(dataset[ data_choice[batch] ][3])
            dst_x = np.array(dataset[ data_choice[batch] ][4])
            if batch == 20:
                agent.save("rl_w2")
            ori_dis = np.linalg.norm(cur_x - dst_x)
            best = [float('inf'), [], 0]
            start_clock = clock()
            if batch > 9:
                curves[data_choice[batch-1]] = curve
            curve = []
            if batch == 20:
                open("sb_t", "w").write(json.dumps(curves))
                exit()
            print('Episode: %d - Score: %f.  %f.  %f.' % (episode, score, mx_score, cnt))
            print best, dataset[ data_choice[batch-1] ][5]
            continue
            #dst_x = np.array(cur_x)
            #rg = sorted(np.random.randint(0, 45, 10))
            #dst_x[rg[0]:rg[4]] += -np.random.randint(0, 100)
            #dst_x[rg[1]:rg[3]] += np.random.randint(0, 100)
            #dst_x[rg[2]:rg[7]] += -np.random.randint(0, 100)
            #dst_x[rg[5]:rg[8]] += np.random.randint(0, 100)
            #dst_x[rg[6]:rg[9]] += -np.random.randint(0, 100)
	#"""
#        print "sp"
#	exit()
        agent.train()
        if episode % 50 == 0:
            print('Episode: %d - Score: %f.  %f.  %f.' % (episode, score, mx_score, cnt))
            print best, dataset[0][3], min_op
        score = 0
        state = env.reset()
        mx_score = -1000
        #cnt = 0
    else:
        cnt += 1
# agent.save("rl_w2")
