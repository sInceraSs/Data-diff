# coding: utf-8

# import gym
import numpy as np
from keras.models import Model
from keras.layers import Dense, Reshape, Flatten, Input, Activation, Dropout
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from time import clock
import json

dataset, super_ = [], []
with open("dataset", 'r') as infile:
	for line in infile:
		dataset.append(json.loads(line))

train_choice = [0, 1, 6, 10, 11, 12, 14, 16, 20, 23, 25, 28, 31, 32, 36, 37, 40, 46, 56, 58] 
test_choice = [3, 4, 15, 19, 21, 33, 41, 43, 53, 55]
test_idx = 50
with open("sb_sd_", 'r') as infile:
    for i, line in enumerate(infile):
        if i:
            super_ = json.loads(line)
train_choice += test_choice
print train_choice
data_choice = test_choice
cur_x = np.array(dataset[test_idx][3])
dst_x = np.array(dataset[test_idx][4])
print dataset[test_idx][5]
print dst_x
ori_dis = np.linalg.norm(cur_x - dst_x)
ori_op = dataset[test_idx][0] * 2 - 1
min_op = dataset[test_idx][0] * 2 - 1
def greedy_search(x):
    global dst_x
    res_set, cur_tuples = list(), list(x)
    while len(cur_tuples):
        cur_set = dict()
        max_flag, max_coef = list(), list()
        for ot_idx, o_tuple in enumerate(cur_tuples):
            max_o_coef = list()
            ot_coef = { (0, dst_x[ot_idx]):[ o_tuple ] }
            for pt_idx, pk_tuple in enumerate(cur_tuples):
                if pk_tuple != o_tuple:
                    coef = 1
                    #coef = float( dst_x[ot_idx] - dst_x[pt_idx]) / (o_tuple - pk_tuple)
                    intercept = float(dst_x[ot_idx]) - o_tuple*coef
                    if (coef, intercept) not in ot_coef:
                        ot_coef[ (coef, intercept) ] = [ o_tuple ]
                    ot_coef[ (coef, intercept) ].append( pk_tuple )
            for (coef, intercept) in ot_coef:
                if len( ot_coef[ (coef, intercept) ] ) > len( max_o_coef ):
                    max_o_coef = ot_coef[ (coef, intercept) ]
            if max_o_coef not in max_flag:
                max_flag.append(  [coef, intercept, max_o_coef] )
            del ot_coef
        for pk_set in max_flag:
            if len(pk_set) > len(max_coef):
                max_coef = pk_set
        for pk_tuple in max_coef[2]:
            try:
				cur_tuples.remove(pk_tuple)
            except:
                print cur_tuples, pk_tuple, x
                exit()
        res_set.append( [ max_coef[0], max_coef[1], min(max_coef[2]), max(max_coef[2]), len(max_coef[2]) ] )
    return len(res_set)

def cal_op(s):
    global dst_x
    pre_coef, pre_add = float('nan'), float('nan')
    cur_op = []
    #cur_op = [ [1, pre_add, 0, 1]  ]
    #cur_op = [ [pre_coef, dst_x[0], 0, 1] ]
    for i in range( 0, len(s) ):
        o_tuple, d_tuple = s[i], dst_x[i]
	"""
	try:
        	coef = float(d_tuple - dst_x[i-1]) / (o_tuple - s[i-1])
	except:
		#pre_coef = float('nan')
		print s
		exit()
        intercept = float(dst_x[i]) - o_tuple*coef
        if coef == pre_coef or pre_coef == 0:
            cur_op[-1][:2] = coef, intercept
            cur_op[-1][3] = i+1
            pre_coef = coef
        else:
            cur_op.append( [ 0, d_tuple, i, i+1] )
            pre_coef = 0
	"""
    	cur_add = dst_x[i] - s[i]
        # if cur_add == 0:
        #     continue
        if cur_add == pre_add:
            cur_op[-1][3] = i+1
        else:
            cur_op.append( [1, cur_add, i, i+1] )
            pre_add = cur_add
    return cur_op

def cal_dis(s, cur_op):
    global dst_x, ori_dis, ori_op
    tmp = []
    num_op, num_diff, dis = 0, 0, 0
    for pk_op_c, pk_op_i, start, end in cur_op:
        if pk_op_c == 1 and pk_op_i == 0 and (pk_op_c, pk_op_i) in tmp:
            continue
        num_op += 1
        tmp.append( (pk_op_c, pk_op_i) )
    dis = np.linalg.norm(s - dst_x)
    for i in range( len(s) ):
        if abs(s[i] - dst_x[i]) > 1e-6:
            num_diff += 1
    greedy_dis = greedy_search(s)
    del tmp
    return  num_op*1.0/ori_op,  num_diff*1.0/ len(dst_x), np.abs(dis-ori_dis)*1.0/ (ori_dis), greedy_dis

def cal_state(s_op, depth, filter_, super_=False):
    global ori_op, cur_x
    ss, candidate_op, true_op = [], [], []
    tmp_r, tmp_f = [], []
    if (not super_) and len(filter_) > 0:
          #print filter_
          for op in filter_:
              if len(op) == 0:
                  continue
              op = stateTrans(op)[0]
              tmp_f.append((op[1], op[2], op[3]))
    s_ = np.array(cur_x)
    if super_:
        for i, (pk_op_i, start_id, end_id) in enumerate(s_op):
            s_[start_id:end_id] = s_[start_id:end_id] + pk_op_i
    #        tmp_r.append( (start_id, end_id)  )
    else:
        for i, (pk_op_c, pk_op_i,start_id, end_id) in enumerate(s_op):
            s_[start_id:end_id] = s_[start_id:end_id] + pk_op_i
	#print s_op[i]
#	s_op[i][1] *= -1
            tmp_r.append( (start_id, end_id)  )
    cur_op = cal_op(s_)
    #expand state
    pk_op_c_idx = -1
    # print filter_, cur_op
    for pk_op_idx in range( len(cur_op) ):
        pk_op_c, pk_op_i = cur_op[ pk_op_idx ][:2]
        if pk_op_i == 0:
            # print cur_op
            continue
            exit()
        for start in range( pk_op_idx + 1 ):
            for end in range( pk_op_idx, len(cur_op) ):
                start_id, end_id = cur_op[ start ][2], cur_op[ end ][3]
                if ( start_id, end_id ) in tmp_r or (pk_op_i, start_id, end_id) in tmp_f:
                    #print "cao", s_op,"sb", cur_op
                    continue
                tmp_x = np.array(list( s_ ))
                tmp_x[start: end] = tmp_x[start: end]*pk_op_c + pk_op_i
                tmp_op = cal_op( tmp_x )
                candidate_op.append( [ pk_op_c, pk_op_i, start_id, end_id, tmp_op ] )
                if super_ and pk_op_i == filter_[0] and start_id == filter_[1] and end_id == filter_[2]:
                    pk_op_c_idx = len(candidate_op) - 1
    if super_:
        choice = [(len(candidate_op[i][4]), i) for i in range( len(candidate_op) ) if i != pk_op_idx]
        choice.insert(0, (len(candidate_op[pk_op_c_idx][4]), pk_op_c_idx))
    else:
        choice = [(len(candidate_op[i][4]), i) for i in range( len(candidate_op) )]
    choice = np.array(choice)
    np.random.shuffle(choice[:20])
    #choice.sort()
    if len(choice) != len(candidate_op):
        print "wocao", len(choice)
        print len(candidate_op)
        exit()
    for i in range( min(20, len(candidate_op)) ):
        pk_op_c, pk_op_i, start, end, tmp_op = candidate_op[ choice[i][1] ]
        tmp_x = np.array(list( s_ ))
        tmp_x[start: end] = tmp_x[start: end] + pk_op_i
        num_op, num_diff, dis, gd = cal_dis( tmp_x, tmp_op )
        ss += [np.array([[num_op, num_diff, dis, (depth+1)/(ori_op*1.0), 1.0*(gd+depth+1)/ori_op, (depth+1)*1.0/min_op]])]
        true_op.append( candidate_op[ choice[i][1] ][:-1] )
    for i in range( min(20, len(candidate_op)), 20 ):
        ss += [np.array([[0, 0, 0, 0, 0, 0]])]
    return ss, true_op, len(candidate_op)

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
			s = s + str(pk_op_i) + "," + str(start) + "," + str(end) + " "
		return s
	else:
		s = []
		ops = state.split(' ')
		for op in ops:
			if len(op) == 0:
				continue
			op_i = op.split(",")
			s.append([1, int(op_i[0]), int(op_i[1]),int(op_i[2])])
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
        #net1 = Dense(1, name='rank_layer', init='he_uniform')
        net1 = Dense(8, name='rank_layer1', init='he_uniform')
        net2 = Dense(4, name='rank_layer2', init='he_uniform')
        net3 = Dense(1, name='rank_layer3', init='he_uniform')
        outs = []
        inps = []
        for i in range(self.state_shape[0]):
            inp = Input(shape=[self.state_shape[1]])
            inps += [inp]
            #rank = net1(inp)
            #rank = net3(net2(net1(inp)))
            rank = net2(net1(inp))
            rank = net3(Dropout(0.3)(rank))
            outs += [rank]
        out = Concatenate()(outs)
        prob = Activation(activation='softmax')(out)
        #prob = Dense(self.action_size, activation='softmax', init='he_uniform')(out)
        opt = Adam(lr = 0.005)
        model = Model(inputs=inps, outputs=prob)
        model.compile(loss='categorical_crossentropy', optimizer=opt,  metrics=['accuracy'])
        #model.compile(loss='categorical_crossentropy', optimizer=opt)
        print(model.summary())
        return model

    def remember(self, state, action, prob, reward, aprob, label=[]):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.probs.append(aprob)
        self.labels.append(label) 
        self.gradients.append(np.array(y).astype('float32') - prob)
        for i,j in zip(self.states, state):
            i.append(j[0])
        self.rewards.append(reward)

    def act(self, state, lenth, true_op=[]):
		global super_
		_aprob, label = 0.0, []
		aprob = self.model.predict(state, batch_size=1).flatten()
		for i,item in enumerate(state):
			if np.sum(item[0]) != 0:
				_aprob += aprob[i]
			else:
				aprob[i] = 0
			if i < len(true_op) and true_op[i] in super_:
				label[i] = 1
		if _aprob != 0.0:
			aprob[:] /= _aprob
		#self.probs.append(aprob)
		if np.sum(aprob) == 0:
			prob = np.array([0.1 for i in range(20)])
			return -1, prob, aprob, label
		else:
			prob = aprob / np.sum(aprob)
		action = np.random.choice(self.action_size, 1, p=prob)[0]
		return action, prob, aprob, label

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

    def train_super(self, feature_vs, label_s, length):
        v = self.model.fit(feature_vs, label_s, epochs = 50, batch_size = length)
        return v.__dict__         

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class Maze(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.col = np.array(cur_x)
        self.op = []
        self.over = False
        self.ms = 0
        self.step = 0
        return self.op

    def traceT(self, state):
		if len(state) > 0:
			pk_op_c, pk_op_i, start_id, end_id = state[-1]
			#self.col[start_id:end_id] -= pk_op_i
			state.pop()
		self.op = list(state)
		for pk_op_c, pk_op_i, start_id, end_id in state:
			self.col[start_id:end_id] += pk_op_i
		return list(state)
		#global stateT
		#pk_op_c, pk_op_i, start_id, end_id = state[-1]
		#state.pop()
		#s = stateTrans(state)
		#if s in stateT:
		#x, true_op, size = stateT[s][:3]
		#stateT[s][3] -= 1
		#else:
		#	x, true_op, size = cal_state(state, env.ms)
		#	stateT[s] = [x, true_op, size, size]
		#self.col[start_id:end_id] -= pk_op_i
		#sele.ms -= 1
		#return state, x, true_op, size

    def trace(self, supervise=False):
        global trie, stateT, cnt, curve, best, min_op, agent
        mx_op, mx_para = [], []
        mx_score = -1000 
        batch = 0
        para_s, score = [[], [], [], [], [], []], 0
        #node, pre_node = trie.root, trie.root
        state = list(self.reset())
        filter_ = []
        if "" in stateT:
            filter_ = stateT[""][0]
            if (stateT[""][1] <= 0) or len(stateT[""][0]) >= stateT[""][1]:
                return -2, -1000
        x, true_op, size = cal_state(state, len(self.op), filter_)
        if "" not in stateT:
            stateT[ "" ] = [ [], size ]
        #print stateT
        while True:
            #if batch % 5 == 0:
            #    print batch
            if (supervise and batch > 0) or batch >= 10:
                break
                exit()
            s0 = stateTrans(state)
            if (s0 in stateT and stateT[s0][1] <= 0) or len(stateT[s0][0]) >= stateT[s0][1]: 
                #if len(state):
                 #   state.pop()
                #self.op = list(state)
                if s0 == "":
                    break
                #print stateT[s0], "s0 xie"
                state = self.traceT(state)
                s = stateTrans(state)
                if s0[len(s):] not in stateT[s][0]:
                    stateT[s][0].append( s0[len(s):] )
                self.ms -= 1
                x, true_op, size = cal_state(state, len(self.op), stateT[s][0])
                para_s, score = [[], [], [], [], [], []], 0
                batch += 1
                continue
            action_id, prob, aprob, label = agent.act(x, len(true_op), true_op)
            #print true_op,  len(true_op), action_id, batch, state, len(state), s0
            if action_id == -1 or len(true_op) == 0:
                #batch += 1
                stateT[s0][1] = -1
                #print "no next state", s0
                #print stateT
                #if len(state):
                #    state.pop()
               # self.op = list(state)
                state = self.traceT(state)
                s = stateTrans(state)
                action = ""
                for a_id in range(len(s), len(s0)):
                    action = action + s0[a_id]
                if not action in stateT[s][0]:
                    stateT[s][0].append(action)               
                self.ms -= 1
                x, true_op, size = cal_state(state, len(self.op), stateT[s][0])
                para_s, score = [[], [], [], [], [], []], 0
                continue
            random_action = np.random.choice(2, 1, p=np.array([0.9, 0.1]))[0]
            if random_action:
                action_id = np.random.randint(0, len(true_op))
            #state.append(true_op[action_id])
            #print state, s0
            action = true_op[action_id]
            state.append(action)
            s = stateTrans(state)
            #print "wt",s, "s", self.op, s0
            if (s in stateT and stateT[s][1] <= 0):
                #print "a", s, s0, stateT[s0]
                action = ""
                for a_id in range(len(s0), len(s)):
                    action = action + s[a_id]
                if not action in stateT[s0][0]:
                    stateT[s0][0].append(action)
                state = self.traceT(state)
                self.ms -= 1
                #print state
                x, true_op, size = cal_state(state, len(self.op), stateT[s0][0])
                para_s, score = [[], [], [], [], [], []], 0
                #batch += 1
                continue
            state, reward, done = self.step_(action_id, true_op)
            filter_ = []
            if s in stateT:
                filter_ = stateT[s][0]
            x, true_op, size = cal_state(state, len(self.op), filter_)
            #print stateT, s0, true_op[action_id], s
            if not s in stateT:
                stateT[s] = [[], size]
            #pos = trie.find(node, op,)
            #if pos < 0:
             #   node.childs.append(Node(op))
              #  pos = len(node.childs) - 1 
            #elif len(node.childs) == node.size or node.end:
             #   print node.childs, self.op
                #break
            #x, true_op, size = cal_state(state, env.ms)
            #node.size = size
            #node = node.childs[pos]
            if done:
                para_s[0].append(x), para_s[1].append(action_id), para_s[2].append(prob), para_s[3].append((reward+score)), para_s[4].append(aprob), para_s[5].append(label)
                if reward + score > mx_score or supervise:
                    mx_score = reward + score
                    mx_para = para_s
                cnt += 1
                curve += (cnt, min_op, best) 
                if reward != -1000 and len(self.op) < best[0]:
                    best[0] = len(self.op)
                    best[1] = list(self.op)
                    mx_op = list(self.op)
                    best[2] = cnt
                #    cnt += 1
                 #   curve += (cnt, min_op)
                s = stateTrans(state)
                stateT[s] = [[], -1]
                state = self.traceT(state)
                s = stateTrans(state)
                stateT[s] = [[], -1]
                state = self.traceT(state)
                s0 = stateTrans(state)
                action = ""
                #print s, s0
                for a_id in range(len(s0), len(s)):
                    action = action + s[a_id]
                if not action in stateT[s0][0]:
                    stateT[s0][0].append(action)
                batch += 1
                self.ms -= 1
                x, true_op, size = cal_state(state, len(self.op), stateT[s0][0])
                para_s, score = [[], [], [], [], [], []], 0
                continue
                #node.end = True
                #break
            else:
                 para_s[0].append(x), para_s[1].append(action_id), para_s[2].append(prob), para_s[3].append((reward-score)), para_s[4].append(aprob), para_s[5].append(label)
                 score = reward
            #del para_s
        if len(mx_para) == 0:
            return -1, mx_score
        print min_op, mx_op
        #exit()
        for idx in range( len(mx_para[0]) ):
            x, action_id, prob, score, aprob, label = mx_para[0][idx], mx_para[1][idx], mx_para[2][idx], mx_para[3][idx], mx_para[4][idx], mx_para[4][idx]
            agent.remember(x, action_id, prob, score, aprob, label)
        return mx_op, mx_score

    def step_(self, action_id, true_op):
		global min_op, start_clock, dst_x
		self.ms += 1
		if len(self.op)+1 >= min_op:
			#self.over = True
			#print self.op, "done"
			return list(self.op), -1000, True
		if action_id >= len(true_op):
			self.op.append([])
			exit()
			return list(self.op), -1500, False
		action = true_op[ action_id ]
		#if self.over:
		#	print('over')
		#	return self.op, -10, True
		self.op.append(action)
		self.col[ action[2]:action[3] ] = self.col[ action[2]:action[3] ] * action[0] + action[1]
		tmp_op = cal_op( self.col )
		dis = cal_dis( self.col, tmp_op )
		if len(tmp_op) == 1:
			#self.over = True
			#if batch > 20:
			if tmp_op[0][0] != 1 or tmp_op[0][1] != 0:
				self.op.append(tmp_op[0])
				self.ms += 1
				min_op = min(min_op, len(self.op))
			print self.ms, self.op, clock() - start_clock, min_op
			if len(self.op) == min_op:
				return list(self.op), -1000, True
			return list(self.op), (1.0*ori_op)/(len(self.op))*10000, True
	#print self.ms + dis[3]
		return list(self.op), (1.0*ori_op)/(len(self.op) + dis[3])*1000, False
def generate_super_train():
    global super_, cur_x, stateT, agent, cnt, curve, best, state_shape, action_size
    curve = []
    best_path = super_[0][2]
    ori_x = np.array(cur_x)
    feature_vs, label_s = [[] for i in range(20)], []
    test_vs, test_ls = list(feature_vs), []
    #"""
    with open("super_data_v1", 'r') as infile:
        for i,line in enumerate(infile):
            if not i:
                feature_vs = json.loads(line)
            else:
                label_s = json.loads(line)
    #print feature_vs[0], label_s[0]
    """
    for i in range(len(feature_vs)):
        for j in range(len(feature_vs[i])):
            #if label_s[j][i] > 0:
             #   feature_vs[i][j] = np.insert(feature_vs[i][j], 0, 1)
            feature_vs[i][j] = np.insert(feature_vs[i][j], 0, label_s[j][i])
            feature_vs[i][j] = np.array(feature_vs[i][j])
            #if feature_vs[i][j][0] != label_s[j][i]:
             #   print "cao"
              #  exit()
            for k in range(len(feature_vs[i][j])):
                if feature_vs[i][j][k] > 1:
                    feature_vs[i][j][k] /= 2
                if np.abs(feature_vs[i][j][k]) > 1:
                    print "yaosi"
                    exit()
    #"""
    """
    for qi, (depth, batch, path) in enumerate(super_):
        print qi, len(super_), best_path, path
        if qi == 51:
            break
        cur_x = np.array(ori_x)
        for j in range(100):
            feature_v,  state = [], []
            cur_x = np.array(ori_x)
            for idx, op in  enumerate(path):
                x, true_op, size = cal_state(state, idx+1, op, True)
                state.append(op)
                label =  [0 for k in range(20)]
                for i, test_op in enumerate(true_op):
                    if test_op[1:] in best_path:
                        label[i] = 1.0
                if np.sum(label) > 1:
                    sum_ = np.sum(label)*1.0
                    for v, item in enumerate(x):
                        tmp = item[0]
                        tmp = np.insert(tmp, 0, label[v])
                        feature_vs[v].append(tmp)
                    label = list(np.array(label) / sum_)
                    label_s.append(list(label))
    print label_s[0]
    f = open("super_data_v2", "w")
    tmp = list(feature_vs)
    for i in range(len(tmp)):
        print len(tmp[i])
        for j in range(len(tmp[i])):
            tmp[i][j] = list(tmp[i][j])
    f.write(json.dumps(list(tmp)))
    f.write("\n")
    f.write(json.dumps(label_s))
    f.close()
    #"""
    length = int(0.9*len(feature_vs[0]))
    print length
    #exit()
    for i in range(len(feature_vs)):
        print len(feature_vs[i])
        test_vs[i] = np.array(feature_vs[i][length:])
        feature_vs[i] = np.array(feature_vs[i][:length])
    test_ls = list(label_s[length:])
    label_s = list(label_s[:length])
    #print feature_vs
    #exit()
    state_shape = [20,6]
    action_size = 20
    cnt = 0
    agent = PGAgent(state_shape, action_size)
    v = agent.model.fit(feature_vs, label_s, epochs = 50, batch_size = 32, validation_data=(test_vs, test_ls))
    print v.history['val_loss']
    print v.history['loss']
    #agent.save("rl_s0")
    #exit()
    """
    pre_s = agent.model.predict(test_vs, batch_size=32)
    #learning_c = [v['val_acc'] for v in agent.model.history]
    #print learning_c
    pre_s = agent.model.predict(test_vs, test_ls, batch_size=32)
    for i in range(len(pre_s)):
        print pre_s[i]
        action = np.random.choice(20, 1, p=pre_s[i])[0]
        if test_ls[i][action] == 1:
            cnt += 1
    print cnt * 1.0 / len(pre_s)
    """ 
    #exit()
    # test
    env = Maze()
    state = env.reset()
    start_clock = clock()
    best = [float('inf'), [], 0]
    stateT = dict()
    for i in range(50):
        print best_path
        op, score = env.trace()
        agent.train()
    exit()

generate_super_train()  
trie = Trie()
if_log = False
env = Maze()
state = env.reset()
prev_x = None
score = 0
episode = 0
state_shape = [20,6]
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
    if episode == 10000:
        break
    if episode < 500:
        op, score = env.trace(True)
    else:
        op, score = env.trace()
    if op == -1:
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
    if episode < 500  or cnt == 5:
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
#	print "sp"
#	exit()
        if episode < 500:
            print episode, clock() - start_clock
            agent.train_super()
        else:
            agent.train()
        if episode % 50 == 0:
            print('Episode: %d - Score: %f.  %f.  %f.' % (episode, score, mx_score, cnt))
            print best, dataset[ data_choice[batch] ][5], min_op
        score = 0
        state = env.reset()
        mx_score = -1000
        #cnt = 0
    else:
        cnt += 1
agent.save("rl_w2")
