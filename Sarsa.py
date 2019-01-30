import numpy as np
import random
import matplotlib.pyplot as plt

POINTS = {'R': [0, 0], 'Y': [0, 4], 'G': [4, 0], 'B': [3, 4]}
IN_TAXI = False 
GOAL = None 
POSITION = None
PASSENGER = None
NUMBER_OF_MOVES = 0
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK_UP', 'DROP_OFF']
ITERATION_COUNT = 100000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.6
EPSILON = 0.1

def setRandomGoal():
	global GOAL
	GOAL = random.choice(['R', 'G', 'Y', 'B'])

def setRandomPassenger():
	global PASSENGER
	global POSITION
	PASSENGER = random.choice(['R', 'G', 'Y', 'B', 'I'])
	if PASSENGER=='i':
		IN_TAXI = True

def setRandomPosition():
	global POSITION
	POSITION = [random.choice([0, 1, 2, 3, 4]), random.choice([0, 1, 2, 3, 4])]

def hitTheWall(action):
	global POSITION
	if POSITION[0]==1 and POSITION[1]==0 and action=='RIGHT':
		return -1
	if POSITION[0]==1 and POSITION[1]==1 and action=='RIGHT':
		return -1
	if POSITION[0]==2 and POSITION[1]==0 and action=='LEFT':
		return -1
	if POSITION[0]==2 and POSITION[1]==1 and action=='LEFT':
		return -1
	if POSITION[0]==0 and POSITION[1]==3 and action=='RIGHT':
		return -1
	if POSITION[0]==0 and POSITION[1]==4 and action=='RIGHT':
		return -1
	if POSITION[0]==1 and POSITION[1]==3 and action=='LEFT':
		return -1
	if POSITION[0]==1 and POSITION[1]==4 and action=='LEFT':
		return -1
	if POSITION[0]==2 and POSITION[1]==3 and action=='RIGHT':
		return -1
	if POSITION[0]==2 and POSITION[1]==4 and action=='RIGHT':
		return -1
	if POSITION[0]==3 and POSITION[1]==3 and action=='LEFT':
		return -1
	if POSITION[0]==3 and POSITION[1]==4 and action=='LEFT':
		return -1
	if POSITION[0]==0 and action=='LEFT':
		return -1
	if POSITION[0]==4 and action=='RIGHT':
		return -1
	if POSITION[1]==0 and action=='UP':
		return -1
	if POSITION[1]==4 and action=='DOWN':
		return -1
	return 0

def getRewardValue(action):
	global GOAL
	global PASSENGER
	global POSITION
	global IN_TAXI
	global NUMBER_OF_MOVES
	if not(PASSENGER == 'I'):
		if POSITION[0]==POINTS[PASSENGER][0] and POSITION[1]==POINTS[PASSENGER][1] and action=='PICK_UP':
			return 1
		if not (POSITION[0]==POINTS[PASSENGER][0] and POSITION[1]==POINTS[PASSENGER][1]) and action=='PICK_UP':
			return -1
	elif (PASSENGER == 'I'):
		if action=='PICK_UP':
			return -1
	if POSITION[0]==GOAL[0] and POSITION[1]==GOAL[1] and action=='DROP_OFF' and IN_TAXI==True:
		return 10/NUMBER_OF_MOVES
	if action=='DROP_OFF' and not(POSITION[0]==GOAL[0] and POSITION[1]==GOAL[1]) and IN_TAXI==True:
		return -1 
	if action=='DROP_OFF' and IN_TAXI==False:
		return -1
	else:
		return hitTheWall(action)


def createQTable():
	Q_table = {}
	for i1 in range(0, 5):
		for i2 in range(0, 5):
			for passenger in ['R', 'G', 'Y', 'B', 'I']:
				for goal in ['R', 'G', 'Y', 'B']:
					Q_table[(i1, i2, passenger, goal)] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0, 'PICK_UP':0, 'DROP_OFF':0}

	return Q_table

def pick_action_based_on_policy(Q_table, curr_state):
	global ACTIONS
	if random.uniform(0, 1) > EPSILON:
		max_q = Q_table[curr_state][ACTIONS[0]]
		max_q_action = [ACTIONS[0]]
		for a in ACTIONS:
			if Q_table[curr_state][a] > max_q:
				max_q = Q_table[curr_state][a] 
				max_q_action = [a]
			elif Q_table[curr_state][a] == max_q:
				max_q_action.append(a)
		return random.choice(max_q_action)
	else:
		return random.choice(ACTIONS)

def createNextState(action):
	global POSITION
	global GOAL
	global PASSENGER
	global IN_TAXI
	if not(PASSENGER=='I'):
		if POSITION[0]==POINTS[PASSENGER][0] and POSITION[1]==POINTS[PASSENGER][1] and action=='PICK_UP':
			IN_TAXI = True
			PASSENGER = 'I'
	if hitTheWall(action)==0:
		if action=='UP':
			POSITION[1] -= 1
		elif action=='DOWN':
			POSITION[1] += 1
		if action=='RIGHT':
			POSITION[0] += 1
		elif action=='LEFT':
			POSITION[0] -= 1
	return (POSITION[0], POSITION[1], PASSENGER, GOAL)

def pick_best_action_based_on_policy(Q_table, curr_state):
	global ACTIONS
	max_q = Q_table[curr_state][ACTIONS[0]]
	max_q_action = [ACTIONS[0]]
	for a in ACTIONS:
		if Q_table[curr_state][a] > max_q:
			max_q = Q_table[curr_state][a] 
			max_q_action = [a]
		elif Q_table[curr_state][a] == max_q:
			max_q_action.append(a)
	return random.choice(max_q_action)


def saveTheQTable(Q_table):
	f= open("q_table.txt","w")
	f.write(str(Q_table))
	f.close()

def drawPlot(x, y, plot_title):
	plt.plot(x, y)
	plt.xlabel('epoch number')
	plt.ylabel('reward value')

	plt.title(plot_title)
	plt.legend()
	plt.show()

def trainTheModel():
	global POSITION
	global GOAL
	global PASSENGER
	global IN_TAXI
	global NUMBER_OF_MOVES
	Q_table = createQTable()
	plot_x, plot_y = [], []
	for n in range(ITERATION_COUNT):
		setRandomPosition()
		setRandomPassenger()
		setRandomGoal()
		NUMBER_OF_MOVES = 0
		IN_TAXI = False
		curr_state = (POSITION[0], POSITION[1], PASSENGER, GOAL)
		total_reward = 0
		action = pick_action_based_on_policy(Q_table, curr_state)
		while not(reachedGoal()):
			reward_for_this_action = getRewardValue(action)
			total_reward += reward_for_this_action
			next_state = createNextState(action)
			next_action = pick_action_based_on_policy(Q_table, next_state)
			next_action_q = Q_table[next_state][next_action]
			NUMBER_OF_MOVES += 1
			Q_table[curr_state][action] = (1-LEARNING_RATE) * Q_table[curr_state][action] + LEARNING_RATE*(reward_for_this_action + DISCOUNT_FACTOR * next_action_q)
			curr_state = next_state
			action = next_action
		plot_x.append(n)
		plot_y.append(total_reward)
	drawPlot(plot_x, plot_y, 'Sarsa')
	return Q_table

def reachedGoal():
	global GOAL
	global PASSENGER
	if not PASSENGER=='I':
		return POINTS[GOAL][0]==POINTS[PASSENGER][0] and POINTS[GOAL][1]==POINTS[PASSENGER][1]
	else:
		return POINTS[GOAL][0]==POSITION[0] and POINTS[GOAL][1]==POSITION[1]

def testTheModel(Q_table):
	global POSITION
	global GOAL
	global PASSENGER
	global IN_TAXI
	global NUMBER_OF_MOVES
	NUMBER_OF_MOVES = 0
	PASSENGER = 'Y'
	GOAL = 'B'
	POSITION = [0, 1]
	IN_TAXI = False
	curr_state = (POSITION[0], POSITION[1], PASSENGER, GOAL)
	while not(reachedGoal()):
		action = pick_best_action_based_on_policy(Q_table, curr_state)
		print('['+str(NUMBER_OF_MOVES)+']:'+str(curr_state)+':[ACTION]:'+str(action))
		reward_for_this_action = getRewardValue(action)
		next_state = createNextState(action)
		NUMBER_OF_MOVES += 1
		curr_state = next_state
	print('REACHED GOAL!')

def main():
	Q_table = trainTheModel()
	testTheModel(Q_table)

	

