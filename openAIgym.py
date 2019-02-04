import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep

alpha = 0.1
gamma = 0.6
epsilon = 0.1
all_epochs = []
all_penalties = []


def print_frames(frames):
	for i, frame in enumerate(frames):
		clear_output(wait=True)
		print(frame['frame'].getvalue())
		print(f"Timestep: {i + 1}")
		print(f"State: {frame['state']}")
		print(f"Action: {frame['action']}")
		print(f"Reward: {frame['reward']}")
		sleep(.1)

def TrainTheModel(env):
	alpha = 0.1
	gamma = 0.6
	epsilon = 0.1

	q_table = np.zeros([env.observation_space.n, env.action_space.n])
	all_epochs = []
	all_penalties = []

	for i in range(1, 100001):
		state = env.reset()

		epochs, penalties, reward, = 0, 0, 0
		done = False
		
		while not done:
			if random.uniform(0, 1) < epsilon:
				action = env.action_space.sample()
			else:
				action = np.argmax(q_table[state])

			next_state, reward, done, info = env.step(action) 
			
			old_value = q_table[state, action]
			next_max = np.max(q_table[next_state])
			
			new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
			q_table[state, action] = new_value

			if reward == -10:
				penalties += 1

			state = next_state
			epochs += 1
			
		if i % 100 == 0:
			clear_output(wait=True)
			print(f"Episode: {i}")

	print("Training finished.\n")
	return q_table

def testTheModel(env, q_table):
	state = env.reset()
	penalties, reward, total_reward = 0, 0, 0
	frames = []
	done = False
	
	while not done:
		action = np.argmax(q_table[state])
		state, reward, done, info = env.step(action)

		if reward == -10:
			penalties += 1

		total_reward += reward

		frames.append({
			'frame': env.render(mode='ansi'),
			'state': state,
			'action': action,
			'reward': reward
			}
		)


	print(f"timesteps: {penalties}")
	print(f"total reward: {total_reward}")
	print(f"penalties: {penalties}")
	print_frames(frames)

def main():
	env = gym.make("Taxi-v2").env
	q_table = TrainTheModel(env)
	testTheModel(env, q_table)
