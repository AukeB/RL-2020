# Reincorment Learning 2020 - Assignment 3: Function Approximation
# Auke Bruinsma (s1594443), Meng Yao (s2308266), Ella (s2384949).
# This file contains part 1 of the assignment: Mountain Car - 3 points

# Imports
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import os,sys

import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Initialize the Mountain Car environment.
env = gym.make('MountainCar-v0') # Mountain car environment.
env.reset() # Reset the environment so that it starts in its initial state.

# Setting up hyperparameters.
num_games = 5000 # Number of games that will be played.
min_score = -198 # Minimum score required for the data to be allowed to be training data.
num_steps = 200 # Steps in a single simulation.

max_actions = 3 # In this specific environment there are 3 possible actions.
threshold = -0.2 # Position threshold that will allow the reward to be equal to 1 instead of -1.
train_iterations = 50 # Number of training iterations.
play_games = 5 # Number of times network will be playing the game after training.

def random_actions(printformat=1,visual=False):
	for i in range(num_steps):
		if visual == True:
			env.render() # For visualization.
		random_action = rd.randrange(0,max_actions)
		observation, reward, done, info = env.step(random_action)

		if printformat == 0:
			print(f'Step {i}')
			print(f' Action:   {random_action}')
			print(f' Position: {observation[0]}')
			print(f' Velocity: {observation[1]}')
			print(f' Reward:   {reward}')
			print(f' Done:     {done}')
			print(f' Info:     {info}')
		if printformat == 1:
			if i == 0:
				print('Step  Action  Pos    Vel   Reward  Done    Info')
			print(f'{i}     {random_action}       {observation[0]:.2f}  {observation[1]:.2f}  {reward}    {done}  {info}')

		# Statement to finish the simulation when the car has reached the top.
		# Will never occur with random motions.
		if done:
			break

	# Close and reset the environment.
	if visual == True:
		env.close()
		env.reset()

def convert2training_data(data,train_data):
	for i in range(len(data)):
		if data[i][1] == 0: train_data.append((data[i][0],[1,0,0]))
		if data[i][1] == 1: train_data.append((data[i][0],[0,1,0]))
		if data[i][1] == 2: train_data.append((data[i][0],[0,0,1]))

def make_train_sets(threshold=-0.2):
	# Counter that keeps track of games that will be used for training.
	succesful_games_counter = 0
	train_data = []

	# Output some statistics.
	print('\nSuccesful score(s):')

	# Play a certain amount of games to generate training data.
	for i in range(num_games):
		sum_reward = 0 # Variable that hold the sum of all the rewards.
		game_states = [] # Will hold information for all possible states of the game.
		
		# Iteratate through each game.
		for j in range(num_steps):
			random_action = rd.randrange(0,max_actions) # Make a random move.
			observation,reward,done,info=env.step(random_action) # Obtain state values.

			game_states.append([observation,random_action]) # Add relevant variables to game_states variable.
				
			# This is the important part: Set result equal to 1 if the position ...
			# ... of the car is above a certain threshold.
			if observation[0] > threshold:
				reward = 1

			# Find the sum of all the rewards through each game.
			sum_reward += reward

			# If the car has reached the top, stop the game (will probably ...
			# ... never happen with random moves.
			if done:
				break

		# If this occurs, the game is fit for training data.
		if sum_reward >= min_score:
			sys.stdout.write(f'{sum_reward} ')
			succesful_games_counter += 1
			convert2training_data(game_states,train_data) # Construct training data.

		env.reset() # Reset environment so all state variables will have their initial value again. IMPORTANT.

	# Print some statistics.
	print(f'\n\nTotal games simulated:  {num_games}')
	print(f'Total succesful games:  {succesful_games_counter}')
	fraction = float(succesful_games_counter)/num_games*100
	print(f'Percentage succesful games: {fraction:.2f} %')

	return train_data

def loss_plotter(data):
	plt.figure(figsize=(12,8))
	plt.plot(data.history['loss'])
	plt.title('Model accuracy',fontsize=18)
	plt.xlabel('Epoch',fontsize=15)
	plt.ylabel('Loss',fontsize=15)
	plt.savefig('loss.png')
	plt.show()	
	plt.close()

def data_trainer(train_data):
	# Create two lists so that we can store data in there in the correct format for the Keras model.
	observations = []
	actions = []

	# Loop trhough data.
	for i in range(len(train_data)):
		observation = (train_data[i][0][0],train_data[i][0][1])
		action = (train_data[i][1][0],train_data[i][1][1],train_data[i][1][2])
		observations.append(observation)
		actions.append(action)

	# Convert to numpy arrays.
	observations = np.asarray(observations)
	actions = np.asarray(actions)

	model = Sequential() # Make a linear neural network model.
	model.add(Dense(100,input_dim=len(observations[0]),activation='relu')) # Set up some nodes with relu activation function.
	model.add(Dense(50,activation='relu'))
	model.add(Dense(len(actions[0]),activation='linear')) # Linear activation.
	model.compile(loss='mse',optimizer=Adam())

	history = model.fit(observations,actions,epochs=train_iterations)

	loss_plotter(history)

	return model

def play_game(model,visual=True):
	sum_reward = [] # Total sum of the rewards during a single game.
	succesful_games_counter = 0 # Counter for keeping track of succesful games.

	# Loop through number of games.
	for i in range(play_games):
		sum_reward = 0

		# Loop through all the steps.
		for j in range(num_steps):
			# Render the mountain_car if visual is set to True.
			if visual == True:
				env.render() 
			if j == 0: action = 1 # Start the game with no push.
			else:
				action = np.argmax(model.predict(observation.reshape(-1, len(observation)))[0])

			observation,reward,done,info = env.step(action)

			# Add the rewards together.
			sum_reward += reward

			# Stop the game if num_steps is reached or position of the mountaintop is reached.
			if done:
				break

		# Reset game.
		env.reset()

		print(f'Game {i}: Total reward: {sum_reward}')

if __name__ == '__main__':
	# See what happens if Mountain car is launched with only random moves.
	#random_actions(printformat=1,visual=True)

	# Prepare data.
	train_data = make_train_sets()

	# Set up model and train it.
	model = data_trainer(train_data)

	# Allow your game to be played with the trained model.
	play_game(model,visual=True)
