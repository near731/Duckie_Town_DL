# Imports

import create_env
from create_env import create_env, button2action, do_action
from Q_model import q_model
from Q_preprocess import preprocess
from collections import deque
import numpy as np
from random import sample
from keras.models import  load_model

# Loading the model

dir_ = 'datas/'
input_shape = (40, 80, 3)

#model,input_shape = q_model() #uncomment if model doesn't exist 
model=load_model(dir_ + 'q_model') #comment if model doesn't exist 

# Hyper Parameters

epochs = 50
observetime = 1000                       
epsilon = 0.5                             
decay = 0.95				   				
gamma = 0.5                                
mb_size = 30                               
env = create_env()

for epoch in range(epochs):
	
	#Start observing
	
	# Creating environment
	env.reset()
	
	obs = env.render('rgb_array')
	state = preprocess(obs)
	state = np.expand_dims(state,axis=0)
	done = False
	D = deque()

	
	for t in range(observetime):
			env.render()
			if np.random.rand() <= epsilon: # random action
					Q = np.random.uniform(low=0.0, high=1.0, size=(5,))
			else:
					state = state.reshape(1,40,80,3)
					Q = model.predict(state/255)

			button = np.argmax(Q)
			action = button2action(button)
			state_new, reward, done, info = do_action(env, button)
			D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
			state = state_new         # Update state
			if done:
					env.reset()           # Restart game if it's finished
					state = preprocess(x = env.render('rgb_array'))

	# Learning from the observations

	
	minibatch = sample(list(D), mb_size) 

	x_train_shape = (mb_size,) + input_shape
	x_train = np.zeros(x_train_shape)
	y_train = np.zeros((mb_size, 5))

	for i in range(0, mb_size):
			# Reading from minibatch

			state = minibatch[i][0]
			action = np.argmax(minibatch[i][1])
			reward = minibatch[i][2]
			state_new = minibatch[i][3]
			done = minibatch[i][4]

			# Build Bellman equation for the Q function
			x_train[i] = state
			state = state.reshape(1,40,80,3)
			y_train[i] = model.predict(state/255)
			state_new = state_new.reshape(1,40,80,3)
			Q_sa = model.predict(state_new/255)
			
			if done:
					y_train[i, action] = reward
			else:
					y_train[i, action] = reward + gamma * np.max(Q_sa)

			# Train network to output the Q function
			model.train_on_batch(x_train, y_train)
	
	epsilon *= decay
	print("\n \n \n")
	print("Epoch=", epoch+1, "/",epochs)
	print("\nnext epsilon = ", epsilon)
	print("\n \n \n")
	
	if epoch%25 == 0:	
		model.save(dir_ + 'q_model')
		print("saved")
