# Imports

import tensorflow
from tensorflow import keras
from PIL import Image
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from os import path,remove
from Q_preprocess import preprocess
from Q_model import q_model
import os

#Loading the Model

if not os.path.isdir("datas/q_model"):
    os.makedirs("datas/q_model")

path = 'datas/' 
model = keras.models.load_model(path + 'q_model')

# Creating Envirement

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="Duckietown-udem1-v0")

#Selecting map
parser.add_argument("--map-name", default="zigzag_dists")

parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()

if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
    )
else:
    env = gym.make(args.env_name)
while True:
	env.reset()
	env.render()

    # Game begins
	observation = env.render('rgb_array')
	state = preprocess(observation)
	done = False
	tot_reward = 0.0

	# Limit radius of curvature
	wheel_distance=0.102
	min_rad = 0.065


	while not done:
	    env.render()           
	    state=tensorflow.keras.utils.img_to_array(state)  
	    state/=255  
	    state = np.expand_dims(state,axis=0)
	    y_pred = model.predict(state) 
	    button = np.argmax(y_pred)
	    action = np.array([0.0, 0.0])
	    if button == 0: # UP
	        action += np.array([0.44, 0.0])
	        print('UP',end="\r")
	    if button == 1: # DOWN
	        action -= np.array([0.44, 0])
	        print('DOWN',end="\r")
	    if button == 2: # LEFT
	        action += np.array([0, 1])
	        print('LEFT',end="\r")
	    if button == 3: # RIGHT
	        action -= np.array([0, 1])
	        print('RIGHT',end="\r")
	    if button == 4: # SPACE
	        action = np.array([0, 0])
	        print('SPACE',end="\r")

	    v1 = 1.5*action[0]
	    v2 = 3*action[1]

	    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
	        # adjust velocities evenly such that condition is fulfilled
	        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
	        v1 += abs(delta_v)
	        v2 -= delta_v

	    action[0] = v1
	    action[1] = v2       
	    # action = ...        TO DO    
	    observation, reward, done, info = env.step(action)
	    state = preprocess(observation)  
	    tot_reward += reward
	print('Game ended! Total reward: {}'.format(reward))
