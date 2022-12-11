# Imports

import tensorflow
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import argparse
import sys
import gym
import numpy as np
import pyglet
import CONV_preprocessing
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import os
from os import path,remove
from CONV_preprocessing import preprocess

#Loading model

if not os.path.isdir("datas/reinf_learning_model"):
    os.makedirs("datas/reinf_learning_model")

path = 'datas/' 
model = load_model(path + 'reinf_learning_model')

#Parsing

parser = argparse.ArgumentParser()
#Selecting environment
parser.add_argument("--env-name", default="Duckietown-udem1-v0")
#Selecting the map:
  
  #Possible Maps:
  # -4way
  # -loop_dyn_duckiebots
  # -loop_empty
  # -loop_obstacles
  # -loop_pedestrians
  # -small_loop
  # -small_loop_cw
  # -straight_road
  # -udem1
  # -zigzag_dists

parser.add_argument("--map-name", default="loop_empty")
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

# Beggining of test simulation

while True:
	env.reset()
	env.render()
        
	# init
	observation = env.render('rgb_array')
	state = preprocess(observation)
	done = False
	tot_reward = 0.0
        
	while not done:
	    env.render()           
	    state=tensorflow.keras.utils.img_to_array(state)  
	    state/=255  
	    state = np.expand_dims(state,axis=0)
	    
		# State predicting

	    y_pred = model.predict(state) 
	    button = np.argmax(y_pred)
	    action = np.array([0.0, 0.0])
	    wheel_distance = 0.102
	    min_rad = 0.065
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

	    v1 = 0.5*action[0]
	    v2 = 0.75*action[1]

	    # Limit radius of curvature

	    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
	        
	        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
	        v1 += abs(delta_v)
	        v2 -= delta_v

	    action[0] = v1
	    action[1] = v2       
	    
	    observation, reward, done, info = env.step(action)
	    state = preprocess(observation)  
	    tot_reward += reward

	print("Test ended. Total reward:",tot_reward)
