import os

import gym
from env import DroneEnv
import numpy as np
import math, random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib.pyplot as plt

import logging
import cv2
import csv
import statistics
import time
import datetime

from exp_replay import PrioritizedExperienceReplay
from network import DQN, CnnDQN, DuelingCnnDQN

PROJECT_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


# Argument parser
parser = argparse.ArgumentParser(description='Some input flags for testing PERD3QN')
parser.add_argument('--dueling', default=False, action='store_true', help='Load if required Dueling')
parser.add_argument('--load_model', default=True, action='store_true', help='Load model')
parser.add_argument('--model_path', default='PRDDQN.pth', help='Model Path')
parser.add_argument('--test_episode', default=50, help='Testing episodes')
parser.add_argument('--log_path', default='log\\test\\', help='Log Path')


args = parser.parse_args()

file_path = PROJECT_ABSOLUTE_PATH + '\\'+ args.log_path + '\\{}.log'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
directory_path = os.path.dirname(file_path)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
logging.basicConfig(filename=file_path, level=logging.INFO)


# # Log file
# timestr = time.strftime("%Y%m%d-%H%M%S")
# csv_name = args.log_path + 'statistics' + timestr + '.csv'
# fields = ['Episodes done','Episode Reward', 'Frames Done', 'Loss', 'framerate']
# # writing to csv file
# with open(csv_name, 'w', newline = '') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(fields)

# Initialize Epsilon
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 10000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# Initialize Environment
env = DroneEnv()

# Initialize NN
if args.dueling:
    target_model = DuelingCnnDQN([1,32,32], 3)
else:
    target_model  = CnnDQN([1,32,32], 3)

# USE CUDA if available
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
if USE_CUDA:
    target_model  = target_model.cuda()

# Load model if continuing training
if args.load_model:
    path_ = PROJECT_ABSOLUTE_PATH + '\\{}'.format(args.model_path)
    target_model.load_state_dict(torch.load(path_))

time_list = []

for e_ in range(int(args.test_episode)):
    # print("--------------Episode {}--------------:".format(e_))
    logging.info("--------------Episode {}--------------:".format(str(e_)))
    env.reset_destination()
    logging.info("--------------Destination coordinate {}--------------:".format(str(env.get_destination())))
    test_current_time = time.time()
    logging.info("--------------test_current_time: {}--------------:".format(str(test_current_time)))
    current_time = 0.001
    prev_time = 0
    framerate = 0
    state = env.reset()
    env.setObsRandom()
    frame_idx = 1
    while True:
        framerate = (1 - math.exp(-(frame_idx - 1) / 1000)) * framerate + math.exp(-(frame_idx - 1) / 1000) / (
                    current_time - prev_time)
        prev_time = current_time
        current_time = time.time()
        epsilon = epsilon_by_frame(frame_idx)
        action = target_model.act(state, epsilon)
        logging.info("-------Action {}-------: {}".format(str(frame_idx), str(action)))
        # print("-------Action {}-------:".format(frame_idx), action)
        env.setObsDynamic()

        next_state, reward, done = env.step(action)
        logging.info("Distance to destination is {}".format(str(env.distance_to_destination())))
        # print("Distance to destination is {}".format(env.distance_to_destination()))
        frame_idx += 1

        if env.is_collision():
            # print('Unfortunately!\nYou have collided with the obstacles!')
            logging.info('Unfortunately!\nYou have collided with the obstacles!')
            time_list.append(-10.0)
            break
        elif env.achieve_destination():
            time_list.append(time.time() - test_current_time)
            # print('Congrats!\nYou have reached the destination!')
            logging.info('Congrats!\nYou have reached the destination!')
            break
        elif frame_idx > 500:
            # print('Unfortunately!\nYou have taken too may steps!')
            logging.info('Unfortunately!\nYou have taken too may steps!')
            time_list.append(-2.0)
            break

positive_time_list= [x for x in time_list if x > 0.0]
logging.info('Finish rate is {}'.format(str(float(len(positive_time_list)/len(time_list)))))
logging.info('Collision rate is {}'.format(str(float(time_list.count(-10.0)/len(time_list)))))
logging.info('Over step limit rate is {}'.format(str(float(time_list.count(-2.0)/len(time_list)))))
average_time = -1 if len(positive_time_list) == 0 else float(sum(positive_time_list)) / len(positive_time_list)
logging.info('Average finish time is {} seconds'.format(str(average_time)))