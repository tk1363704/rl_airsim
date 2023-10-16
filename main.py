"""
Deep RL training for obstacle avoidance in AirSim
Author : Varun Pawar
E-mail : varunpwr897@gmail.com
"""
import datetime
import logging
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

PROJECT_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

import cv2
import csv
import statistics
import time

from exp_replay import PrioritizedExperienceReplay
from network import DQN, CnnDQN, DuelingCnnDQN
# Argument parser
parser = argparse.ArgumentParser(description='Some input flags for the PERD3QN')
parser.add_argument('--env', default='DroneEnv', help='AirSim Quadrotor Environment by default')
parser.add_argument('--dueling', default=False, action='store_true', help='Load if required Dueling')
parser.add_argument('--load_model', default=False, action='store_true', help='Load model')
parser.add_argument('--model_path', default='log\\model\\PRDDQN_model.pth', help='Model Path')
parser.add_argument('--optimizer_path', default='log\\model\\PRDDQN_optimizer.pth', help='Optimizer Path')
parser.add_argument('--log_path', default='log\\train\\', help='Log Path')
parser.add_argument('--update_freq', default=10, help='Update frequency of model')
parser.add_argument('--save_freq', default=5000, help='Save frequency of model')
parser.add_argument('--save_function', default='best_distance', help='save function: [initial, best_distance (save the model that are the closest to the destination)]')
parser.add_argument('--num_frames', default=100000, help='Number of Frames for training')
parser.add_argument('--batch_size', default=32, help='Training batch size')
parser.add_argument('--replay_size', default=10000, help='Replay memory size')
parser.add_argument('--replay_initial', default=1000, help='Initial untrained replay')
parser.add_argument('--reward', default='dist', help='reward design: [initial, dist(distance to destination), dist_with_heuristic]')
parser.add_argument('--directions', default='3_minus_y', help='the directions that drone can move towards: [4(4 discrete directions), 3(original setting), 3_minus_y(original setting but y can move towards backward direction, x can only move toward forward), 3_customized(3 discrete directions)]')

args = parser.parse_args()

file_path = PROJECT_ABSOLUTE_PATH + '\\'+ args.log_path + '\\{}_{}_{}_directions_{}.log'.format(time.strftime("%Y%m%d-%H%M%S"), args.reward, args.save_function, args.directions)
directory_path = os.path.dirname(file_path)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
logging.basicConfig(filename=file_path, level=logging.INFO)


# Log file
timestr = time.strftime("%Y%m%d-%H%M%S")
csv_name = args.log_path + 'statistics' + timestr + '.csv'
fields = ['Episodes done','Episode Reward', 'Frames Done', 'Loss', 'framerate']    
# writing to csv file  
with open(csv_name, 'w', newline = '') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

# Initialize Environment
if args.env == 'DroneEnv':
    env = DroneEnv(reward_design=args.reward, logging=logging)
else:
    env = gym.make(args.env)

# Initialize Epsilon
epsilon_start = 1.0
# epsilon_start = 0.01
epsilon_final = 0.01
epsilon_decay = 10000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# Initialize Beta
beta_start = 0.4
beta_frames = 1000 
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

# Initialize NN
direction_num = int(args.directions.split('_')[0])
if args.dueling:
    current_model = DuelingCnnDQN([1,32,32], direction_num)
    target_model = DuelingCnnDQN([1,32,32], direction_num)
else:
    current_model = CnnDQN([1,32,32], direction_num)
    target_model  = CnnDQN([1,32,32], direction_num)

# USE CUDA if available
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()

# NN Parameters
maxlr = 0.0001   
optimizer = optim.Adam(current_model.parameters(), lr = maxlr)

# Training Parameters
replay_initial = int(args.replay_initial)
replay_size = int(args.replay_size)
replay_buffer = PrioritizedExperienceReplay(replay_size)

# Load model if continuing training
if args.load_model:
    current_model.load_state_dict(torch.load(args.model_path))
    optimizer.load_state_dict(torch.load(args.optimizer_path))

# Update target model with current model
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
    print("Update target model with current model!")

update_target(current_model, target_model)

# Adjust learning rate
def adjust_learning_rate(optimizer, frame_idx):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lrr = maxlr * (0.1 ** (frame_idx//replay_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrr

# Compute TD-error for a given batch
def compute_td_loss(batch_size, beta):
    state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta) 

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))
    weights    = Variable(torch.FloatTensor(weights))

    q_values      = current_model(state.unsqueeze(1))
    next_q_values = target_model(next_state.unsqueeze(1))

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    # If done is True, it means the episode is finished, so the next Q-value doesn't contribute to the target.
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss  = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss  = loss.mean()
        
    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()
    
    return loss

num_frames = int(args.num_frames)
batch_size = int(args.batch_size)
gamma      = 0.99

global_mini_distance = 100000.0

no_episodes = 0
losses = [0]
all_rewards = []
episode_reward = 0
current_time = 0.001
prev_time = 0
framerate = 0
state = env.reset()
env.setObsRandom()
print("--------------Episode {}--------------:".format(str(no_episodes)))
logging.info("--------------Episode {}--------------:".format(str(no_episodes)))

for frame_idx in range(1, num_frames + 1):
    framerate = (1 - math.exp(-(frame_idx-1)/1000))*framerate + math.exp(-(frame_idx-1)/1000)/(current_time - prev_time)
    prev_time = current_time
    current_time = time.time()

    epsilon = epsilon_by_frame(frame_idx)
    action = current_model.act(state, epsilon)
    print("-------Action:", action)
    logging.info("-------Action: {}".format(str(action)))
    env.setObsDynamic()

    next_state, reward, done = env.step(action, args.directions)
    print("-------reward:", reward)
    logging.info("-------Reward: {}".format(str(reward)))
    # replay_buffer.push(state, action, Fmain, next_state, done)
    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward
    if done:
        if args.save_function == 'best_distance':
            if not env.client.simGetCollisionInfo().has_collided:
                current_dist = env.distance_to_destination()
                if current_dist < global_mini_distance:
                    path_ = PROJECT_ABSOLUTE_PATH + '\\log\\model\\reward_{}_savefunc_{}_directions_{}\\PRDDQN_currentdist_{}_model.pth'.format(args.reward, args.save_function, args.directions, str(int(current_dist)))
                    directory_path = os.path.dirname(path_)
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                    torch.save(current_model.state_dict(),
                               path_)
                    optimizer_path_ = PROJECT_ABSOLUTE_PATH + '\\log\\model\\reward_{}_savefunc_{}_directions_{}\\PRDDQN_currentdist_{}_optimizer.pth'.format(args.reward, args.save_function, args.directions, str(int(current_dist)))
                    torch.save(optimizer.state_dict(),
                               optimizer_path_)
                    print("Torch saving best model...")
                    logging.info("Torch saving best model...")
                    global_mini_distance = current_dist
                    logging.info('global_mini_distance is updated to {}'.format(str(global_mini_distance)))
                    print('global_mini_distance is updated to {}'.format(str(global_mini_distance)))

        no_episodes += 1
        print("--------------Episode {}--------------:".format(str(no_episodes)))
        logging.info("--------------Episode {}--------------:".format(str(no_episodes)))
        state = env.reset()
        all_rewards.append(episode_reward)
        
        data = [str(no_episodes), str(episode_reward), str(frame_idx), statistics.mean(losses), framerate]
        losses = [0]
        with open(csv_name, 'a', newline = '') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(data)
        episode_reward = 0
        env.setObsRandom()
        if no_episodes % int(args.update_freq) == 0:
            # update_target(policy_net, target_net)
            update_target(current_model, target_model)

        
    print('-------Frame Rate:', framerate)
    logging.info('-------Frame Rate: {}'.format(str(framerate)))
    if (len(replay_buffer) > replay_initial):
        adjust_learning_rate(optimizer, frame_idx)
        beta = beta_by_frame(frame_idx)
        loss = compute_td_loss(batch_size, beta)
        losses.append(loss.item())

    if frame_idx % 100 == 0:
        print("frame_idx is {}".format(frame_idx))
        logging.info("frame_idx is {}".format(str(frame_idx)))

    if frame_idx % int(args.save_freq) == 0:
        path_ = PROJECT_ABSOLUTE_PATH + '\\log\\model\\reward_{}_savefunc_{}_directions_{}\\PRDDQN_freq_{}_model.pth'.format(args.reward, args.save_function, args.directions, str(int(frame_idx/5000)))
        directory_path = os.path.dirname(path_)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        torch.save(current_model.state_dict(), path_)
        optimizer_path = PROJECT_ABSOLUTE_PATH + '\\log\\model\\reward_{}_savefunc_{}_directions_{}\\PRDDQN_freq_{}_optimizer.pth'.format(args.reward, args.save_function, args.directions, str(int(frame_idx/5000)))
        torch.save(optimizer.state_dict(), optimizer_path)
        print("Torch saving ...")
        logging.info("Torch saving ...")

# torch.save(current_model.state_dict(), 'log\\model\\PRDDQN_{}.pth'.format(args.reward))
# torch.save(optimizer.state_dict(), 'log\\model\\PRDDQN_{}_optimizer.pth'.format(args.reward))
# print("Torch saving ...")
# print("Training ends here ...")
# logging.info("Torch saving ...")
# logging.info("Training ends here ...")


