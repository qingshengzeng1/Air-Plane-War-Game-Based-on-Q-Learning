
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from dqn.network import DQN
from dqn.replay_buffer import ReplayBuffer
from game.config import *

class Agent:
    def __init__(self, state_size, action_size, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q-Network
        self.policy_net = DQN(state_size, HIDDEN_SIZE, action_size)
        self.target_net = DQN(state_size, HIDDEN_SIZE, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        
        self.steps_done = 0
        self.epsilon = EPSILON_START

    def act(self, state):
        if random.random() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net(state)
            self.policy_net.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        if len(self.memory) > BATCH_SIZE:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1)
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().unsqueeze(1)
        
        # Get expected Q values from policy net
        q_expected = self.policy_net(states).gather(1, actions)
        
        # Get next Q values from target net
        q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())
