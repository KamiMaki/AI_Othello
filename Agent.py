import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from ReplayBuffer import ReplayBuffer
import logging
from Network import *

# 設置logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DDQNAgent():
    def __init__(self, **kwargs):
        self.state_size = kwargs.get('state_size', 64)
        self.action_size = kwargs.get('action_size', 64)
        self.seed = random.seed(kwargs.get('seed', 123))
        self.buffer_size = kwargs.get('buffer_size', 2**16)
        self.batch_size = kwargs.get('batch_size', 128)
        self.gamma = kwargs.get('gamma', 0.95)
        self.lr = kwargs.get('lr', 1e-3)
        self.update_every = kwargs.get('update_every', 10)
        self.epsilon = kwargs.get('eps', 0.1)
        self.save_record = kwargs.get('save_record', False)
        self.log_dir = kwargs.get('log_dir', None)
        self.net_type = kwargs.get('net_type', 'DNN')
        self.t_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.state_size, self.action_size)
    
        self.ddqn_local = self.get_NN(self.state_size, self.action_size).to(self.device)
        self.ddqn_target = self.get_NN(self.state_size, self.action_size).to(self.device)

        self.optimizer = optim.Adam(self.ddqn_local.parameters(), lr=self.lr)
        
        if self.save_record:
            self.writer = SummaryWriter(log_dir+'/agent')
            
    def get_NN(self,state_size, action_size):
        if self.net_type == 'DNN':
            return DNN(state_size, action_size)
        elif self.net_type == 'CNN':
            return CNN(state_size, action_size)
        elif self.net_type == 'AttentionNN':
            return AttentionNN(state_size,action_size)
        else: 
            raise ValueError(f'Unsupported network type: {self.net_type}')
                                 
    def save(self,filename):
        torch.save(self.ddqn_local.net,filename)
        logger.info(f"Model saved to {filename}")

    def load(self,filename):
        try:
            self.ddqn_local.net = torch.load(filename, map_location=self.device)
            self.ddqn_target.net = torch.load(filename, map_location=self.device)
            logger.info(f"Model loaded from {filename}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            
            
    def set_eps(self,eps):
        if eps >=0 and eps <=1:
            self.epsilon = eps
        else:
            logger.error(f"Set epsilon failed, wrong value:{eps}")
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1 
        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
                
    def get_action(self, state, legal_act):
        state = torch.from_numpy(state).float().unsqueeze(1).to(self.device)
        self.ddqn_local.eval()
        with torch.no_grad():
            action_values = self.ddqn_local(state)
        self.ddqn_local.train()
        actions = np.zeros(legal_act.shape[0])
         
        for i,act in enumerate(legal_act):
            if random.random() > self.epsilon: # exploit
                actions[i] = np.where(act)[0][action_values[i].cpu().numpy()[act].argmax()]
            else: # explore
                actions[i] = np.random.choice(np.arange(self.action_size)[act])
        return actions

    def get_Q_values(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        Q_expected = self.ddqn_local(states.to(self.device)).cpu().gather(1, actions.long()).max(1)[0].unsqueeze(1)
        return Q_expected.mean()
            
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.ddqn_target(next_states.to(self.device)).max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next.cpu() * (1 - dones))
        Q_expected = self.ddqn_local(states.to(self.device)).cpu().gather(1, actions.long()).max(1)[0].unsqueeze(1)

        loss = F.mse_loss(Q_expected, Q_targets)
        if self.save_record:
            self.writer.add_scalar("Critic loss",loss,self.t_step)
            self.writer.add_scalar("Q value",Q_expected.mean(),self.t_step)
            self.writer.flush()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.ddqn_local, self.ddqn_target)
        
    def soft_update(self, local_model, target_model, tau=1e-3):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# 隨機策略
class RandomAgent():
    def __init__(self):
        pass
    
    def get_action(self, state, legal_act):
        actions = np.array([np.random.choice(np.arange(legal_act.shape[1])[act]) for act in legal_act])
        return actions
        
    def step(self, state, action, reward, next_state, done):
        pass

# 選擇最佳位置策略
class PositionAgent():
    def __init__(self):
        self.position_value = np.array([[-100,8,5,4,4,5,8,-100],
                                        [8,-10,-6,-6,-6,-6,-6,8],
                                        [5,-6,0,0,0,0,-6,5],
                                        [4,-6,0,0,0,0,-6,4],
                                        [4,-6,0,0,0,0,-6,4],
                                        [5,-6,0,0,0,0,-6,5],
                                        [8,-10,-6,-6,-6,-6,-10,8],
                                        [-100,8,5,4,4,5,8,-100]]).flatten()
    
    def get_action(self, state, legal_act):
        mask = np.where(legal_act, self.position_value, -np.inf)
        actions = np.argmax(mask, axis=1)
        return actions  

    def step(self, state, action, reward, next_state, done): 
        pass




