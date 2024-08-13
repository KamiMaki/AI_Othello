import torch
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_dim, action_dim, seed=123):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(seed)
        
        # Define data types for compatibility with PyTorch
        self.states = torch.empty((buffer_size, 8,8))
        self.next_states = torch.empty((buffer_size, 8,8))
        self.actions = torch.empty((buffer_size,1))
        self.rewards = torch.empty((buffer_size, 1))
        self.dones = torch.empty((buffer_size, 1))
        
        self.ptr = 0  # Pointer to the next insertion index
        self.size = 0  # Current size of the buffer

    def add(self, state, action, reward, next_state, done):
        # Convert numpy arrays or scalars to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        # Store the experience
        store_length = state.shape[0]
        end_idx = (self.ptr+store_length)%self.memory.maxlen
        if end_idx > self.ptr:
            self.states[self.ptr:end_idx] = state
            self.actions[self.ptr:end_idx] = action
            self.rewards[self.ptr:end_idx] = reward
            self.next_states[self.ptr:end_idx] = next_state
            self.dones[self.ptr:end_idx] = done
        else: # buffer is full
            self.states[self.ptr:] = state[:self.memory.maxlen-self.ptr]
            self.actions[self.ptr:] = action[:self.memory.maxlen-self.ptr]
            self.rewards[self.ptr:] = reward[:self.memory.maxlen-self.ptr]
            self.next_states[self.ptr:] = next_state[:self.memory.maxlen-self.ptr]
            self.dones[self.ptr:] = done[:self.memory.maxlen-self.ptr]

            self.states[:end_idx] = state[self.memory.maxlen-self.ptr:]
            self.actions[:end_idx] = action[self.memory.maxlen-self.ptr:]
            self.rewards[:end_idx] = reward[self.memory.maxlen-self.ptr:]
            self.next_states[:end_idx] = next_state[self.memory.maxlen-self.ptr:]
            self.dones[:end_idx] = done[self.memory.maxlen-self.ptr:]
            
        
        self.ptr = end_idx
        self.size = min(self.size + store_length, self.memory.maxlen)
    
    def update_reward(self,reward):
        self.rewards[-1] = reward
        self.dones[-1] = True
        
        
    def sample(self):
        # Sample a batch of experiences randomly
        try:
            idxs = np.random.choice(self.size, self.batch_size, replace=False)
        except:
            idxs = np.arange(len(self.states))
        
        batch_states = self.states[idxs].unsqueeze(1)
        batch_actions = self.actions[idxs]
        batch_rewards = self.rewards[idxs]
        batch_next_states = self.next_states[idxs].unsqueeze(1)
        batch_dones = self.dones[idxs]
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def __len__(self):
        return self.size