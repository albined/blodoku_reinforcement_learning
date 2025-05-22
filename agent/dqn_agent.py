import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agent.model import DuelingDQN as DQN
from agent.replay_buffer import ReplayBuffer
from agent.utils import encode_state

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.replay_buffer = ReplayBuffer()
        self.batch_size = 256
        self.action_dim = action_dim

    def select_action(self, state, action_mask):
        if np.random.rand() < self.epsilon:
            valid_indices = np.where(action_mask)[0]
            return np.random.choice(valid_indices)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Shape: (1, C, H, W)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)  # Shape: (num_actions,)
        
        # Mask invalid actions by setting their Q-values to -inf
        q_values[~torch.tensor(action_mask).to(self.device)] = float('-inf')
        
        return torch.argmax(q_values).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, next_action_masks = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        next_action_masks = torch.BoolTensor(next_action_masks).to(self.device)
        
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            # next_q_values_target = self.target_network(next_states)
            # next_q_values_target[~next_action_masks] = -1e6
            # max_next_q_values = next_q_values_target.max(1)[0]
            
            # Above normal below double dqn
            next_q_values_online = self.q_network(next_states)
            next_q_values_online[~next_action_masks] = -1e6
            best_next_actions = next_q_values_online.argmax(1)

            # Use target network to evaluate selected action
            next_q_values_target = self.target_network(next_states)
            max_next_q_values = next_q_values_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

            
            target_q = rewards + self.gamma * max_next_q_values * (1 - dones)
            target_q = torch.clamp(target_q, -10, 10)

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    # def update_target(self):
    #     self.target_network.load_state_dict(self.q_network.state_dict())
        
    def update_target(self, tau=0.005):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
