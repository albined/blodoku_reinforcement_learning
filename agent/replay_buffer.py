import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_action_mask):
        self.buffer.append((state, action, reward, next_state, done, next_action_mask))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_action_masks = map(np.array, zip(*samples))
        return states, actions, rewards, next_states, dones, next_action_masks

    def __len__(self):
        return len(self.buffer)
