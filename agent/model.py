import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.model(x)

class DQN_CNN(nn.Module):
    def __init__(self, in_channels=17, hidden_dim=64, num_blocks=3, grid_height=10, grid_width=12):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)

        # Output Q-value per (block, x, y)
        self.head = nn.Conv2d(hidden_dim // 2, num_blocks, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Input:
            x: tensor of shape (B, C=17, H, W)
        Output:
            q_maps: tensor of shape (B, num_blocks, H, W)
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        q_maps = self.head(x)  # (B, num_blocks, H, W)
        return q_maps
