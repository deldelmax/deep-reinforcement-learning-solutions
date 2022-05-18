import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworkMlp(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        hs1 = 128
        hs2 = 128
        hs3 = 128
        hs4 = 128
        self.net = nn.Sequential(
            nn.Linear(state_size, hs1),
            nn.ReLU(),
            nn.Linear(hs1, hs2),
            nn.ReLU(),
            nn.Linear(hs2, hs3),
            nn.ReLU(),
            nn.Linear(hs3, hs4),
            nn.ReLU(),
            nn.Linear(hs4, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.net(state)
