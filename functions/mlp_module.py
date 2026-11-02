import torch
from torch import nn


class AmmoniaMLP(nn.Module):
    """Predict one ammonia emission value per row of tabular data."""

    def __init__(self, input_size, output_size, cat_dims, embedding_dims):
        super().__init__()

        self.n_cont = input_size - len(cat_dims)

        self.embeddings = nn.ModuleList([
            nn.Embedding(int(cat_dim), int(embed_dim))
            for cat_dim, embed_dim in zip(cat_dims, embedding_dims)
        ])

        self.fc1 = nn.Linear(self.n_cont + sum(embedding_dims), 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 12)
        self.fc4 = nn.Linear(12, 6)
        self.fc5 = nn.Linear(6, output_size)

        # Garder un gradient même pour les activations négatives.
        self.relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x_cont, x_cat):

        x_embeds = [
            embedding(x_cat[:, j])
            for j, embedding in enumerate(self.embeddings)
        ]

        x = torch.cat([x_cont] + x_embeds, dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))

        return self.fc5(x)

