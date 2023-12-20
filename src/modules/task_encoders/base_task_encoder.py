import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseTaskEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(BaseTaskEncoder, self).__init__()

        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )
    
    def forward(self, task_indices):
        return self.embedding(task_indices)