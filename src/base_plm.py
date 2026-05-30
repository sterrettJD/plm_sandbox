import torch
import torch.nn as nn

class ProteinLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                      num_heads: int = 4, num_layers: int = 4):
        super(ProteinLanguageModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers)
        
        self.linear_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)           # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        encoded_output = self.encoder(x) # (batch, seq_len, embedding_dim)
        logits = self.linear_head(encoded_output)  # (batch, seq_len, vocab_size)
        return logits