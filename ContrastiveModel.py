import torch
import torch.nn as nn


class ContrastiveTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_embed_dim):
        super(ContrastiveTransformerModel, self).__init__()

        self.embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(embed_dim, nhead=8, batch_first=True),
            num_layers=3,
            norm=torch.nn.LayerNorm([embed_dim]),
            enable_nested_tensor=False
        )

        self.projection = torch.nn.Linear(embed_dim, output_embed_dim)

    def forward(self, tokenizer_output):
        x = self.embedding_layer(tokenizer_output['input_ids'])
        x = self.encoder(x, src_key_padding_mask=tokenizer_output['attention_mask'].logical_not())
        cls_embed = x[:, 0, :]
        return self.projection(cls_embed)
