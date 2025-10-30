import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        # factorized embedding
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden) # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len)

        # factorized embedding
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads
        
    def split_last(self, x, shape):
        "split the last dimension to given shape"
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
        return x.view(*x.size()[:-1], *shape)

    def merge_last(self, x, n_dims):
        "merge the last n_dims to a dimension"
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)


    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = F.softmax(scores, dim=-1)
        self.scores = scores # for visualization
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge_last(h, 2)
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    
class EncoderBlock(nn.Module):
    """Bert encoder block with Self-Ateention and Position-Wise FFN. Return result of forward and embeding"""
    def __init__(self, cfg):
        super().__init__()
        
        self.block_embedding_mode = cfg.block_embedding_mode
        
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        
    def forward(self, x):
        h = self.attn(x)
        h = self.norm1(x + self.proj(h))
        if self.block_embedding_mode == 'hidden':
            embedding = h.clone()
        h = self.norm2(h + self.pwff(h))
        if self.block_embedding_mode == 'not_hidden':
            embedding = h.clone()
        return h, embedding

    
class BioBert(nn.Module):
    """ Transformer with Encoder Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.use_parametr_sharing_strategy = cfg.use_parametr_sharing_strategy
        self.n_layers = cfg.n_layers
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        
        if not self.use_parametr_sharing_strategy:
            # Original BERT not used parameter-sharing strategies
            self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        else:
            # To used parameter-sharing strategies
            self.blocks = EncoderBlock(cfg)
        self.embedding_mode = cfg.embedding_mode
    
    def calculateEmbedding(self, embeddings, mode):
        """Colculating embeddimgs in different modes"""
        if mode == 'last':
            return embeddings[-1]
        elif mode == 'sum_all':
            sum = embeddings[0]
            for embedding in embeddings[1:]:
                sum += embedding
            return sum

    def forward(self, x):
        h = self.embed(x)
        embeddings = []
        if self.use_parametr_sharing_strategy:
            for _ in range(self.n_layers):
                h, embedding = self.blocks(h)
                embeddings.append(embedding)
        else:
            for block in self.blocks:
                h, embedding = block(h)
                embeddings.append(embedding)
        h = self.decoder(h)
        return h, self.calculateEmbedding(embeddings, self.embedding_mode)

    
'''
Classification
'''
class BinaryClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, cfg.hidden))
        self.lin1 = nn.Linear(cfg.hidden, 2)
        self.attn_pool = None
        
    def forward(self, x):
        w = self.q.expand(x.size(0), -1, -1)
        attn_pool = torch.softmax(w @ x.transpose(1, 2), dim=-1)
        self.attn_pool = attn_pool
        x = attn_pool @ x
        logits = self.lin1(x.squeeze(1))
        return logits

    
class BioBertClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = BioBert(cfg)
        self.classification_blocks = BinaryClassifier(cfg)
    
    def forward(self, x):
        _, embedding = self.bert(x)
        logits = self.classification_blocks(embedding)
        return logits
    
    def _load_bert(self, state_dict):
        self.bert.load_state_dict(state_dict)
    
    def _initialize(self):
        def init_weights(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.model.apply(init_weights)
    