import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

# 1. 位置编码
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=20, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].detach()
        return self.dropout(x)

class LearnablePE(nn.Module):
    def __init__(self, d_model, max_len=20, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class NoPE(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(x)

# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
    def forward(self, Q, K, V, mask=None):
        # 分别获取 Q, K, V 的 batch size 和序列长度
        B_q, T_q, C = Q.shape
        B_k, T_k, _ = K.shape
        B_v, T_v, _ = V.shape

        # 线性变换 + 分头
        Q = self.Wq(Q).view(B_q, T_q, self.h, self.d_k).transpose(1, 2)  # (B, h, T_q, d_k)
        K = self.Wk(K).view(B_k, T_k, self.h, self.d_k).transpose(1, 2)  # (B, h, T_k, d_k)
        V = self.Wv(V).view(B_v, T_v, self.h, self.d_k).transpose(1, 2)  # (B, h, T_v, d_k)

        # 注意力计算
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, h, T_q, T_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B_q, T_q, C)  # 拼回
        return self.Wo(out)

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x): return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        return self.norm3(x + self.dropout(self.ffn(x)))

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=3, num_heads=4,
                 d_ff=512, max_len=15, dropout=0.1, pe_type="sin"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        if pe_type == "sin": self.pe = SinusoidalPE(d_model, max_len, dropout)
        elif pe_type == "learnable": self.pe = LearnablePE(d_model, max_len, dropout)
        else: self.pe = NoPE(dropout)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size)
    def make_masks(self, src, tgt, pad_idx=0):
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        T = tgt.shape[1]
        nopeak = torch.triu(torch.ones(1, 1, T, T), diagonal=1) == 0
        tgt_mask = tgt_mask & nopeak.to(src.device)
        return src_mask, tgt_mask
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.make_masks(src, tgt)
        enc_x = self.pe(self.emb(src))
        for layer in self.enc_layers: enc_x = layer(enc_x, src_mask)
        dec_x = self.pe(self.emb(tgt))
        for layer in self.dec_layers: dec_x = layer(dec_x, enc_x, src_mask, tgt_mask)
        return self.head(dec_x)

# 3. 排序任务数据生成
def get_sort_data(batch_size, seq_len=6, vocab_size=20):
    """输入: [5, 2, 7, 1] -> 输出: [1, 2, 5, 7] (排序)"""
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt, _ = src.sort(dim=1)
    return src, tgt

# 4. 训练函数
def train(pe_type, epochs=150):
    print(f"\n=== {pe_type} ===")
    vocab_size = 20
    model = Transformer(vocab_size, d_model=128, num_layers=3, num_heads=4, pe_type=pe_type)
    opt = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    model.train()
    for i in range(epochs):
        src, tgt = get_sort_data(128)
        opt.zero_grad()
        out = model(src, tgt[:, :-1])
        loss = criterion(out.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if i % 30 == 0:
            print(f"Step {i:3d} | Loss: {loss.item():.4f}")
    print(f"===  {pe_type} ===")
    return losses

# 运行三组实验
print("start...")
loss_sin = train("sin")
loss_learn = train("learnable")
loss_none = train("none")

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(loss_sin, label="Sinusoidal PE", linewidth=2)
plt.plot(loss_learn, label="Learnable PE", linewidth=2, linestyle='--')
plt.plot(loss_none, label="No PE", linewidth=2, linestyle=':')
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Position Encoding Ablation Study (Sorting Task)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()