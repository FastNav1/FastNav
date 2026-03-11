import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    
class TFDR(nn.Module):
    """
    input:[B,L,D]
    output:[B,L,D]
    """
    def __init__(self, seq_len=7, dim=256, reduction=16): 
        super(TFDR, self).__init__() 
        hidden_dim = (seq_len + dim) // reduction
        self.avg_pool_L = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_D = nn.AdaptiveAvgPool1d(1)

        self.act = h_swish()

        self.mlp = nn.Sequential(
            nn.Conv1d(seq_len + dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            self.act,
            nn.Conv1d(hidden_dim, seq_len + dim, 1),
            nn.Sigmoid()
        )

        # Temporary Storage: For Visualization Only (while navigation), Not for Backpropagation
        self.extracted_T_weights = None
        self.extracted_actual_weights = None
    
    def forward(self, x):
        # [B, L, D]
        _x = x
        B, L, D = x.shape
        # [B, L, D] -> [B, L, 1]
        x_T = self.avg_pool_L(x)
        # [B, L, D] -> [B, D, L] -> [B, D, 1] 
        x_D = self.avg_pool_D(x.permute(0, 2, 1))

         # [B, L+D, 1]
        combined = torch.cat([x_T, x_D], dim=1)
        weights = self.mlp(combined)  # [B, L+D, 1]
        T_weights = weights[:, :L, :]  # [B, L, 1]
        # print(T_weights.shape)
        D_weights = weights[:, L:, :].permute(0, 2, 1)  # [B, D, 1]->[B, 1, D]
        # print(D_weights.shape)

        # Temporary Storage: For Visualization Only (while navigation), Not for Backpropagation
        self.extracted_T_weights = T_weights.detach().cpu() # [B, 7, 1]
        self.extracted_actual_weights = (T_weights * D_weights).detach().cpu() #  [B, 7, D]

        # apply weights to tokens
        out = _x * T_weights * D_weights
        return out
    
if __name__ == '__main__':
    x = torch.rand(2, 7, 256)
    reweighting = TFDR()
    y = reweighting(x)

    print(y.shape) # torch.Size([2, 7, 256])