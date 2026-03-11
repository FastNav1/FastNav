import torch
import torch.nn as nn
import torch.nn.functional as F

class DilateAttentionToken(nn.Module):
    """
    input  : [B, d, 1, L] (H=1, W=L).
    output : [B, 1, L, d]
    """
    def __init__(self, head_dim=32, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), 
                                dilation=(1,dilation), 
                                padding=(0, dilation*(kernel_size-1)//2),
                                stride=(1,1))
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, D//4, H, W
        B, d, H, W = q.shape    # [B, 256/4, 1, 7]
        assert H == 1, "Expect H==1 for token_mode"
        num_heads = d // self.head_dim
        N = W*H
        # reshape to [B, num_heads, head_dim, 1, L] 
        # permute to [B, num_heads, L, 1, head_dim]
        q = q.reshape([B, num_heads, self.head_dim, 1, N]).permute(0, 1, 4, 3, 2) # [B, 2, 7, 1, 64/2] -> B,h,N,1,d

        k_unf = self.unfold(k).reshape([B, d, self.kernel_size, N])  # [B, d, k, N]
        # reshape to [B, num_heads, N, d_head, k]
        k_unf = k_unf.reshape([B, num_heads, self.head_dim, self.kernel_size, N]).permute(0, 1, 4, 2, 3)  # B,h,N,d,k
        attn = (q @ k_unf) * self.scale  # B,h,N,1,k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # v unfold
        v_unf = self.unfold(v).reshape([B, d, self.kernel_size, N]) # # [B, d, k, N]
        v_unf = v_unf.reshape([B, num_heads, self.head_dim, self.kernel_size, N]).permute(0,1,4,3,2)  # B,h,N,k,d
        # B,h,N,1,d -> B,N,h,1,d -> B,N,D
        x = (attn @ v_unf).transpose(1,2).reshape(B, 1, N, d)  # [B,1,7,256/4]
        # convert to [B, D, 1, N] to be compatible upstream if needed
        # x = x.reshape(B, N, d).permute(0, 2, 1).unsqueeze(2)  # [B, d, 1, N]
        return x



class MultiDilateTokenAttention(nn.Module):
    """
    Multi-dilation attention for token sequences.
    Input: x_tokens [B, L, D]
    Output: x_out [B, L, D]
    """
    def __init__(self, dim=256, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0.1, proj_drop=0.1, kernel_size=3, dilation=(1,2,3,4)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads    # 32,
        self.kerbel_size = kernel_size
        self.dilation = list(dilation)
        self.num_dilation = len(dilation)   # 4 dilation blocks
        assert num_heads % self.num_dilation == 0, "num_heads must be divisible by num_dilation"

        # qkv总体：[B,D,1,L] -> [B, D*3, 1, L]
        self.qkv = nn.Conv2d(self.dim, self.dim*3, kernel_size=1, bias=qkv_bias)

        # create dilated attention heads groups
        self.dilate_attns = nn.ModuleList([
            DilateAttentionToken(self.head_dim, qk_scale=qk_scale, attn_drop=attn_drop,
                                 kernel_size=kernel_size, dilation=d) for d in self.dilation
        ])
        self.norm1 = nn.LayerNorm(self.dim)  # first LayerNorm
        self.norm2 = nn.LayerNorm(self.dim)  # second LayerNorm

        self.ff = nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Dropout(proj_drop),
        nn.Linear(dim * 4, dim),
        nn.Dropout(proj_drop),
        )

    def forward(self, x_tokens):
        x_ = x_tokens
        x_tokens = self.norm1(x_tokens)
        # x_tokens [B, L, D] [B, 7, 256]
        B, L, D = x_tokens.shape
        assert D == self.dim, "input feature dimension must be equal to dim"
        # reshape to [B, C=D, H=1, W=L]
        x = x_tokens.permute(0, 2, 1).contiguous().view(B, D, 1, L) # [B, 256, 1, 7]
        qkv = self.qkv(x) # [B, 3*D, 1, L]

        # [num_dilation, 3, B, D//num_dilation, 1, L]
        qkv = qkv.reshape(B, 3, self.num_dilation, D // self.num_dilation, 1, L).permute(2,1,0,3,4,5)

        # [num_dilation, B, 1, L, D//num_dilation]
        x_list = []

        for i in range(self.num_dilation):
            # [num_dilation, B, 1, L, D//num_dilation]
            x_i = self.dilate_attns[i](qkv[i][0], qkv[i][1], qkv[i][2])  # [B,1,7,256/4]
            x_list.append(x_i)
        
        # [4, B, 1, 7, 64] -> [B, 1, 7, 4, 64] -> [B, 1, 7, 256]
        x = torch.stack(x_list, dim=0).permute(1, 2, 3, 0, 4).reshape(B, 1, L, D) # B,H,W,D
        
        x = self.ff(x)

        x = x.squeeze(1)    # [B,L,D]
        
        return x
    
if __name__ == '__main__':
    x = torch.rand(2, 7, 256)
    dilation_attn = MultiDilateTokenAttention()
    y = dilation_attn(x)

    print(y.shape) # torch.Size([2, 7, 256])