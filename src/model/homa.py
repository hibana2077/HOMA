import torch
import torch.nn as nn
import torch.nn.functional as F

class HOMA(nn.Module):
    def __init__(self, in_ch, out_dim=2048, rank=64, orders=(2, 3, 4)):
        super().__init__()
        self.orders = orders
        self.rank = rank
        
        # Pre-compute projection dimensions
        proj_dims = {
            2: in_ch**2,
            3: in_ch,
            4: in_ch
        }
        
        self.proj = nn.ModuleDict({
            str(k): nn.Linear(proj_dims[k], rank, bias=False)
            for k in orders
        })
        
        self.fuse = nn.Sequential(
            nn.Linear(rank * len(orders), out_dim),
            nn.Mish(inplace=True)
        )
        
        # Pre-register buffer for random tensor in order 3
        if 3 in orders:
            self.register_buffer('random_tensor', torch.randn(1, in_ch, 1, 1))

    def forward(self, x):  # x: B,C,H,W
        B, C, H, W = x.size()
        
        # Normalize input
        x_norm = x - x.mean(dim=(-1, -2), keepdim=True)
        
        feats = []
        
        # Process each order
        for order in self.orders:
            if order == 2:
                # Second-order moment (covariance)
                x_flat = x_norm.view(B, C, -1)
                m2 = torch.bmm(x_flat, x_flat.transpose(1, 2)).view(B, -1)
                feats.append(self.proj['2'](m2))
                
            elif order == 3:
                # Third-order approximation using fixed random tensor
                m3 = (x_norm * self.random_tensor).sum((-1, -2))
                feats.append(self.proj['3'](m3))
                
            elif order == 4:
                # Fourth-order cumulant
                x_mean = x_norm.mean((-1, -2), keepdim=True)
                m4 = (x_norm**2).mean((-1, -2)) - 3 * (x_mean.squeeze(-1).squeeze(-1)**2)
                feats.append(self.proj['4'](m4))
        
        # Concatenate and fuse features
        h = torch.cat(feats, dim=-1)
        return self.fuse(h)

if __name__ == "__main__":
    # Example usage
    model = HOMA(in_ch=3, out_dim=2048, rank=64, orders=(2, 3, 4))
    x = torch.randn(8, 3, 224, 224)  # Batch of 8 images with 3 channels and 224x224 resolution
    output = model(x)
    print(output.shape)  # Should be [8, 2048]