import torch
import torch.nn as nn

class ContextUnet(nn.Module):
    def __init__(self, in_channels=1, n_feat=64, n_classes=10):
        super(ContextUnet, self).__init__()
        self.n_feat = n_feat

        # FIX 1: Time Embedding must match the bottleneck size (2 * n_feat)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, n_feat),
            nn.GELU(),
            nn.Linear(n_feat, 2 * n_feat), # Changed output to 2*n_feat to match down2
        )
        
        self.init_conv = nn.Conv2d(in_channels, n_feat, 3, padding=1)
        self.down1 = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, padding=1), nn.GroupNorm(8, n_feat), nn.GELU())
        self.down2 = nn.Sequential(nn.Conv2d(n_feat, 2 * n_feat, 3, padding=1), nn.GroupNorm(8, 2 * n_feat), nn.GELU())
        
        self.up1 = nn.Sequential(nn.Conv2d(2 * n_feat, n_feat, 3, padding=1), nn.GroupNorm(8, n_feat), nn.GELU())
        self.up2 = nn.Sequential(nn.Conv2d(2 * n_feat, n_feat, 3, padding=1), nn.GroupNorm(8, n_feat), nn.GELU())
        self.out = nn.Conv2d(2 * n_feat, in_channels, 3, padding=1)

    def forward(self, x, t):
        # Embed time
        t_emb = self.time_mlp(t) 
        # Extend to [Batch, 2*n_feat, 1, 1] to broadcast over spatial dims
        t_emb = t_emb[(..., ) + (None, ) * 2] 
        
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1) 
        
        # Now shapes match: down2 is [B, 128, 28, 28] and t_emb is [B, 128, 1, 1]
        up1 = self.up1(down2 + t_emb) 
        
        # Concatenate skip connection
        up2 = self.up2(torch.cat((up1, down1), 1))
        
        return self.out(torch.cat((up2, x), 1))

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        self.n_T = n_T
        self.device = device

        for k, v in betas.items():
            self.register_buffer(k, v)

    def forward(self, x, c=None):
        """
        Training Step
        """
        # FIX 2: Create a 1D tensor [Batch], not 2D [Batch, 1]
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0], )).to(self.device)
        
        noise = torch.randn_like(x) 
        
        # Now indexing works correctly creating [Batch, 1, 1, 1]
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )

        # We normalize t to [0, 1] and ensure it has shape [Batch, 1] for the MLP
        return self.nn_model(x_t, _ts.view(-1, 1) / self.n_T), noise

    def sample(self, n_sample, size, device):
        x_i = torch.randn(n_sample, *size).to(device) 
        history = [] 
        
        self.nn_model.eval()
        with torch.no_grad():
            for i in range(self.n_T, 0, -1):
                t_is = torch.tensor([i / self.n_T]).to(device)
                t_is = t_is.repeat(n_sample, 1) # Shape [Batch, 1]

                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

                eps = self.nn_model(x_i, t_is)
                
                # Use standard integers for indexing buffers
                # We need to grab the specific value for timestep 'i'
                # Since 'i' is an int, we don't need fancy indexing, just scalar access
                
                alpha_val = self.oneover_sqrta[i]
                mab_val = self.mab_over_sqrtmab[i]
                beta_val = self.sqrt_beta_t[i]
                
                x_i = (
                    alpha_val * (x_i - eps * mab_val)
                    + beta_val * z
                )
                
                if i % 20 == 0 or i == 1:
                    history.append(x_i.cpu().numpy())

        return x_i, history