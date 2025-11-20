import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    """A tiny U-Net for simple 32x32 patterns."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        # Decoder
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, t):
        # (Simplification: Ignoring 't' embedding for brevity, assuming fixed noise schedule)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x_up = torch.nn.functional.interpolate(x2, scale_factor=2)
        return self.out(self.dec1(x_up + x1)) # Skip connection is crucial for gradient flow

class PatternDDPM(nn.Module):
    def __init__(self, n_steps=50):
        super().__init__()
        self.network = SimpleUNet()
        self.n_steps = n_steps
        self.betas = torch.linspace(1e-4, 0.2, n_steps) # The "Inhibitor" schedule
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample(self, shape):
        """
        Returns the full evolution history for RD comparison.
        """
        x = torch.randn(shape) # Start with Chaos
        history = []
        
        with torch.no_grad():
            for t in reversed(range(self.n_steps)):
                # The Network acts as the ACTIVATOR (Structure formation)
                predicted_noise = self.network(x, t)
                
                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]
                beta = self.betas[t]
                
                # Standard DDPM sampling math
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + torch.sqrt(beta) * noise
                
                # Store for analysis
                history.append(x.cpu().clone())
                
        return torch.stack(history) # Shape: [Time, C, H, W]