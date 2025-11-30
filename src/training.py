import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from ddpm import DDPM, ContextUnet

def train_lstm(model, data, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("--- Training LSTM ---")
    model.train()
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        #print(data[0, :, :].size())
        predictions, _ = model(data)
        
        # Target: We want the model to predict the NEXT step.
        # Input at t should predict Data at t+1.
        # We crop the last prediction and the first data point to align them.
        preds_shifted = predictions[:, :-1, :]
        targets_shifted = data[:, 1:, :]
        
        loss = criterion(preds_shifted, targets_shifted)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.5f}")
            
    return losses

def train_ddpm(ddpm_model, data, epochs=50):
    optimizer = optim.Adam(ddpm_model.network.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print("\n--- Training DDPM ---")
    ddpm_model.train()
    
    losses = []
    
    for epoch in range(epochs):
        avg_loss = 0
        # Simple batch processing (treating whole dataset as one batch for simplicity here)
        x0 = data 
        n = len(x0)
        
        optimizer.zero_grad()
        
        # 1. Sample random timesteps for each image in batch
        t = torch.randint(0, ddpm_model.n_steps, (n,))
        
        # 2. Generate random noise (The "Inhibitor" we want to predict)
        epsilon = torch.randn_like(x0)
        
        # 3. Add noise to image (Forward Diffusion)
        # Formula: x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * epsilon
        a_bar = ddpm_model.alpha_bars[t].view(-1, 1, 1, 1)
        noisy_image = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * epsilon
        
        # 4. Model attempts to predict the noise
        noise_pred = ddpm_model.network(noisy_image, t)
        
        # 5. Loss: How close was the predicted noise to the actual noise?
        loss = criterion(noise_pred, epsilon)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss:.5f}")

    return losses

def train_ddpm_on_mnist():
    # --- Hyperparameters ---
    n_epoch = 20 # Enough for MNIST digits to appear
    batch_size = 128
    n_T = 400 # Timesteps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lrate = 1e-4

    # --- Data Loading ---
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # --- Setup Schedules (The "Inhibitor" Physics) ---
    beta_1 = 1e-4
    beta_T = 0.02
    betas = torch.linspace(beta_1, beta_T, n_T + 1).to(device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    # Pre-calculate standard DDPM constants to save compute
    ddpm_schedules = {
        "sqrtab": torch.sqrt(alphas_bar),
        "sqrtmab": torch.sqrt(1 - alphas_bar),
        "oneover_sqrta": 1 / torch.sqrt(alphas),
        "mab_over_sqrtmab": (1 - alphas) / torch.sqrt(1 - alphas_bar),
        "sqrt_beta_t": torch.sqrt(betas),
    }

    # --- Model Init ---
    model = ContextUnet(in_channels=1, n_feat=64).to(device)
    ddpm = DDPM(model, ddpm_schedules, n_T, device)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    # --- Training Loop ---
    print(f"Starts training on {device}...")
    
    for ep in range(n_epoch):
        ddpm.train()
        pbar = torch.optim.lr_scheduler
        loss_ema = None
        
        for x, _ in dataloader:
            optim.zero_grad()
            x = x.to(device)
            
            # DDPM Forward Pass
            noise_pred, noise = ddpm(x)
            
            # Loss: Activator Error
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optim.step()
            
            if loss_ema is None: loss_ema = loss.item()
            else: loss_ema = 0.95 * loss_ema + 0.05 * loss.item()

        print(f"Epoch {ep:02d} | Loss: {loss_ema:.4f}")
        
        # --- Visualization for sanity check ---
        if ep % 5 == 0 or ep == n_epoch - 1:
            ddpm.eval()
            with torch.no_grad():
                x_gen, _ = ddpm.sample(16, (1, 28, 28), device)
                grid = make_grid(x_gen * -1 + 1, nrow=4) # Invert colors for visibility
                plt.figure(figsize=(4,4))
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                plt.title(f"Generated at Epoch {ep}")
                plt.show()

    return ddpm