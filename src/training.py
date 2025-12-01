import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def train_lstm(model, data, epochs=50, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print(f"--- Training LSTM on {device}---")
    model.train()
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()

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

        if len(losses)>2 :
            if losses[-2] - loss < 1e-6:
                print(f"Loss has converged enough with n_epochs of {epoch}")
                break
        
    return losses


def train_ddpm(model, data, epochs=50, device="cpu"):
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print(f"--- Training DDPM on {device}---")
    model.train()

    losses = []
    
    for ep in range(epochs):
        pbar = torch.optim.lr_scheduler
        loss_ema = None
        
        for x, _ in data:
            optim.zero_grad()
            x = x.to(device)
            
            noise_pred, noise = model(x)
            
            loss = criterion(noise_pred, noise)

            loss.backward()
            optim.step()
            
            if loss_ema is None: 
                loss_ema = loss.item()
            else: 
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
        
        losses.append(loss_ema)
        print(f"Epoch {ep:02d} | Loss: {loss_ema:.4f}")

        # --- Visualization for sanity check ---
        if ep % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                x_gen, _ = model.sample(16, (1, 28, 28), device)
                grid = make_grid(x_gen * -1 + 1, nrow=4) # Invert colors for visibility
                plt.figure(figsize=(4,4))
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                plt.title(f"Generated at Epoch {ep}")
                plt.show()

    return losses

def train_ddpm_temp(ddpm_model, data, epochs=50):
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