import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from timm.utils import ModelEmaV3
from tqdm import tqdm
import torch.nn.functional as F
import os

from ddpm import set_seed, DDPM_Scheduler


def train_lstm(model, data, num_epochs=50, lr=1e-4, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"--- Training LSTM on {device} ---")
    model.train()
    
    losses = []
    
    for epoch in range(num_epochs):
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


def train_ddpm(model,
          data,
          batch_size: int=64,
          num_time_steps: int=1000,
          num_epochs: int=15,
          seed: int=-1,
          ema_decay: float=0.9999,  
          lr=2e-5,
          checkpoint_path: str=None,
          dataset_size: int=None,
          device: str="cpu"):
    
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    train_set = data
    if dataset_size is not None:
        indices = list(range(len(data)))
        random.shuffle(indices)
        subset_indices = indices[:dataset_size]
        train_set = Subset(data, subset_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    unet_model = model.to(device)

    optimizer = optim.Adam(unet_model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')
    
    print(f"--- Training DDPM on {device} ---")
    losses = []

    ema = ModelEmaV3(unet_model, decay=ema_decay)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        unet_model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,_) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            x = x.to(device)
            x = F.pad(x, (2,2,2,2))
            t = torch.randint(0,num_time_steps,(batch_size,))
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size,1,1,1).to(device)
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            output = unet_model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(unet_model)
        print(f'Epoch {i+1} | Loss {total_loss / ((60000 if dataset_size is None else dataset_size)/batch_size):.5f}')
        losses.append(loss.item())

    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = {
        'weights': unet_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    torch.save(checkpoint, 'checkpoints/ddpm_checkpoint')

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