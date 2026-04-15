import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LifeDataset(Dataset):
    def __init__(self, grids):
        # grids shape: (N, 64, 64)
        self.data = torch.from_numpy(grids).float().unsqueeze(1) # Add channel dim: (N, 1, 64, 64)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class VAE(nn.Module):
    def __init__(self, input_dim=64, latent_dim=128):
        super(VAE, self).__init__()
        
        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent vectors (Mean and Log-Variance)
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # DECODER
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> 64x64
            nn.Sigmoid() # Output probabilities between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=1.0, pos_weight=10.0):
    # GoL patterns are overwhelmingly dead cells (~2-5% alive). Unweighted BCE
    # converges to "predict all zeros" which is globally optimal for the prior
    # distribution but useless for generation. pos_weight upweights the alive
    # class so the model actually has to reconstruct live cells.
    eps = 1e-7
    recon = torch.clamp(recon_x, eps, 1.0 - eps)
    bce_elems = -(pos_weight * x * torch.log(recon) + (1 - x) * torch.log(1 - recon))
    BCE = bce_elems.mean()

    # KL per-latent-dim, averaged over batch. Same scale as BCE so beta is a
    # meaningful knob instead of being swamped by sum-reduced BCE.
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    KLD = KLD / (64 * 64)

    return BCE + beta * KLD

class ModelTrainer:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = VAE().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, grids, epochs=50, batch_size=32):
        if len(grids) == 0:
            print("No data to train on.")
            return
            
        dataset = LifeDataset(grids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        print(f"Starting training on {len(grids)} patterns for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(batch)
                loss = loss_function(recon_batch, batch, mu, logvar)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataset.data):.2f}")

    def generate_seeds(self, num_seeds=10):
        self.model.eval()
        with torch.no_grad():
            # Sample from the latent space (Normal distribution)
            z = torch.randn(num_seeds, 128).to(self.device)
            generated = self.model.decode(z)
            # Threshold to binary
            grids = (generated.cpu().numpy() > 0.5).astype(np.float32)
            # Squeeze channel dim
            return grids.squeeze(1)
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()