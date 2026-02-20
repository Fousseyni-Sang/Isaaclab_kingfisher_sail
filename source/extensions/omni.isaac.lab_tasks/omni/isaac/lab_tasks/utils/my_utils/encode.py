import torch
import torch.nn as nn
import torch.optim as optim

class FeasibilityAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # (H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # (H/8, W/8)
            nn.ReLU(),
            nn.Flatten(),
        )

        # compute flattened size dynamically later
        self._latent_dim = latent_dim
        self.fc_mu = None
        self.fc_decode = None

        # ----- Decoder -----
        # we will initialize decoder layers after first forward pass
        self.decoder = None

    def _init_decoder(self, encoded_shape):
        """Initialize decoder layers once we know the encoded shape."""
        flat_dim = encoded_shape[1]
        self.fc_decode = nn.Linear(self._latent_dim, flat_dim)

        C, H, W = encoded_shape[2:]

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (C, H, W)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x):
        # First pass: initialize decoder dynamically
        if self.fc_mu is None:
            with torch.no_grad():
                enc = self.encoder(x)
            flat_dim = enc.shape[1]
            self.fc_mu = nn.Linear(flat_dim, self._latent_dim)
            self._init_decoder(enc.shape)

        # Encode
        enc = self.encoder(x)
        z = self.fc_mu(enc)

        # Decode
        dec = self.fc_decode(z)
        out = self.decoder(dec)

        return out, z

def train_autoencoder(model, dataloader, epochs=20, lr=1e-3, device="cuda"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # binary maps → BCE works well

    for epoch in range(epochs):
        total_loss = 0.0
        for feas_map in dataloader:
            feas_map = feas_map.to(device).float()

            optimizer.zero_grad()
            recon, _ = model(feas_map)
            loss = criterion(recon, feas_map)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(dataloader):.4f}")

from torch.utils.data import Dataset, DataLoader

class FeasibilityDataset(Dataset):
    def __init__(self, maps):
        self.maps = maps  # shape (N, H, W) or (N, 1, H, W)

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        x = self.maps[idx]
        if x.ndim == 2:
            x = x.unsqueeze(0)  # add channel
        return x

# Example:
dataset = FeasibilityDataset(feasibility_maps_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model.eval()
with torch.no_grad():
    _, latent_z = model(feas_map.unsqueeze(0).to(device))
