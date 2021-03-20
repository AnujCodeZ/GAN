import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator, Generator, initialize_weights


# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 2e-4
batch_size = 128
image_size = 64
channels_img = 3
z_dim = 100
num_epochs = 5
features_d = 64
features_g = 64

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(channels_img)], 
        [0.5 for _ in range(channels_img)], 
    )
])


disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim)
initialize_weights(disc)
initialize_weights(gen)
fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

dataset = datasets.ImageFolder(root='../data/celeb_dataset', transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# TensorBoard
writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0

# Training
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1).to(device)
        batch_size = real.shape[0]
        
        # For Discriminator
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_fake + lossD_real) / 2
        
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()
        
        # For Generator
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1