import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Critic, Generator, initialize_weights
from utils import gradient_penalty

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 1e-4
batch_size = 64
image_size = 64
channels_img = 3
z_dim = 100
num_epochs = 5
features_d = 64
features_g = 64
critic_n = 5
lambda_gp = 10

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(channels_img)], 
        [0.5 for _ in range(channels_img)], 
    )
])


critic = Critic(channels_img, features_d).to(device)
gen = Generator(z_dim, channels_img, features_g)
initialize_weights(critic)
initialize_weights(gen)
fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

dataset = datasets.ImageFolder(root='../data/celeb_dataset', transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))

# TensorBoard
writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0

# Training
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1).to(device)
        
        # For Critic
        for _ in range(critic_n):
            noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real)
            critic_fake = critic(fake)
            gp = gradient_penalty(critic, real, fake, device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)
                            + lambda_gp * gp)
            
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
        
        # For Generator
        output = critic(fake).view(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image(
                    "Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Real Images", img_grid_real, global_step=step
                )
                step += 1