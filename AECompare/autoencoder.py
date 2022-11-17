import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class VariationalAutoEncoder(nn.Module):
        def __init__(self, latent_len, digit):
            super(VariationalAutoEncoder, self).__init__()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.digit = digit
            self.encoder = nn.Sequential(
                # 28 x 28
                nn.Conv2d(1, 4, kernel_size=5),
                # 4 x 24 x 24
                nn.ReLU(True),
                nn.Conv2d(4, 8, kernel_size=5),
                nn.ReLU(True),
                # 8 x 20 x 20 = 3200
                nn.Flatten(),
                nn.Linear(3200, 400),
                # 400
                nn.ReLU(True),
                # 128
                nn.Linear(400, 128),
                nn.ReLU(True),
                )

            self.linear1 = nn.Linear(128, latent_len)
            self.linear2 = nn.Linear(128, latent_len)

            self.decoder = nn.Sequential(
                # 2
                nn.Linear(latent_len, 400),
                # 400
                nn.ReLU(True),
                nn.Linear(400, 4000),
                # 4000
                nn.ReLU(True),
                nn.Unflatten(1, (10, 20, 20)),
                # 10 x 20 x 20
                nn.ConvTranspose2d(10, 10, kernel_size=5),
                # 24 x 24
                nn.ConvTranspose2d(10, 1, kernel_size=5),
                # 28 x 28
                nn.Sigmoid(),
                )
            self.kl_loss = 0
        def sample(self, mu, sigma):
            """
            :param mu: mean from the encoder's latent space
            :param log_var: log variance from the encoder's latent space
            """
            eps = torch.randn_like(sigma)
            sample = mu + sigma * eps
            return sample

        def forward(self, x):
            x = self.encoder(x)
            mu = self.linear1(x)
            sigma = torch.exp(self.linear2(x))
            
            z = self.sample(mu, sigma)
            self.encoded_sample = z
            self.kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

            dec = self.decoder(z)
            return dec

        ####################### Training the model ##########################
        def train_loop(self, dataloader, optimizer, device):
            self.train()
            running_loss = 0.0
            for data in dataloader:
                data, _ = data
                data = data.to(device)
                reconstruction = self.cuda().forward(data)
                #mse_loss = criterion(reconstruction, data)
                loss = ((data - reconstruction)**2).sum() + self.kl_loss
                optimizer.zero_grad()
                #loss = final_loss(mse_loss, mu, sigma)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                #orig.append(data)
                #recons.append(reconstruction)
                
            train_loss = running_loss/len(dataloader.dataset)
            return train_loss

        def test(self, test_dataset):
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False
            )
            self.eval()
            running_loss = 0.0
            latent_spaces = []
            with torch.no_grad():
                for data in test_loader:
                    data, _ = data
                    data = data.to(self.device)
                    reconstruction = self.forward(data)
                    latent_space = self.encoded_sample
                    latent_spaces.append(np.array(latent_space[0]))
                    loss = ((data - reconstruction)**2).sum() + self.kl_loss
                    running_loss += loss.item()
            val_loss = running_loss/len(test_loader.dataset)
            return val_loss, latent_spaces

        def fit(self, train_dataset, num_epochs, lr, batch_size=32):
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            optimizer = optim.Adam(self.parameters(), lr=lr)
            train_loss = []
            for epoch in range(num_epochs):
                train_epoch_loss = self.train_loop(train_loader, optimizer, self.device)
                train_loss.append(train_epoch_loss)
                print(f'Epoch {epoch+1}, train loss: {train_epoch_loss}')
                #writer.add_scalar('Loss/train', train_epoch_loss, epoch)
            torch.save(self.state_dict(), f'AECompare/MNIST_digits_models/{self.digit}_model_2.pth')

        def evaluate(self, test_data, n=10):
            plt.figure(figsize=(16,4.5))
            for i in range(n):
                ax = plt.subplot(2,n, 1+ i)
                img = test_data[i][0].unsqueeze(0).to(self.device)
                self.eval()
                with torch.no_grad():
                    rec_img  = self.forward(img)
                plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == n//2:
                    ax.set_title('Original images')
                ax = plt.subplot(2, n, i + 1 + n)
                plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)  
                if i == n//2:
                    ax.set_title('Reconstructed images')
            plt.show()

        def store_latent(self, latent_spaces):
            import csv
            file = open(f'AECompare/MNIST_digits_latents/{self.digit}_model_2.csv', 'w+', newline='')
            with file:
                write = csv.writer(file)
                write.writerows(latent_spaces)




