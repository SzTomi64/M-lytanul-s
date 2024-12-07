from torchvision import transforms
import torch
import random


class DataWithNoise(torch.utils.data.Dataset):
    def __init__(self, dataset, timesteps, transform = None):
        self.dataset = dataset
        self.transform = transform
        self.timesteps = timesteps
        self.beta_schedule = torch.linspace(1e-4, 2e-2, timesteps)
        #self.alpha_schedule = torch.cumprod(1-self.beta_schedule, dim=0)
        self.alpha_schedule = torch.cos(torch.linspace(0, 3.1415/2, timesteps))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        image = self.transform(image)
        
        noise = torch.randn_like(image)
        t = random.randint(0, self.timesteps-1)
        noisy_image = torch.sqrt(self.alpha_schedule[t])*image+(1-self.alpha_schedule[t])*noise
        return noisy_image, noise, torch.Tensor([t])