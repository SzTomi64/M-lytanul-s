import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def generate_image(model, x_t, device, T = 300):
    beta = torch.linspace(1e-4, 2e-2, T).to(device)
    alpha = (1-beta).to(device)
    alpha_cumprod = torch.cumprod(alpha, dim=0).to(device)
    #alpha_cumprod = torch.cos(torch.linspace(0, 3.1415/2, T))

    for t in reversed(range(T)):
        x_t = timestep(model, x_t, t, beta, alpha, alpha_cumprod, device)
    return x_t


@torch.no_grad()
def timestep(model, x_t, t, beta, alpha, alpha_cumrpod, device):
    if t==0:
        z = torch.zeros_like(x_t).to(device)
    else:
        z = torch.randn_like(x_t).to(device)
    sigma_t = beta[t].to(device)
    model_coeff = ((1-alpha[t])/torch.sqrt(1-alpha_cumrpod[t])).to(device)
    main_coeff = (1/torch.sqrt(alpha[t])).to(device)
    noise = model(x_t, torch.tensor([t]).to(device))
    #print(model_coeff)
    #print(main_coeff)
    return main_coeff*(x_t-model_coeff*noise)+sigma_t*z

@torch.no_grad()
def normalize_images(images):
    normalized_images = []
    for img in images:
        normalized_channels = []
        for channel in img:
            channel_min = channel.min()
            channel_max = channel.max()
            normalized_channel = (channel - channel_min) / (channel_max - channel_min + 1e-5)  # Add small epsilon to avoid division by zero
            normalized_channels.append(normalized_channel)
        normalized_images.append(torch.stack(normalized_channels))
    return torch.stack(normalized_images)

@torch.no_grad()
def plot_images(images, n_rows, n_cols):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            # Convert image from (C, H, W) to (H, W, C) and display
            img = images[i].permute(1, 2, 0).detach().cpu().numpy()
            ax.imshow(img)
            ax.axis('off')
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def generate_show_images(model, device, image_size):
    model = model.to(device)
    x_T = torch.randn([24, 3, image_size, image_size]).to(device)
    x = generate_image(model, x_T, device)
    y = normalize_images(x)
    plot_images(y, 4, 6)

    #y = (y.cpu().detach())[0].permute(1, 2, 0)
    #plt.imshow(y)