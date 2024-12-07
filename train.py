import torch
from tqdm import tqdm


def train_loop(num_epoch, model, optimizer, scheduler, train_loader, criterion, device, batch_size, model_path):
    for epoch in range(num_epoch):
        tepoch = tqdm(train_loader, leave=True, position=0)
        sum_loss = 0
        i = 0
        for noisy_image, noise, t in tepoch:
            i+=1
            optimizer.zero_grad()

            noisy_image = noisy_image.to(device)
            noise = noise.to(device)
            t = t.to(device)

            output = model(noisy_image, t)
            loss = criterion(noise, output)
            loss.backward()

            optimizer.step()

            sum_loss += loss.item()
            running_loss = sum_loss / (i*batch_size)

            tepoch.set_postfix(epoch = epoch, loss=running_loss)
            tepoch.refresh()

            scheduler.step()

    torch.save(model.state_dict(), model_path)


def batch_train_loop(num_epoch, model, optimizer, scheduler, batch, criterion):
    tepoch = tqdm(range(num_epoch), position=0)
    for epoch in tepoch:
        noisy_image, noise, t = batch
        optimizer.zero_grad()

        noisy_image = noisy_image.to(device)
        noise = noise.to(device)
        t = t.to(device)

        output = model(noisy_image, t)

        loss = criterion(noise, output)
        loss.backward()

        optimizer.step()

        tepoch.set_postfix(epoch = epoch, loss=loss.item())
        tepoch.refresh()

        scheduler.step()