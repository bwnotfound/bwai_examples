import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from bwai.utils.trainer.vae_trainer import VAE_Trainer
from bwai.utils.vae import VAE


if __name__ == '__main__':
    batch_size = 32
    img_size, color_channels = 32, 1
    filter_dim = 8
    z_dim = 32
    lr = 1e-4

    fp16 = False

    save_dir = "./models"

    data_dir = "./data"
    # train_dataset = datasets.ImageFolder(
    #     data_dir + "/img_align_celeba",
    #     transform=transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Resize(img_size, antialias=True),
    #             transforms.CenterCrop(img_size),
    #         ]
    #     ),
    # )
    train_dataset = datasets.MNIST(
        data_dir,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
                transforms.CenterCrop(img_size),
            ]
        ),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    
    

    device = torch.device("cuda")
    vae = VAE(img_size, color_channels, filter_dim, z_dim, 2, min_size=3)
    optimer = torch.optim.Adam(vae.parameters(), lr=lr)
    
    def save_step():
        n = 5
        fig_digit = torch.zeros(
            (color_channels, img_size * n, img_size * n)
        )
        vae.eval()
        result = vae.predict(n**2, device).cpu()
        vae.train()
        for i in range(n):
            for j in range(n):
                fig_digit[
                    :,
                    i * img_size : (i + 1) * img_size,
                    j * img_size : (j + 1) * img_size,
                ] = result[i * n + j]
        fig_digit = fig_digit.permute(1, 2, 0)
        if color_channels == 1:
            fig_digit = fig_digit.repeat(1, 1, 3)
        fig_digit = fig_digit.detach().numpy().astype(np.float32)
        plt.figure(figsize=(10, 10))
        plt.imshow(fig_digit)
        plt.savefig("./test.png")
        plt.close()
        
    trainer = VAE_Trainer(
        vae,
        optimer,
        fp16=fp16,
        device=device,
        train_loader=train_loader,
        save_step_func=save_step,
    )
        
    trainer.load_model(save_dir)
        
    save_step()
    trainer.train(
        epochs=200000,
        save_dir=save_dir,
        save_iter=100,
        show_bar=True,
        show_hz=10,
        leave_bar=False,
    )
    save_step()