import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from bwai.utils.trainer.vae_trainer import VAE_Trainer
from bwai.utils.vae import VAE, VAE_cluster, VQ_VAE


def vae_main():
    batch_size = 128
    img_size, color_channels = 32, 1
    filter_dim = 16
    z_dim = 8
    lr = 1e-4
    latent_dim = 8

    kl_use_bn = False

    fp16 = False
    load = False
    load = True

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
    vae = VAE(
        img_size,
        color_channels,
        filter_dim,
        z_dim,
        latent_dim=latent_dim,
        min_size=3,
        kl_use_bn=kl_use_bn,
    )
    optimer = torch.optim.Adam(vae.parameters(), lr=lr)

    def save_step():
        n = 5
        fig_digit = torch.zeros((color_channels, img_size * n, img_size * n))
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
    if load:
        trainer.load_model(save_dir)

    save_step()
    trainer.train(
        epochs=200000,
        save_dir=save_dir,
        save_iter=300,
        show_bar=True,
        show_hz=10,
        leave_bar=False,
    )
    save_step()


def vae_cluster_main():
    batch_size = 64
    img_size, color_channels = 64, 3
    filter_dim = 64
    z_dim = 256
    lr = 1e-5
    latent_dim = 128
    num_classes = 64
    rows, cols = 8, 8

    fp16 = False
    load = False
    load = True

    save_dir = "./models"

    data_dir = "./data"
    train_dataset = datasets.ImageFolder(
        data_dir + "/img_align_celeba",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
                transforms.CenterCrop(img_size),
            ]
        ),
    )
    # train_dataset = datasets.FashionMNIST(
    #     data_dir,
    #     train=True,
    #     transform=transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Resize(img_size, antialias=True),
    #             transforms.CenterCrop(img_size),
    #         ]
    #     ),
    # )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    device = torch.device("cuda")
    vae = VAE_cluster(
        img_size,
        color_channels,
        filter_dim,
        z_dim,
        num_classes=num_classes,
        latent_dim=latent_dim,
        min_size=3,
    )
    optimer = torch.optim.Adam(vae.parameters(), lr=lr)

    def save_step():
        n = 2
        figure = plt.figure(figsize=(rows * n * 2, cols * n * 2))
        vae.eval()
        for index in range(num_classes):
            fig_digit = torch.zeros((color_channels, img_size * n, img_size * n))
            result = vae.predict(n**2, device, category=index).cpu()
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

            figure.add_subplot(rows, cols, index + 1)
            plt.axis("off")
            plt.imshow(fig_digit)
            plt.title(f"cluster {index}")
        plt.subplots_adjust(
            wspace=0.1, hspace=0.1, top=0.9, bottom=0.1, left=0.1, right=0.9
        )
        plt.savefig("./test.png")
        plt.close()
        vae.train()

    trainer = VAE_Trainer(
        vae,
        optimer,
        fp16=fp16,
        device=device,
        train_loader=train_loader,
        save_step_func=save_step,
    )

    if load:
        trainer.load_model(save_dir)

    save_step()
    trainer.train(
        epochs=5000,
        save_dir=save_dir,
        save_iter=2000,
        show_bar=True,
        show_hz=10,
        leave_bar=False,
    )
    save_step()



def vq_vae_main():
    batch_size = 32
    img_size, color_channels = 256, 3
    z_dim = 64
    codebook_size = 16
    num_heads = None
    lr = 1e-2

    n = 2
    fp16 = False
    load = False
    # load = True

    save_dir = "./models"

    data_dir = "./data"
    train_dataset = datasets.ImageFolder(
        data_dir + "/img_align_celeba",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
                transforms.CenterCrop(img_size),
            ]
        ),
    )
    # train_dataset = datasets.FashionMNIST(
    #     data_dir,
    #     train=True,
    #     transform=transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Resize(img_size, antialias=True),
    #             transforms.CenterCrop(img_size),
    #         ]
    #     ),
    # )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        train_dataset,
        batch_size=n*n,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ).__iter__()

    device = torch.device("cuda")
    vae = VQ_VAE(
        img_size,
        color_channels,
        z_dim,
        codebook_size,
        num_heads
    )
    optimer = torch.optim.Adam(vae.parameters(), lr=lr)

    def save_step():
        plt.figure(figsize=(5 * n, 5 * n))
        vae.eval()
        fig_digit = torch.zeros((color_channels, img_size * n, img_size * n))
        imgs, _ = next(test_loader)
        imgs = imgs.to(device)
        result = vae(imgs)[0].cpu()
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

        plt.imshow(fig_digit)
        plt.savefig("./test.png")
        plt.close()
        vae.train()

    trainer = VAE_Trainer(
        vae,
        optimer,
        fp16=fp16,
        device=device,
        train_loader=train_loader,
        save_step_func=save_step,
    )

    if load:
        trainer.load_model(save_dir)

    save_step()
    trainer.train(
        epochs=5000,
        save_dir=save_dir,
        save_iter=500,
        show_bar=True,
        show_hz=10,
        leave_bar=False,
    )
    save_step()


if __name__ == '__main__':
    # vae_main()
    # vae_cluster_main()
    vq_vae_main()
