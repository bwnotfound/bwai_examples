import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from bwai.utils.trainer.gan_trainer import WGAN_Trainer as WTrainer
from bwai.utils.generator import G_simple as GNet
from bwai.utils.discriminator import D_simple as DNet

if __name__ == '__main__':
    batch_size = 32
    img_size, color_channels = 32, 3
    z_dim = 128
    g_filter_dim = 32
    d_filter_dim = 32
    g_lr, d_lr = 1e-4, 3e-4
    betas = (0, 0.9)
    
    fp16 = False
    
    data_dir = "./data/test_images"
    save_dir = "./models"

    train_dataset = datasets.ImageFolder(
        data_dir,
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
    g_net = GNet(img_size, z_dim, g_filter_dim, color_channels)
    d_net = DNet(img_size, color_channels, d_filter_dim, min_size=3)
    g_optimer = torch.optim.Adam(g_net.parameters(), lr=g_lr, betas=betas)
    d_optimer = torch.optim.Adam(d_net.parameters(), lr=d_lr, betas=betas)

    def save_step():
        n = 5
        fig_digit = torch.zeros(
            (color_channels, img_size * n, img_size * n)
        )
        g_net.eval()
        result = g_net(torch.randn(n**2, g_net.z_dim, device=device)).cpu()
        g_net.train()
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
        

    trainer = WTrainer(
        g_net=g_net,
        d_net=d_net,
        g_optimer=g_optimer,
        d_optimer=d_optimer,
        device=device,
        train_loader=train_loader,
        clip_value=0.01,
        critic_iter=5,
        use_gp=True,
        save_step_func=save_step,
        fp16=fp16
    )
    # trainer.load_model("./models")
    g_net_param = sum(
        [
            param.nelement()
            for param in filter(lambda p: p.requires_grad, g_net.parameters())
        ]
    )
    d_net_param = sum(
        [
            param.nelement()
            for param in filter(lambda p: p.requires_grad, d_net.parameters())
        ]
    )
    print(
        f"G_param: {g_net_param}, D_param: {d_net_param}, G/D: {g_net_param/d_net_param}"
    )
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
