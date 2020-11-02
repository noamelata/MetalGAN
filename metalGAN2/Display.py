from __future__ import print_function
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def plot_graph(G_losses, D_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator1 and Discriminator1 Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_img_progress(img_list):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())


def compare_with_real(device, dataloader, img_list=None, Generator=None, noise=None):
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    if img_list:
        images = img_list[-1]
    else:
        fake = Generator(noise).detach().cpu()
        images = vutils.make_grid(fake, padding=2, normalize=True)
    plt.imshow(np.transpose(images, (1, 2, 0)))
    plt.show()

