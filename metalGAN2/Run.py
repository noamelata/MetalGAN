
from __future__ import print_function
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from MoreTransforms import *
from Params import *
from Networks import *
from Weights import *
from Checkpoint import *
from Trainer import *
from Display import *


def prep_data():
    dataset64 = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.RandomAffine(degrees=0, translate=(0.0625,0.0625),
                                                scale=None, shear=None, resample=False, fillcolor=0),
                                   RandomBrightness([0.9, 1.0, 1.1, 1.2, 1.3]),
                                   RandomContrast([0.9, 1.0, 1.1, 1.2, 1.3]),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                               ]))

    dataloader64 = torch.utils.data.DataLoader(dataset64, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    dataset128 = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size*2),
                                   transforms.CenterCrop(image_size*2),
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.RandomAffine(degrees=0, translate=(0.0625,0.0625),
                                                scale=None, shear=None, resample=False, fillcolor=0),
                                   RandomBrightness([0.9, 1.0, 1.1, 1.2, 1.3]),
                                   RandomContrast([0.9, 1.0, 1.1, 1.2, 1.3]),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                               ]))

    dataloader128 = torch.utils.data.DataLoader(dataset128, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    return dataloader64, dataloader128


def create_nn():
    netG1 = Generator_Stage_1(ngpu).to(device)
    netG2 = Generator_Stage_2(ngpu).to(device)
    netD1 = Discriminator_Stage_1(ngpu).to(device)
    netD2 = Discriminator_Stage_2(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netG1 = nn.DataParallel(netG1, list(range(ngpu)))
        netG2 = nn.DataParallel(netG2, list(range(ngpu)))
        netD1 = nn.DataParallel(netD1, list(range(ngpu)))
        netD2 = nn.DataParallel(netD2, list(range(ngpu)))

    netG1.apply(weights_init)
    netG2.apply(weights_init)
    netD1.apply(weights_init)
    netD2.apply(weights_init)

    return netG1, netG2, netD1, netD2


def create_optimizers(netG1, netG2, netD1, netD2):
    optimizerD1 = optim.Adam(netD1.parameters(), lr=lr_pretrain, betas=(beta1, 0.999))
    optimizerG1 = optim.Adam(netG1.parameters(), lr=lr_pretrain, betas=(beta1, 0.999))
    optimizerD2 = optim.Adam(netD2.parameters(), lr=lr_pretrain, betas=(beta1, 0.999))
    optimizerG2 = optim.Adam(netG2.parameters(), lr=lr_pretrain, betas=(beta1, 0.999))

    return optimizerG1, optimizerG2, optimizerD1, optimizerD2


def train_part_1(args_bundle, lists, checkpoint, fixed_noise, start_at_epoch=0):
    dataloader64, dataloader128, \
    netG1, netG2, netD1, netD2, \
    optimizerG1, optimizerG2, optimizerD1, optimizerD2 \
        = args_bundle
    img_list_64, img_list_128, D1_losses, G1_losses, G2_losses, D2_losses = lists
    iters = 0

    print("Starting Training Loop...")

    for epoch in range(start_at_epoch, num_epochs):
        for i, data64 in enumerate(dataloader64, 0):
            errD, errG, D_x, D_G_z1, D_G_z2 = train_stage_1(device, netD1, netG1, optimizerD1, optimizerG1, i, data64)

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader64),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G1_losses.append(errG.item())
            D1_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader64) - 1)):
                with torch.no_grad():
                    fake = netG1(fixed_noise).detach().cpu()
                img_list_64.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        checkpoint.save()
        checkpoint.epoch += 1



def train_part_2(args_bundle, lists, checkpoint, fixed_noise, offset, start_at_epoch=10):
    dataloader64, dataloader128, \
    netG1, netG2, netD1, netD2, \
    optimizerG1, optimizerG2, optimizerD1, optimizerD2 \
        = args_bundle

    img_list_64, img_list_128, D1_losses, G1_losses, G2_losses, D2_losses = lists

    iters = 0

    print("Starting Training Loop...")

    for epoch in range(start_at_epoch, offset + num_epochs):
        # For each batch in the dataloader
        for i, (data64, data128) in enumerate(zip(dataloader64, dataloader128)):
            errD1, errG1, D1_x, D1_G1_z1, D1_G1_z2 = train_stage_1(device, netD1, netG1,
                                                                   optimizerD1, optimizerG1, i, data64)
            errD2, errG2, D2_x, D2_G2_z1, D2_G2_z2 = train_stage_2(device, netD2, netG2,
                                                                   optimizerD2, optimizerG2, i, data128, netG1)

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader64),
                         errD2.item(), errG2.item(), D2_x, D2_G2_z1, D2_G2_z2))

            # Save Losses for plotting later
            G1_losses.append(errG1.item())
            D1_losses.append(errD1.item())
            G2_losses.append(errG2.item())
            D2_losses.append(errD2.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader64) - 1)):
                with torch.no_grad():
                    fake64 = netG1(fixed_noise).detach().cpu()
                    fake128 = netG2(fake64).detach().cpu()
                img_list_64.append(vutils.make_grid(fake64, padding=2, normalize=True))
                img_list_128.append(vutils.make_grid(fake128, padding=2, normalize=True))

            iters += 1
        checkpoint.save()
        checkpoint.epoch += 1


def show_off(lists, dataloader128):
    img_list_64, img_list_128, D1_losses, G1_losses, G2_losses, D2_losses = lists
    plot_graph(G1_losses, D1_losses)
    plot_graph(G2_losses, D2_losses)
    show_img_progress(img_list_64)
    show_img_progress(img_list_128)
    compare_with_real(device, dataloader128, img_list_128)

def execute(load=False, train_from_start=False):
    dataloader64, dataloader128 = prep_data()
    netG1, netG2, netD1, netD2 = create_nn()
    optimizerG1, optimizerG2, optimizerD1, optimizerD2 = \
        create_optimizers(netG1, netG2, netD1, netD2)

    img_list_64 = []
    img_list_128 = []
    G1_losses = []
    D1_losses = []
    G2_losses = []
    D2_losses = []
    lists = (img_list_64, img_list_128, D1_losses, G1_losses, G2_losses, D2_losses)

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    checkpoint = Checkpoint(
        {'Generator_Stage_1': netG1,
         'Generator_Stage_2': netG2,
         'Discriminator_Stage_1': netD1,
         'Discriminator_Stage_2': netD2
         },
        {'optimizerG1': optimizerG1,
         'optimizerG2': optimizerG2,
         'optimizerD1': optimizerD1,
         'optimizerD2': optimizerD2
         },
        epoch=0,
        loss=criterion,
        script_name="metalGAN2",
        load=load)

    args_bundle = dataloader64, dataloader128, netG1, netG2, netD1, netD2, \
                 optimizerG1, optimizerG2, optimizerD1, optimizerD2

    if train_from_start:
        checkpoint.epoch = 0
    start_at_epoch = checkpoint.epoch

    train_part_1(args_bundle, lists, checkpoint, fixed_noise, start_at_epoch)
    print("starting stage 2...")
    adjust_learning_rate(optimizerD1, lr)
    adjust_learning_rate(optimizerG1, lr)

    train_part_2(args_bundle, lists, checkpoint, fixed_noise, num_epochs, max(10, start_at_epoch))

    adjust_learning_rate(optimizerD1, lr / 2)
    adjust_learning_rate(optimizerG1, lr / 2)
    adjust_learning_rate(optimizerD2, lr)
    adjust_learning_rate(optimizerG2, lr)

    train_part_2(args_bundle, lists, checkpoint, fixed_noise, num_epochs*2, max(20, start_at_epoch))

    checkpoint.save()
    show_off(lists, dataloader128)
    return 0

def load_and_compare():
    dataloader64, dataloader128 = prep_data()
    netG1, netG2, netD1, netD2 = create_nn()
    optimizerG1, optimizerG2, optimizerD1, optimizerD2 = \
        create_optimizers(netG1, netG2, netD1, netD2)

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    checkpoint = Checkpoint(
        {'Generator_Stage_1': netG1,
         'Generator_Stage_2': netG2,
         'Discriminator_Stage_1': netD1,
         'Discriminator_Stage_2': netD2
         },
        {'optimizerG1': optimizerG1,
         'optimizerG2': optimizerG2,
         'optimizerD1': optimizerD1,
         'optimizerD2': optimizerD2
         },
        epoch=0,
        loss=criterion,
        script_name="metalGAN2",
        load=True)

    compare_with_real(device, dataloader128, img_list=None, Generator=netG1, noise=fixed_noise)

if __name__ == "__main__":
    #execute(load=True)
    load_and_compare()


