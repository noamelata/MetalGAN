
import torch

dataroot = r"C:\Users\Noam\PycharmProjects\metalcrawler\MLDS"
workers = 5

batch_size = 32
image_size = 64
nc = 1
nz = 128

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

num_epochs = 10
lr_pretrain = 0.0003
lr = 0.0002
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
