#!/usr/bin/env python
# encoding: utf-8

import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

import datasets as mydset

from PIL import Image
from os import listdir
from os.path import join
import random

from rdp_accountant import compute_rdp, get_privacy_spent


def get_dade_args(parser):
    parser.add_argument('--exp', choices=['svhn_mnist_32', 'mnist_svhn_32',
                                          'svhn_mnist_rgb', 'mnist_svhn_rgb',
                                          'cifar_stl', 'stl_cifar',
                                          'mnist_usps_16', 'usps_mnist_16',
                                          'mnist_usps_32', 'usps_mnist_32'
                                          ], default='usps_mnist', help='experiment to run')
    parser.add_argument('--dataset', choices=['usps', 'mnist',
                                              'svhn', 'syndigits',
                                              'cifar', 'stl',
                                              'gtsrb', 'synsigns',
                                              'cifar10', 'cifar100',
                                              'lsun', 'imagenet',
                                              'folder', 'lfw'
                                              ], default='mnist',  help='mnist | cifar10 | cifar100 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', type=str, default='data', help='path to args.dataset')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='learning rate (Adam)')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--epoch_size', choices=['large', 'small', 'target'], default='large',
                  help='epoch size is either that of the smallest dataset, the largest, or the target')
    parser.add_argument('--seed', type=int, default=0, help='random seed (0 for time-based)')
    parser.add_argument('--log_file', type=str, default='', help='log file path (none to disable)')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='number of channel')
    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--gpu_device', type=int, default=0, help='using gpu device id')
    parser.add_argument('--netd', default='', help="path to netD (to continue training)")
    parser.add_argument('--plot_interval', type=int, default=50, help='Interval for plotting images after n batches')
    parser.add_argument('--experiment', default=None, help='Folder to store samples and models')
    return parser


def get_args(parser):
    parser.add_argument('--dataset', required=True, help='mnist | cifar10 | cifar100 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=1, help='number of channel')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--max_iter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=0.00005')
    parser.add_argument('--gpu_device', type=int, default=0, help='using gpu device id')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--plot_interval', type=int, default=50, help='Interval for plotting images after G iters')
    return parser


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class FolderWithImages(data.Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        super(FolderWithImages, self).__init__()
        self.image_filenames = [join(root, x)
                                for x in listdir(root) if is_image_file(x.lower())]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


class ALICropAndScale(object):
    def __call__(self, img):
        return img.resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))


class ToRGB(object):
    def __call__(self, img):
        return img.convert('RGB')
        # a, _, _ = img.split()
        # return Image.merge("RGB", (a, a, a))


class RandomClampTensors(object):
    def __init__(self, min_margin=0, max_margin=0.2):
        self.max_margin = max_margin
        self.min_margin = min_margin

    def __call__(self, tnsr):
        margin = self.min_margin + (random.random() * (self.max_margin - self.min_margin))
        return torch.clamp(tnsr, -0.5+margin, 0.5-margin)


class Normalize_RandomInvert_pixels(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, nc=3):
        self.p = p
        self.nc = nc

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """

        if random.random() < self.p:
            if self.nc == 3:
                img = transforms.Normalize((0.5, 0.5, 0.5), (-0.5, -0.5, -0.5))(img)
            else:
                img = transforms.Normalize((0.5,), (-0.5,))(img)
        else:
            if self.nc == 3:
                img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
            else:
                img = transforms.Normalize((0.5,), (0.5,))(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def get_data(args, train_flag=True, transform=None):
    if train_flag:
        # p = 0.5  # Invert 50% only randomly
        p = 1.0  # Invert all
    else:
        p = -1.0  # disable random transformations for testset

    if not transform:
        # if args.nc == 1:
        #     transform = transforms.Compose([
        #         transforms.Resize(args.image_size),
        #         transforms.Grayscale(),
        #         transforms.CenterCrop(args.image_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ])
        # else:
        if args.dataset in {'usps', 'mnist'} and args.exp == 'mnist_svhn':
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=args.nc),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                Normalize_RandomInvert_pixels(p=p, nc=args.nc),
                RandomClampTensors(min_margin=0, max_margin=0.3),
            ])
        elif args.dataset in {'usps', 'mnist'} and args.exp == 'svhn_mnist':
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=args.nc),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                Normalize_RandomInvert_pixels(p=-1, nc=args.nc),
                # RandomClampTensors(min_margin=0, max_margin=0.0),
            ])
        elif args.dataset in {'usps', 'mnist'}:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=args.nc),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                Normalize_RandomInvert_pixels(p=-1, nc=args.nc),
            ])
        elif args.dataset in {'svhn', 'syndigits'} and args.exp == 'mnist_svhn':
            if args.nc == 1:
                transform = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.Grayscale(num_output_channels=args.nc),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    Normalize_RandomInvert_pixels(p=-1, nc=args.nc),
                    # RandomClampTensors(min_margin=0, max_margin=0.3),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=args.nc),
                    transforms.Resize(args.image_size),
                    transforms.CenterCrop(args.image_size),
                    # transforms.RandomResizedCrop(args.image_size),
                    transforms.ToTensor(),
                    Normalize_RandomInvert_pixels(p=-1, nc=args.nc),
                ])
        elif args.dataset in {'svhn', 'syndigits'} and args.exp == 'svhn_mnist':
            if args.nc == 1:
                transform = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.Grayscale(num_output_channels=args.nc),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    Normalize_RandomInvert_pixels(p=-1, nc=args.nc),
                    # RandomClampTensors(min_margin=0, max_margin=0.2),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=args.nc),
                    transforms.Resize(args.image_size),
                    transforms.CenterCrop(args.image_size),
                    # transforms.RandomResizedCrop(args.image_size),
                    transforms.ToTensor(),
                    Normalize_RandomInvert_pixels(p=-1, nc=args.nc),
                ])
        elif args.dataset in {'svhn', 'syndigits'}:
            if args.nc == 1:
                transform = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.Grayscale(num_output_channels=args.nc),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    Normalize_RandomInvert_pixels(p=-1, nc=args.nc),
                    # RandomClampTensors(min_margin=0, max_margin=0.2),
                ])
            else:
                transform = transforms.Compose([
                    # transforms.Grayscale(num_output_channels=args.nc),
                    transforms.Resize(args.image_size),
                    transforms.CenterCrop(args.image_size),
                    # transforms.RandomResizedCrop(args.image_size),
                    transforms.ToTensor(),
                    Normalize_RandomInvert_pixels(p=-1, nc=args.nc),
                ])
        elif args.dataset in {'cifar9', 'stl9'}:
            transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                # transforms.RandomHorizontalFlip(p=p),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset in {'amazon', 'dslr', 'webcam'}:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    if args.dataset == 'usps':
        train_dataset = mydset.USPS(root=args.dataroot,
                                    download=True,
                                    train=True,
                                    transform=transform)
        test_dataset = mydset.USPS(root=args.dataroot,
                                   download=True,
                                   train=False,
                                   transform=transform)

    elif args.dataset == 'mnist':
        train_dataset = mydset.MNIST(root=args.dataroot,
                                     download=True,
                                     train=True,
                                     transform=transform)
        test_dataset = mydset.MNIST(root=args.dataroot,
                                    download=True,
                                    train=False,
                                    transform=transform)

    elif args.dataset == 'svhn':

        train_dataset = mydset.SVHN(root=args.dataroot,
                                    download=True,
                                    train=True,
                                    transform=transform)
        test_dataset = mydset.SVHN(root=args.dataroot,
                                   download=True,
                                   train=False,
                                   transform=transform)

    elif args.dataset == 'syndigits':
        train_dataset = mydset.SYNDIGITS(root=args.dataroot,
                                         download=False,
                                         train=True,
                                         transform=transform)
        test_dataset = mydset.SYNDIGITS(root=args.dataroot,
                                        download=False,
                                        train=False,
                                        transform=transform)

    elif args.dataset == 'cifar9':
        train_dataset = mydset.CIFAR9(root=args.dataroot,
                                      download=True,
                                      train=True,
                                      transform=transform)
        test_dataset = mydset.CIFAR9(root=args.dataroot,
                                     download=True,
                                     train=False,
                                     transform=transform)

    elif args.dataset == 'stl9':
        train_dataset = mydset.STL9(root=args.dataroot,
                                    download=True,
                                    train=True,
                                    transform=transform)
        test_dataset = mydset.STL9(root=args.dataroot,
                                   download=True,
                                   train=False,
                                   transform=transform)

    elif args.dataset == 'gtsrb':
        train_dataset = mydset.GTSRB(root=args.dataroot,
                                     train=True,
                                     transform=transform)
        test_dataset = mydset.GTSRB(root=args.dataroot,
                                    train=False,
                                    transform=transform)

    elif args.dataset == 'amazon':
        # dataset = dset.ImageFolder(root=root_path + dir, transform=transform_dict[phase])
        # train_size = int(0.8 * len(data))
        # test_size = len(data) - train_size
        # data_train, data_val = torch.utils.data.random_split(data, [train_size, test_size])
        dataset_path = os.path.join(args.dataroot, 'office', 'amazon', 'images')
        train_dataset = mydset.OFFICE(root=dataset_path,
                                      train=True,
                                      transform=transform)
        test_dataset = mydset.OFFICE(root=dataset_path,
                                     train=False,
                                     transform=transform)

    elif args.dataset == 'dslr':
        dataset_path = os.path.join(args.dataroot, 'office', 'dslr', 'images')
        train_dataset = mydset.OFFICE(root=dataset_path,
                                      train=True,
                                      transform=transform)
        test_dataset = mydset.OFFICE(root=dataset_path,
                                     train=False,
                                     transform=transform)

    elif args.dataset == 'webcam':
        dataset_path = os.path.join(args.dataroot, 'office', 'webcam', 'images')
        train_dataset = mydset.OFFICE(root=dataset_path,
                                      train=True,
                                      transform=transform)
        test_dataset = mydset.OFFICE(root=dataset_path,
                                     train=False,
                                     transform=transform)

    elif args.dataset == 'synsigns':
        train_dataset = mydset.SYNSIGNS(root=args.dataroot,
                                        train=True,
                                        transform=transform)
        test_dataset = mydset.SYNSIGNS(root=args.dataroot,
                                       train=False,
                                       transform=transform)

    # elif args.dataset == 'celeba':
    #     imdir = 'train' if train_flag else 'val'
    #     dataroot = os.path.join(args.dataroot, imdir)
    #     if args.image_size != 64:
    #         raise ValueError('the image size for CelebA args.dataset need to be 64!')
    #
    #     args.dataset = FolderWithImages(root=dataroot,
    #                                input_transform=transforms.Compose([
    #                                    ALICropAndScale(),
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize(
    #                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                ]),
    #                                target_transform=transforms.ToTensor()
    #                                )
    #
    # elif args.dataset in ['imagenet', 'folder', 'lfw']:
    #     dataset = dset.ImageFolder(root=args.dataroot,
    #                                transform=transform)
    #
    # elif args.dataset == 'lsun':
    #     dataset = dset.LSUN(db_path=args.dataroot,
    #                         classes=['bedroom_train'],
    #                         transform=transform)
    #
    # elif args.dataset == 'cifar10':
    #     dataset = dset.CIFAR10(root=args.dataroot,
    #                            download=False,
    #                            train=train_flag,
    #                            transform=transform)
    #
    # elif args.dataset == 'cifar100':
    #     dataset = dset.CIFAR100(root=args.dataroot,
    #                             download=False,
    #                             train=train_flag,
    #                             transform=transform)
    #
    # elif args.dataset == 'celeba':
    #     imdir = 'train' if train_flag else 'val'
    #     dataroot = os.path.join(args.dataroot, imdir)
    #     if args.image_size != 64:
    #         raise ValueError('the image size for CelebA dataset need to be 64!')
    #
    #     dataset = FolderWithImages(root=dataroot,
    #                                input_transform=transforms.Compose([
    #                                    ALICropAndScale(),
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize(
    #                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                ]),
    #                                target_transform=transforms.ToTensor()
    #                                )
    else:
        raise ValueError("Unknown dataset %s" % (args.dataset))

    print('{}: train: count={}, X.shape={}, X.min={}, X.max={}, test: count={}, X.shape={}, X.min={}, X.max={}'.format(
        args.dataset.upper(), train_dataset.__len__(), train_dataset[0][0].shape, train_dataset[0][0].min(), train_dataset[0][0].max(),
        test_dataset.__len__(), test_dataset[0][0].shape, test_dataset[0][0].min(), test_dataset[0][0].max()))

    return train_dataset, test_dataset


def normalize(x, dim=1):
    return x.div(x.norm(2, dim=dim).expand_as(x))


def match(x, y, dist):
    '''
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    '''
    if dist == 'L2':
        return (x - y).pow(2).mean()
    elif dist == 'L1':
        return (x - y).abs().mean()
    elif dist == 'cos':
        x_n = normalize(x)
        y_n = normalize(y)
        return 2 - (x_n).mul(y_n).mean()
    else:
        assert dist == 'none', 'wtf ?'


def process_grad_batch(params, clipping=1):
    n = params[0].grad_batch.shape[0]
    grad_norm_list = torch.zeros(n).cuda()
    for p in params:
        flat_g = p.grad_batch.reshape(n, -1)
        current_norm_list = torch.norm(flat_g, dim=1)
        grad_norm_list += torch.square(current_norm_list)
    grad_norm_list = torch.sqrt(grad_norm_list)
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1

    for p in params:
        p_dim = len(p.shape)
        scaling = scaling.view([n] + [1]*p_dim)
        p.grad_batch *= scaling
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)


def get_data_loader(dataset, batchsize):
    if (dataset == 'svhn'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.SVHN('./data/SVHN', split='train', download=True, transform=transform)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=73257, pin_memory=True, shuffle=True,
        #                                           num_workers=4)  # load full btach into memory, to concatenate with extra data
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, pin_memory=True, shuffle=True,
                                                  num_workers=0)  # load full btach into memory, to concatenate with extra data

        extraset = torchvision.datasets.SVHN('./data/SVHN', split='extra', download=True, transform=transform)
        # extraloader = torch.utils.data.DataLoader(extraset, batch_size=531131, pin_memory=True, shuffle=True, num_workers=4)  # load full btach into memory
        extraloader = torch.utils.data.DataLoader(extraset, batch_size=batchsize, pin_memory=True, shuffle=True,
                                                  num_workers=0)  # load full btach into memory

        testset = torchvision.datasets.SVHN('./data/SVHN', split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, pin_memory=True, shuffle=False,
                                                 num_workers=0)

        return trainloader, extraloader, testloader, len(trainset) + len(extraset), len(testset)

    elif (dataset == 'mnist'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        training_set = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(training_set, batch_size=batchsize, pin_memory=True, shuffle=True,
                                                  num_workers=0)

        test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batchsize, pin_memory=True, shuffle=False,
                                                 num_workers=0)

        return trainloader, testloader, len(training_set), len(test_set)

    else:
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if not os.path.exists('./data/CIFAR10'):
            os.makedirs('./data/CIFAR10')

        trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True,
                                                transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True,
                                               transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)

        return trainloader, testloader, len(trainset), len(testset)


def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32, rgp=True):
    previous_eps = eps
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T
        if (rgp):
            rdp = compute_rdp(q, cur_sigma, steps,
                              orders) * 2  ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
        if (cur_eps < eps and cur_sigma > interval):
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break
    return cur_sigma, previous_eps


## interval: init search inerval
## rgp: use residual gradient perturbation or not
def get_sigma(q, T, eps, delta, init_sigma=10, interval=1., rgp=True):
    cur_sigma = init_sigma

    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps


def restore_param(cur_state, state_dict):
    own_state = cur_state
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)


def sum_list_tensor(tensor_list, dim=0):
    return torch.sum(torch.cat(tensor_list, dim=dim), dim=dim)


def flatten_tensor(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


def checkpoint(net, acc, epoch, sess):
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'approx_error': net.gep.approx_error
    }

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + sess + '.ckpt')


def adjust_learning_rate(optimizer, init_lr, epoch, all_epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    decay = 1.0
    if (epoch < all_epoch * 0.5):
        decay = 1.
    elif (epoch < all_epoch * 0.75):
        decay = 10.
    else:
        decay = 100.

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr / decay
    return init_lr / decay

