import shutil
import torch
import torch.cuda
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dutils
import torchvision.utils as vutils
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
import lr_schedule
import argparse

matplotlib.use('Agg')


# SE Imports
import numpy as np
import cmdline_helpers
import socket
import datetime
import math
import random
import os
import sys
import timeit

# MMD GAN Imports

from util import get_data, get_sigma, restore_param, sum_list_tensor, flatten_tensor, checkpoint, \
    adjust_learning_rate
from model.build_network import Network

# package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad

from basis_matching import GEP



def test(net, l_src_test, l_tgt_test):
    net.E.eval()
    net.C.eval()

    src_test_correct = 0
    tgt_test_correct = 0
    with torch.no_grad():
        for data, target in l_src_test:
            data, target = data.cuda(), target.cuda()
            if net.C.module.use_gumbel:
                _, output_logits, output = net.discriminator(data)
            else:
                _, output_logits = net.discriminator(data)
                output = F.softmax(output_logits, dim=1)
            src_test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            src_test_correct += src_test_pred.eq(target.view_as(src_test_pred)).sum().item()

        for data, target in l_tgt_test:
            data, target = data.cuda(), target.cuda()
            if net.C.module.use_gumbel:
                _, output_logits, output = net.discriminator(data)
            else:
                _, output_logits = net.discriminator(data)
                output = F.softmax(output_logits, dim=1)
            tgt_test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            tgt_test_correct += tgt_test_pred.eq(target.view_as(tgt_test_pred)).sum().item()

    return src_test_correct, tgt_test_correct



def experiment(args):
    exp = args.exp
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    epoch_size = args.epoch_size
    seed = args.seed
    log_file = args.log_file
    workers = args.workers
    image_size = args.image_size
    nc = args.nc
    nz = args.nz
    use_gpu = True
    gpus = args.gpus
    img_dir = args.img_dir
    logs_dir = args.logs_dir
    ckpt_dir = args.ckpt_dir
    plot_interval = args.plot_interval
    experiment = args.exp

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Some variable hardcoding - Only for development
    seed = args.seed
    simul_train_src_tgt = True
    # use_scheduler = args.use_scheduler  # True/False
    use_sampler = args.use_sampler  # True/False
    use_ramp_sup = args.ramp > 0
    use_ramp_unsup = args.ramp > 0
    ramp_sup_weight_in_list = [1.0]
    ramp_unsup_weight_in_list = [1.0]
    use_gen_sqrt = args.use_gen_sqrt  # True/False (False is better : Jobs 7621/7622)
    train_GnE = args.train_GnE  # True/False

    # sigma for MMD
    base = 1.0
    sigma_list = [1, 2, 4, 8, 16]

    lambda_MMD = 1.0
    lambda_AE_X = 8.0
    lambda_AE_Y = 8.0
    lambda_rg = 16.0

    lambda_ssl = 1.0
    lambda_sul = 0.0
    lambda_tul = 1.0
    if args.exp in {'usps_mnist', 'mnist_usps', 'svhn_mnist', 'mnist_svhn'}:
        lambda_sal = 0.0
        lambda_tal = 0.0
    else:
        lambda_sal = 0.0
        lambda_tal = 0.0

    machinename = socket.gethostname()
    hostname = timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")

    absolute_pyfile_path = os.path.abspath(sys.argv[0])
    args.absolute_pyfile_path = os.path.abspath(sys.argv[0])

    absolute_base_path = os.path.dirname(absolute_pyfile_path)
    args.absolute_base_path = os.path.dirname(absolute_pyfile_path)

    args.dataroot = os.path.expanduser(args.dataroot)
    dataroot = os.path.join(absolute_base_path, args.dataroot)
    dataset_path = os.path.join(absolute_base_path, args.dataroot, args.dataset)
    args.dataroot = os.path.join(absolute_base_path, args.dataroot)
    args.dataset_path = os.path.join(absolute_base_path, args.dataroot, args.dataset)

    args.logs_dir = os.path.expanduser(args.logs_dir)
    logs_dir = os.path.join(absolute_base_path, args.logs_dir)
    args.logs_dir = logs_dir

    log_num = 0
    log_file = logs_dir + '/' + hostname + '_' + machinename + '_' + args.exp + '_' + str(log_num) + '_' + args.epoch_size + '_ss_tu' + '.txt'

    # Setup logfile to store output logs
    if log_file is not None:
        while os.path.exists(log_file):
            log_num += 1
            log_file = '{0}/{3}_{1}_{2}_{4}_ss_tu.txt'.format(args.logs_dir, args.exp, log_num, hostname, args.epoch_size)
        # return
    args.log_file = log_file

    args.img_dir = os.path.expanduser(args.img_dir)
    img_dir = os.path.join(absolute_base_path, args.img_dir, hostname + '_' + machinename + '_' + args.exp) + '_' + str(log_num)
    args.img_dir = img_dir

    args.ckpt_dir = os.path.expanduser(args.ckpt_dir)
    ckpt_dir = os.path.join(absolute_base_path, args.ckpt_dir, hostname + '_' + machinename + '_' + args.exp) + '_' + str(log_num) + '_' + args.epoch_size + '_ss_tu'
    args.ckpt_dir = ckpt_dir

    # Create folder to save log files
    os.system('mkdir -p {0}'.format(logs_dir))  # create folder to store log files

    settings = locals().copy()

    # Setup output
    def log(text):
        print(text)
        if log_file is not None:
            with open(log_file, 'a') as f:
                str = text + '\n'
                f.write(text + '\n')
                f.flush()
                f.close()
                return str

    cmdline_helpers.ensure_containing_dir_exists(log_file)

    try:
        n_iter = 0
        log_str = ''
        log_str += log('\n')
        log_str += log('Output log file {0} created'.format(log_file))
        log_str += log('File used to run the experiment : {0}'.format(absolute_pyfile_path))
        log_str += log('Output image files are stored in {0} directory'.format(img_dir))
        log_str += log('Model files are stored in {0} directory\n'.format(ckpt_dir))

        # Report setttings
        log_str += log('Settings: {}'.format(', '.join(['{}={}'.format(key, settings[key]) for key in sorted(list(settings.keys()))])))

        num_gpu = len(args.gpus.split(','))
        log_str += log('num_gpu: {0}, GPU-ID: {1}'.format(num_gpu, args.gpus))

        random.seed(seed)
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            cudnn.benchmark = True

            # set current cuda device to 0
            log_str += log('current cuda device = {}'.format(torch.cuda.current_device()))
            torch.cuda.set_device(0)
            log_str += log('using cuda device = {}'.format(torch.cuda.current_device()))
            batch_size *= int(num_gpu)
            # args.buffer_size = batch_size * 10
            # args.learning_rate *= torch.cuda.device_count()

        else:
            raise EnvironmentError("GPU device not available!")

        transform = None

        # Get datasets
        if exp == 'usps_mnist':
            args.nc = 1
            args.image_size = 28
            args.dataset = 'usps'
            d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
            args.dataset = 'mnist'
            d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
        elif exp == 'mnist_usps':
            args.nc = 1
            args.image_size = 28
            args.dataset = 'mnist'
            d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
            args.dataset = 'usps'
            d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
        elif exp == 'svhn_mnist':
            if args.network_type == 'dade':
                args.nc = 1
                args.image_size = 28
            else:
                args.nc = 3
                args.image_size = 32
            args.dataset = 'svhn'
            d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
            args.dataset = 'mnist'
            d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
        elif exp == 'mnist_svhn':
            # args.nc = 1
            # args.image_size = 32
            # args.image_size = 28
            if args.network_type == 'dade':
                args.nc = 1
                args.image_size = 28
            else:
                args.nc = 3
                args.image_size = 32
            args.dataset = 'mnist'
            d_src_train_notinverted, _ = get_data(args, transform=transform, train_flag=False)
            d_src_train_inverted, d_src_test = get_data(args, transform=transform, train_flag=True)
            d_src_train = torch.utils.data.ConcatDataset([d_src_train_inverted, d_src_train_notinverted])
            d_src_train.dataset_name = 'mnist'
            d_src_train.n_classes = 10
            d_src_train.transform = d_src_train_notinverted.transform
            args.dataset = 'svhn'
            d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)

        elif exp in {'amazon_dslr', 'amazon_webcam', 'dslr_amazon', 'dslr_webcam', 'webcam_amazon', 'webcam_dslr'}:
            args.nc = 3
            args.image_size = 224
            args.dataset = exp.split('_')[0]
            d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
            args.dataset = exp.split('_')[1]
            d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
        else:
            log_str += log('Error : Unknown experiment type \'{}\''.format(exp))
            raise ValueError('Unknown experiment type \'{}\''.format(exp))

        log_str += log('\nSRC : {}: train: count={}, X.shape={} test: count={}, X.shape={}'.format(
            d_src_train.dataset_name.upper(), d_src_train.__len__(), d_src_train[0][0].shape,
            d_src_test.__len__(), d_src_test[0][0].shape))
        log_str += log('TGT : {}: train: count={}, X.shape={} test: count={}, X.shape={}'.format(
            d_tgt_train.dataset_name.upper(), d_tgt_train.__len__(), d_tgt_train[0][0].shape,
            d_tgt_test.__len__(), d_tgt_test[0][0].shape))

        n_classes = d_src_train.n_classes

        log_str += log('\nTransformations for SRC and TGT datasets ...')
        log_str += log('SRC : {0} - transformation : {1}'.format(d_src_train.dataset_name.upper(), d_src_train.transform))
        log_str += log('TGT : {0} - transformation : {1}'.format(d_tgt_train.dataset_name.upper(), d_tgt_train.transform))
        log_str += log('\nNumber of classes : {}'.format(n_classes))
        log_str += log('\nLoaded  Source and Target data respectively')


        n_samples = max(d_src_train.__len__(), d_tgt_train.__len__())
        if epoch_size == 'large':
            n_samples = max(d_src_train.__len__(), d_tgt_train.__len__())
        elif epoch_size == 'small':
            n_samples = min(d_src_train.__len__(), d_tgt_train.__len__())
        elif epoch_size == 'source':
            n_samples = d_src_train.__len__()
        elif epoch_size == 'target':
            n_samples = d_tgt_train.__len__()
        else:
            raise NotImplementedError

        log_str += log('\nUsing epoch_size : {}'.format(args.epoch_size))

        n_train_batches = n_samples // batch_size
        n_src_train_batches = d_src_train.__len__() // batch_size
        n_tgt_train_batches = d_tgt_train.__len__() // batch_size
        n_train_samples = n_train_batches * batch_size
        n_src_train_samples = n_src_train_batches * batch_size
        n_tgt_train_samples = n_tgt_train_batches * batch_size

        pin_memory = True
        if use_sampler:
            try:
                d_src_train_labels = d_src_train.train_labels.numpy()
            except:
                d_src_train_labels = np.concatenate(
                    (d_src_train_inverted.train_labels.numpy(), d_src_train_notinverted.train_labels.numpy()), axis=0)

            labels, counts_src = np.unique(d_src_train_labels, return_counts=True)

            prior_src = torch.Tensor(counts_src / counts_src.sum()).float().cuda()

            counts_src = torch.from_numpy(counts_src).type(torch.double)
            weights_src = 1.0 / counts_src
            sampler_weights_src = weights_src[d_src_train_labels]
            sampler_src = torch.utils.data.sampler.WeightedRandomSampler(weights=sampler_weights_src,
                                                                         num_samples=n_train_samples,
                                                                         replacement=True)
            l_src_train = dutils.DataLoader(d_src_train,
                                            batch_size=batch_size,
                                            sampler=sampler_src,
                                            num_workers=int(workers),
                                            pin_memory=pin_memory,
                                            drop_last=True)

            d_tgt_train_labels = d_tgt_train.train_labels.numpy()
            labels, counts_tgt = np.unique(d_tgt_train_labels, return_counts=True)

            prior_tgt = torch.Tensor(counts_tgt / counts_tgt.sum()).float().cuda()

            counts_tgt = torch.from_numpy(counts_tgt).type(torch.double)
            weights_tgt = 1.0 / counts_tgt
            sampler_weights_tgt = weights_tgt[d_tgt_train_labels]
            sampler_tgt = torch.utils.data.sampler.WeightedRandomSampler(weights=sampler_weights_tgt,
                                                                         num_samples=n_train_samples,
                                                                         replacement=True)
            l_tgt_train = dutils.DataLoader(d_tgt_train,
                                            batch_size=batch_size,
                                            sampler=sampler_tgt,
                                            num_workers=int(workers),
                                            pin_memory=pin_memory,
                                            drop_last=True)


        else:
            shuffle = True
            l_src_train = dutils.DataLoader(d_src_train,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=int(workers),
                                            pin_memory=pin_memory,
                                            drop_last=True)

            l_tgt_train = dutils.DataLoader(d_tgt_train,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=int(workers),
                                            pin_memory=pin_memory,
                                            drop_last=True)

        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )
        # inv_tensor = inv_normalize(tensor)

        all_labels = []
        for i, (X_src, lab) in enumerate(l_src_train):
            all_labels.extend(lab.numpy())
        labels, counts_src = np.unique(all_labels, return_counts=True)

        prior_src_train = torch.Tensor(counts_src / counts_src.sum()).float().cuda()

        log_str += log('prior_src_train : {}'.format(prior_src_train))

        all_labels = []
        for _, lab in l_tgt_train:
            all_labels.extend(lab.numpy())
        labels, counts_src = np.unique(all_labels, return_counts=True)

        prior_tgt_train = torch.Tensor(counts_src / counts_src.sum()).float().cuda()

        log_str += log('prior_tgt_train : {}'.format(prior_tgt_train))

        # Get Dataloaders from datasets

        shuffle = False
        l_src_test = dutils.DataLoader(d_src_test,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=int(2),
                                       pin_memory=pin_memory,
                                       drop_last=False)

        l_tgt_test = dutils.DataLoader(d_tgt_test,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=int(2),
                                       pin_memory=pin_memory,
                                       drop_last=False)


        num_public_examples = args.aux_data_size

        public_inputs = torch.load('./imagenet_examples_2000')[:num_public_examples]
        if (not args.real_labels):
            public_targets = torch.randint(high=10, size=(num_public_examples,))
        public_inputs, public_targets = public_inputs.cuda(), public_targets.cuda()

        print('\n==>  Creating GEP class instance')
        gep = GEP(args.num_bases, args.batch_size, args.clip0, args.clip1, args.power_iter).cuda()
        ## attach auxiliary data to GEP instance
        gep.public_inputs = public_inputs
        gep.public_targets = public_targets

        if (args.resume):
            try:
                assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
                checkpoint_file = './checkpoint/' + args.sess + '.ckpt'
                checkpoint = torch.load(checkpoint_file)
                net = Network(args)
                restore_param(net.state_dict(), checkpoint['net'])
                best_acc = checkpoint['acc']
                start_epoch = checkpoint['epoch'] + 1
                torch.set_rng_state(checkpoint['rng_state'])
                approx_error = checkpoint['approx_error']
            except:
                print('resume from checkpoint failed')
        else:
            net = Network(args)


        net.gep = gep

        print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP' % (args.eps, args.delta))
        sampling_prob = args.batch_size / n_train_samples # n_tgt_train_samples
        steps = int(args.num_epochs / sampling_prob)
        sigma, eps = get_sigma(sampling_prob, steps, args.eps, args.delta, rgp=args.rgp)
        noise_multiplier0 = noise_multiplier1 = sigma
        print('noise scale for gradient embedding: ', noise_multiplier0, 'noise scale for residual gradient: ',
              noise_multiplier1, '\n rgp enabled: ', args.rgp, '\n privacy guarantee: ', eps)

        # Dividing parameters in to %d groups
        num_params = 0
        np_list = []
        for p in net.parameters():
            num_params += p.numel()
            np_list.append(p.numel())

        print('\n==> Dividing parameters in to %d groups' % args.num_groups)
        gep.num_param_list = group_params(num_params, args.num_groups)

        log_str += log('\nBuilding Network from {} ...'.format(args.network_type.upper()))
        log_str += log('Encoder : {}'.format(net.E))
        log_str += log('Classifier : {}'.format(net.C))
        log_str += log('Network Built ...')

        # sigma for MMD
        sigma_list = [sigma / base for sigma in sigma_list]

        # put variable into cuda device
        fixed_noise = Variable(net.fixed_noise, requires_grad=False)

        # For storing the network output of the last buffer_size samples
        args.buffer_size = int(args.buffer_size*n_train_samples)
        p_sum_denominator = torch.rand(args.buffer_size, n_classes).cuda()
        p_sum_denominator /= p_sum_denominator.sum(1).unsqueeze(1).expand_as(p_sum_denominator)

        # setup optimizer
        log_str += log('\noptimizerE : {}'.format(net.optimizerE))
        log_str += log('optimizerC : {}'.format(net.optimizerC))

        if args.lr_decay_type == 'scheduler':
            scheduler_E = ReduceLROnPlateau(net.optimizerE, 'max')
            scheduler_C = ReduceLROnPlateau(net.optimizerC, 'max')


        # Loss function for supervised loss
        classification_criterion = nn.CrossEntropyLoss().cuda()



        log_str += log('\nTraining...')
        if simul_train_src_tgt:
            log_str += log('Note : Simultaneous training of source and target domains. No swapping after e epochs ...')
        else:
            log_str += log('Note : No Simultaneous training of source and target domains. swapping after e epochs ...')

        best_src_test_acc = 0  # Best epoch wise src acc
        best_src_test_acc_inter = 0  # Best generator wise src acc
        best_tgt_test_acc = 0  # Best epoch wise src acc
        best_tgt_test_acc_inter = 0  # Best generator wise src acc
        src_test_acc = 0  # dummy init for scheduler

        time = timeit.default_timer()
        total_loss_epochs = []


        src_log_prob_buffer = []  # src_log_prob_buffer
        tgt_log_prob_buffer = []  # tgt_log_prob_buffer
        z_src_one_hot_buffer = []  # src_log_prob_buffer
        z_tgt_one_hot_buffer = []  # tgt_log_prob_buffer

        log_str += log('Checkpoint directory to store files for current run : {}'.format(net.args.ckpt_dir))

        net.writer.add_text('Log Text', log_str)

        n_iter = 0
        if args.network_type == 'cdan':
            param_lr = []
            for param_group in net.optimizerE.param_groups:
                param_lr.append(param_group["lr"])
            for param_group in net.optimizerC.param_groups:
                param_lr.append(param_group["lr"])
            schedule_param = {'lr': 0.001, 'gamma': 0.001, 'power': 0.75}  #  optimizer_config["lr_param"]
            # lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
            lr_scheduler = lr_schedule.schedule_dict['inv']


        # # Reset dataloader iterator
        for epoch in range(num_epochs):

            if epoch != 0 and args.lr_decay_type == 'scheduler':
                scheduler_E.step(src_test_acc)
                scheduler_C.step(src_test_acc)
            elif args.lr_decay_type == 'geometric':
                lr = args.learning_rate * (args.lr_decay_rate ** (epoch // args.lr_decay_period))
                for param_group in net.optimizerC.param_groups:
                    param_group['lr'] = lr
                for param_group in net.optimizerE.param_groups:
                    param_group['lr'] = lr
            elif args.lr_decay_type == 'cdan_inv':
                net.optimizerE = lr_scheduler(net.optimizerE, n_iter, **schedule_param)
                net.optimizerC = lr_scheduler(net.optimizerC, n_iter, **schedule_param)
                lr = args.learning_rate
            else:
                lr = args.learning_rate

            if use_ramp_sup:
                epoch_ramp_2 = epoch % (args.ramp * 2)
                ramp_sup_value = 1.0

                ramp_sup_weight_in_list[0] = ramp_sup_value

            if use_ramp_unsup:
                epoch_ramp_2 = epoch % (args.ramp * 2)
                if epoch < (args.ramp):
                    ramp_unsup_value = math.exp(-(args.ramp - epoch) * 5.0 / args.ramp)
                else:
                    ramp_unsup_value = 1.0 - math.exp(-(args.ramp - epoch_ramp_2) * 5.0 / args.ramp)

                ramp_unsup_weight_in_list[0] = ramp_unsup_value

            net.writer.add_scalar('constant/learning_rate', lr, epoch)
            net.writer.add_scalar('weights_loss/sup/src', lambda_ssl, epoch)
            net.writer.add_scalar('weights_loss/unsup/src', lambda_sul, epoch)
            net.writer.add_scalar('weights_loss/unsup/tgt', lambda_tul, epoch)
            net.writer.add_scalar('weights_loss/adv/src', lambda_sal, epoch)
            net.writer.add_scalar('weights_loss/adv/tgt', lambda_tal, epoch)
            net.writer.add_scalar('weights_loss/sup/src_ramp', ramp_sup_weight_in_list[0], epoch)
            net.writer.add_scalar('weights_loss/unsup/src_ramp', ramp_unsup_weight_in_list[0], epoch)
            net.writer.add_scalar('weights_loss/unsup/tgt_ramp', ramp_unsup_weight_in_list[0], epoch)


            net.E.train()
            net.C.train()
            epoch_loss = 0
            # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

            # if use_sampler:
            for i, ((X_src, y_src), (X_tgt, _)) in enumerate(zip(l_src_train, l_tgt_train)):
                n_iter = (epoch * len(l_src_train)) + i
                net.zero_grad()


                total_loss = 0

                # Get Source and Target Images
                X_src = Variable(X_src.cuda())
                X_tgt = Variable(X_tgt.cuda())
                X_src_batch_size = X_src.size(0)
                X_tgt_batch_size = X_tgt.size(0)

                # Exit without processing last uneven batch
                if X_src_batch_size != X_tgt_batch_size:
                    break

                # Get only Source labels
                y_src = Variable(y_src.cuda())

                # Train Enc + Dec + Classifier on both Source and Target Domain data
                if (args.private):
                    logging = epoch % 20 == 0
                    loss_func = nn.CrossEntropyLoss().cuda()
                    loss_func = extend(loss_func)
                    ## compute anchor subspace
                    net.optimizerE.zero_grad()
                    net.optimizerC.zero_grad()
                    net.gep.get_anchor_space(net.discriminator, loss_func=loss_func, logging=logging)
                    ## collect batch gradients
                    batch_grad_list = []
                    # net.optimizerE.zero_grad()
                    net.optimizerE.zero_grad()
                    net.optimizerC.zero_grad()
                    # outputs = net(l_tgt_train)
                    if net.C.module.use_gumbel:
                        src_enc_out, src_logits_out, z_src_one_hot = net.discriminator(X_src)
                        tgt_enc_out, tgt_logits_out, z_tgt_one_hot = net.discriminator(X_tgt)
                    else:
                        src_enc_out, src_logits_out = net.discriminator(X_src)
                        tgt_enc_out, tgt_logits_out = net.discriminator(X_tgt)
                    loss = loss_func(tgt_logits_out, d_tgt_train_labels)
                    with backpack(BatchGrad()):
                        loss.backward()
                    for p in net.parameters():
                        batch_grad_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
                        del p.grad_batch
                    ## compute gradient embeddings and residual gradients
                    clipped_theta, residual_grad, target_grad = net.gep(flatten_tensor(batch_grad_list),
                                                                           logging=logging)
                    ## add noise to guarantee differential privacy
                    theta_noise = torch.normal(0, noise_multiplier0 * args.clip0 / args.batch_size,
                                               size=clipped_theta.shape,
                                               device=clipped_theta.device)
                    grad_noise = torch.normal(0, noise_multiplier1 * args.clip1 / args.batch_size,
                                              size=residual_grad.shape,
                                              device=residual_grad.device)
                    clipped_theta += theta_noise
                    residual_grad += grad_noise
                    ## update with Biased-GEP or GEP
                    if (args.rgp):
                        noisy_grad = gep.get_approx_grad(clipped_theta) + residual_grad
                    else:
                        noisy_grad = gep.get_approx_grad(clipped_theta)
                    if (logging):
                        print('target grad norm: %.2f, noisy approximation norm: %.2f' % (
                            target_grad.norm().item(), noisy_grad.norm().item()))
                    ## make use of noisy gradients
                    offset = 0
                    for p in net.zero_grad().parameters():
                        shape = p.grad.shape
                        numel = p.grad.numel()
                        p.grad.data = noisy_grad[offset:offset + numel].view(
                            shape)  # + 0.1*torch.mean(pub_grad, dim=0).view(shape)
                        offset += numel
                else:
                    if net.C.module.use_gumbel:
                        src_enc_out, src_logits_out, z_src_one_hot = net.discriminator(X_src)
                        tgt_enc_out, tgt_logits_out, z_tgt_one_hot = net.discriminator(X_tgt)
                    else:
                        src_enc_out, src_logits_out = net.discriminator(X_src)
                        tgt_enc_out, tgt_logits_out = net.discriminator(X_tgt)


                # Supervised classification loss
                src_sup_loss = classification_criterion(src_logits_out, y_src)  # Loss 1 : Supervised loss  # torch.Size([64, 10])

                tgt_prob = F.softmax(tgt_logits_out, dim=1)  # torch.Size([64, 10])
                tgt_log_prob = F.log_softmax(tgt_logits_out, dim=1)  # torch.Size([64, 10])

                if not net.C.module.use_gumbel:
                    with torch.no_grad():
                        tgt_prob_cp = tgt_prob.detach().clone()

                        # Update denominator for the unsupervised loss

                        p_sum_denominator = torch.cat((tgt_prob_cp, p_sum_denominator), 0)[0:args.buffer_size]
                        tgt_prob_cp *= (prior_src_train / p_sum_denominator.sum(0)).expand_as(tgt_prob_cp)

                        _, y_tgt_pred = tgt_prob_cp.max(dim=1)
                        z_tgt_one_hot = torch.FloatTensor(X_tgt_batch_size, n_classes).cuda()
                        z_tgt_one_hot.zero_()
                        z_tgt_one_hot.scatter_(1, y_tgt_pred.unsqueeze(1), 1)
                        z_tgt_one_hot = Variable(z_tgt_one_hot)

                # maximization step
                tgt_exponent = torch.mm(z_tgt_one_hot, tgt_log_prob.t())
                tgt_exponent_new = tgt_exponent - torch.diag(tgt_exponent).view(X_tgt_batch_size, 1).expand_as(tgt_exponent)

                tgt_unsup_loss = torch.logsumexp(tgt_exponent_new, dim=1).mean()

                total_classifier_loss = lambda_ssl * ramp_sup_weight_in_list[0] * src_sup_loss + lambda_tul * ramp_unsup_weight_in_list[0] * tgt_unsup_loss  # Total Encoder + Classifier loss for Enc + Class training

                ### Add loss and loss weights to tensorboard logs
                net.writer.add_scalar('classifier_loss/sup/src', src_sup_loss.item(), n_iter)
                net.writer.add_scalar('classifier_loss/unsup/tgt', tgt_unsup_loss.item(), n_iter)
                net.writer.add_scalar('classifier_loss/total/batch', total_classifier_loss.item(), n_iter)

                total_loss += total_classifier_loss

                total_loss.backward()
                net.optimizerE.step()
                net.optimizerC.step()

                epoch_loss += total_loss.item()
                run_time = (timeit.default_timer() - time) / 60.0


                # Generator training ends
                if n_iter % plot_interval == 0:
                    with torch.no_grad():
                        fixed_noise = Variable(fixed_noise)  # total freeze netG_src , netG_tgt
                        X_fixed_noise_S = net.GS(fixed_noise).detach()
                        X_fixed_noise_T = net.GT(fixed_noise).detach()

                        image_grid = torch.zeros(
                            (X_fixed_noise_S.shape[0] * 2, X_fixed_noise_S.shape[1], X_fixed_noise_S.shape[2],
                             X_fixed_noise_S.shape[3]),
                            requires_grad=False)
                        image_grid[0::2, :, :, :] = X_fixed_noise_S
                        image_grid[1::2, :, :, :] = X_fixed_noise_T
                        image_grid_n_row = int(torch.sqrt(torch.FloatTensor([X_fixed_noise_S.shape[0]])).item())//2*2
                        image_grid.data.mul_(0.5).add_(0.5)
                        img_grid = vutils.make_grid(image_grid[:image_grid_n_row ** 2, :, :, :], nrow=image_grid_n_row)  # make an square grid of images and plot

                        net.writer.add_image('Generator_images_{0}'.format(args.exp), img_grid, n_iter)  # Tensor


            # Show the loss
            total_loss_epochs.append(epoch_loss/(i+1))
            net.writer.add_scalar('classifier_loss/total/epoch', epoch_loss/(i+1), epoch)

            # Create folder to save images
            if epoch == 0:
                os.system('mkdir -p {0}/Log'.format(net.args.ckpt_dir))  # create folder to store images


            if epoch % 30 == 0:
                net.save_model(suffix='_bkp')
                image_grid = torch.zeros(
                    (X_src.shape[0] * 2, X_src.shape[1], X_src.shape[2],
                     X_src.shape[3]),
                    requires_grad=False)
                image_grid[0::2, :, :, :] = X_src
                image_grid[1::2, :, :, :] = X_tgt
                # image_grid = image_grid * std + mean
                image_grid_n_row = int(torch.sqrt(torch.FloatTensor([X_src.shape[0]])).item()) // 2 * 2
                image_grid.data.mul_(0.5).add_(0.5)
                img_grid = vutils.make_grid(image_grid[:image_grid_n_row ** 2, :, :, :],
                                            nrow=image_grid_n_row)  # make an square grid of images and plot

                net.writer.add_image('Original_images_{0}'.format(args.exp), img_grid, epoch)  # Tensor

            src_train_acc, tgt_train_acc = test(net, l_src_train, l_tgt_train)
            src_test_acc, tgt_test_acc = test(net, l_src_test, l_tgt_test)

            if src_test_acc >= best_src_test_acc:

                src_prefix = '**'
                best_src_test_acc = src_test_acc
                net.save_model(suffix='_src_test')
            else:
                src_prefix = '  '

            if tgt_test_acc >= best_tgt_test_acc:
                tgt_prefix = '++'
                best_tgt_test_acc = tgt_test_acc
                net.save_model(suffix='_tgt_test')

                with torch.no_grad():
                    for j, (data, target) in enumerate(l_src_test):
                        if ((j+1)*l_src_test.batch_size) > args.n_test_samples:
                            break
                        # data, target = data.to(device), target.to(device)
                        data, target = data.cuda(), target.cuda()
                        if net.C.module.use_gumbel:
                            _, output_logits, output = net.discriminator(data)
                        else:
                            _, output_logits = net.discriminator(data)
                            output = F.softmax(output_logits, dim=1)
                        src_test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

                        labels_orig = target.cpu().numpy().tolist()
                        labels_pred = src_test_pred.squeeze().cpu().numpy().tolist()
                        if j == 0:
                            src_mat = output_logits
                            src_labels_orig = labels_orig
                            src_labels_pred = labels_pred
                            src_label_img = data
                        else:
                            src_mat = torch.cat((src_mat, output_logits), 0)
                            src_labels_orig.extend(labels_orig)
                            src_labels_pred.extend(labels_pred)
                            src_label_img = torch.cat((src_label_img, data), 0)

                    src_labels_orig_str = list(map(str, src_labels_orig))
                    src_labels_pred_str = list(map(str, src_labels_pred))
                    src_metadata = list(map(lambda src_labels_orig_str,
                                                   src_labels_pred_str: 'SRC_TST_Ori' + src_labels_orig_str + '_Pre' + src_labels_pred_str,
                                            src_labels_orig_str, src_labels_pred_str))


                    net.writer.add_pr_curve('SRC_PR', torch.tensor(src_labels_orig), torch.tensor(src_labels_pred), global_step=epoch)

            else:
                tgt_prefix = '  '

            log('{}{} E:{:03d}/{:03d} #B:{:03d}, t={:06.2f}m, L={:07.4f}, ACC : S_TRN= {:5.2%}, T_TRN= {:5.2%}, S_TST= {:5.2%}, T_TST= {:5.2%}'.format(
                src_prefix, tgt_prefix, epoch, num_epochs, i+1, run_time, epoch_loss/(i+1), src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)))

            net.writer.add_scalar('Accuracy/train/src', src_train_acc/(l_src_train.__len__() * batch_size), epoch)
            net.writer.add_scalar('Accuracy/train/tgt', tgt_train_acc/(l_tgt_train.__len__() * batch_size), epoch)
            net.writer.add_scalar('Accuracy/test/src', src_test_acc/len(d_src_test), epoch)
            net.writer.add_scalar('Accuracy/test/tgt', tgt_test_acc/len(d_tgt_test), epoch)

            net.writer.add_text('epoch log', '{}{} E:{:03d}/{:03d} #B:{:03d}, t={:06.2f}m, L={:07.4f}, ACC : S_TRN= {:5.2%}, T_TRN= {:5.2%}, S_TST= {:5.2%}, T_TST= {:5.2%}'.format(
                src_prefix, tgt_prefix, epoch, num_epochs, i+1, run_time, epoch_loss/(i+1), src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)), epoch)
            os.system('cp {0} {1}/Log/'.format(log_file, net.args.ckpt_dir))

        net.save_model()
        net.writer.close()

    except Exception as e:     # most generic exception you can catch
        log('Something went horribly wrong !!!')
        log('Error : {}'.format(str(e)))
        net = Network(args)
        net.writer.add_text('epoch log', 'Something went horribly wrong !!!', n_iter + 1)
        net.writer.add_text('epoch log', 'Error : {}'.format(str(e)), n_iter + 2)
        os.system('mv {0} {0}.err'.format(log_file))
        os.system('mv {1}/Log/{0} {1}/Log/{0}.err'.format(log_file, net.args.ckpt_dir))
        os.system('mv {0} {0}.err'.format(net.args.ckpt_dir))
        net.writer.close()
        raise

    try:
        os.mkdir('approx_errors')
    except:
        pass
    import pickle

    bfile = open('approx_errors/' + args.sess + '.pickle', 'wb')
    pickle.dump(net.gep.approx_error, bfile)
    bfile.close()

def group_params(num_p, groups):
    assert groups >= 1

    p_per_group = num_p // groups
    num_param_list = [p_per_group] * (groups - 1)
    num_param_list = num_param_list + [num_p - sum(num_param_list)]
    return num_param_list

if __name__ == '__main__':
    # Get argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['usps_mnist', 'mnist_usps',
                                          'svhn_mnist', 'mnist_svhn',
                                          'amazon_dslr', 'amazon_webcam',
                                          'dslr_amazon', 'dslr_webcam',
                                          'webcam_amazon', 'webcam_dslr',
                                          ], default='svhn_mnist', help='experiment to run')
    parser.add_argument('--dataset', choices=['usps', 'mnist',
                                              'svhn', 'office',
                                              'visda'
                                              ], default='mnist',  )
    parser.add_argument('--dataroot', type=str, default='./../visual/data', help='path to args.dataset')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate (Adam)')
    parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs : (default=3000)')
    parser.add_argument('--ramp', type=int, default=0, help='ramp for epochs')
    # parser.add_argument('--buffer_size', type=int, default=10000, help='length of the buffer for latent feature selection : (default=10000)')
    parser.add_argument('--buffer_size', type=float, default=0.4, help='length of the buffer for latent feature selection : (default=10000)')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size : (default=64)')
    parser.add_argument('--seed', type=int, default=1126, help='random seed (0 for time-based)')
    parser.add_argument('--log_file', type=str, default='', help='log file path (none to disable)')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--image_size', type=int, default=28, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=1, help='number of channel')
    parser.add_argument('--nz', type=int, default=100, help='dimension of noise input to generator')
    parser.add_argument('--gpus', type=str, default='0', help='using gpu device id')
    parser.add_argument('--resume', '-r', action='store_true', default=False,
                        help='resume from checkpoint(default : False)')
    parser.add_argument('--plot_interval', type=int, default=50, help='Number of plots required to save every iteration')
    parser.add_argument('--img_dir', type=str, default='./images', help='path to save images')
    parser.add_argument('--logs_dir', type=str, default='./logs', help='path to save logs')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoint', help='path to save model checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default='', help='path to checkpoint')

    # Hyperparameter search for reproducing - hp448.sh
    parser.add_argument('--use_sampler', action='store_true', default=True, help='use sampler for dataloader (default : False)')
    parser.add_argument('--use_gen_sqrt', action='store_true', default=False, help='use squareroot for MMD loss (default : False)')
    parser.add_argument('--train_GnE', action='store_true', default=False, help='train encoder with generator training (default : False)')
    parser.add_argument('--network_type', type=str, default='se', help='type of network [se|mcd|dade|cdan]')
    parser.add_argument('--use_tied_gen', action='store_true', default=False, help='use a single generator for source and target domain (default : False)')
    parser.add_argument('--epoch_size', default='large', choices=['large', 'small', 'source', 'target'], help='epoch size is either that of the smallest dataset, the largest, the source, or the target (default=source)')
    parser.add_argument('--use_drop', action='store_true', default=True, help='use dropout for classifier (default : False)')
    parser.add_argument('--use_bn', action='store_true', default=True, help='use batchnorm for classifier (default : False)')
    parser.add_argument('--lr_decay_type', choices=['scheduler', 'geometric', 'none', 'cdan_inv'], default='geometric', help='lr_decay_type (default=none)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.6318, help='learning rate (Adam)')
    parser.add_argument('--lr_decay_period', type=float, default=30, help='learning rate (Adam)')
    parser.add_argument('--weight_init', type=str, default='none', help='type of weight initialization')
    parser.add_argument('--use_gumbel', action='store_true', default=False, help='use Gumbel softmax for label selection (default : False)')
    parser.add_argument('--n_test_samples', type=int, default=1000, help='number of test samples used for plotting t-SNE')

    ## arguments for learning with differential privacy
    parser.add_argument('--private', '-p', default=False, help='enable differential privacy')
    parser.add_argument('--eps', default=8., type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')

    parser.add_argument('--rgp', default=False, help='use residual gradient perturbation or not')
    parser.add_argument('--clip0', default=5., type=float, help='clipping threshold for gradient embedding')
    parser.add_argument('--clip1', default=2., type=float, help='clipping threshold for residual gradients')
    parser.add_argument('--power_iter', default=1, type=int, help='number of power iterations')
    parser.add_argument('--num_groups', default=10, type=int, help='number of parameters groups')
    parser.add_argument('--num_bases', default=1000, type=int, help='dimension of anchor subspace')

    parser.add_argument('--real_labels', default=False, help='use real labels for auxiliary dataset')
    parser.add_argument('--aux_dataset', default='imagenet', type=str,
                        help='name of the public dataset, [cifar10, cifar100, imagenet]')
    parser.add_argument('--aux_data_size', default=2000, type=int, help='size of the auxiliary dataset')

    args = parser.parse_args()
    print(args)

    experiment(args)


