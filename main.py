import time
import argparse
import random
import math
from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data
import timm
import torch
import torch.nn as nn
import numpy as np
from dataset.ImageNetMask import imagenet_r_mask, imagenet_a_mask
from methods import eata, eata_dp, sar_clip, sar, tent, sar_dp, deyo, deyo_come, eata_clip, tent_dp
from methods import deyo_dp, deyo_clip, deyo_come_dp, deyo_come_clip, tent_clip
import copy
from methods.sam_variants import SAM, DPSAM, DPSAT

def validate(val_loader, model, criterion, args, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):

            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            if args.algorithm in ['deyo', 'deyo_dp', 'deyo_clip',
                                   'deyo_come', 'deyo_come_dp', 'deyo_come_clip']:
                # DeYO / DeYO_COME adapt on every forward and return a tuple;
                # the first element is the logits. Adaptation runs under an
                # internal @torch.enable_grad() so the outer no_grad is fine.
                output = model(images, i)[0]
            else:
                output = model(images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)
            if args.debug and i >= 5:
                break
    return top1.avg, top5.avg


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # path of data, output dir
    parser.add_argument('--data', default='./data/imagenet', help='path to dataset')
    parser.add_argument('--data_corruption', default='./data/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--output', default='./output',
                        help='the output directory of this experiment')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')
    parser.add_argument('--rotation', default=False, type=bool,
                        help='if use the rotation ssl task for training (this is TTTs dataloader).')

    # model name
    parser.add_argument('--arch', default='vit', type=str, help='the default model architecture')
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate of the adaptation optimizer')
    parser.add_argument('--max_norm', type=float, default=0.0,
                        help='per-sample gradient clipping norm for the clip/DP variants')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='DP Gaussian noise multiplier; 0.0 = clip-only (no noise)')
    parser.add_argument('--reset', type=int, default=0,
                        help='0 = continual/online (no reset between shifts); 1 = episodic (reset before each shift)')
    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int,
                        help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000.,
                        help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000) * 0.40,
                        help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05,
                        help='\epsilon in Eqn. (5) for filtering redundant samples')
    parser.add_argument('--sar_margin_e0', default=math.log(1000) * 0.50, type=float,
                        help='the threshold for reliable minimization in SAR, Eqn. (2)')
    # overall experimental settings
    parser.add_argument('--exp_type', default='each_shift_reset', type=str, help='continual or each_shift_reset')
    # 'cotinual' means the model parameters will never be reset, also called online adaptation;
    # 'each_shift_reset' means after each type of distribution shift, e.g., ImageNet-C Gaussian Noise Level 5, the model parameters will be reset.
    parser.add_argument('--algorithm', default='tent', type=str, help='eata or eta or tent')

    # DeYO
    parser.add_argument('--aug_type', default='patch', type=str, help='DeYO augmentation for PLPD: patch, pixel, or occ')
    parser.add_argument('--occlusion_size', default=112, type=int, help='DeYO occ-augmentation occlusion size')
    parser.add_argument('--row_start', default=56, type=int, help='DeYO occ-augmentation occlusion row start')
    parser.add_argument('--column_start', default=56, type=int, help='DeYO occ-augmentation occlusion column start')
    parser.add_argument('--deyo_margin', default=0.5, type=float,
                        help='Entropy threshold for sample selection $\tau_\mathrm{Ent}$ in Eqn. (8)')
    parser.add_argument('--deyo_margin_e0', default=0.4, type=float,
                        help='Entropy margin for sample weighting $\mathrm{Ent}_0$ in Eqn. (10)')
    parser.add_argument('--plpd_threshold', default=0.3, type=float,
                        help='PLPD threshold for sample selection $\tau_\mathrm{PLPD}$ in Eqn. (8)')

    parser.add_argument('--filter_ent', default=1, type=int, help='DeYO: 1 = filter samples by entropy threshold (Eqn. 8)')
    parser.add_argument('--filter_plpd', default=1, type=int, help='DeYO: 1 = filter samples by PLPD threshold (Eqn. 8)')
    parser.add_argument('--reweight_ent', default=1, type=int, help='DeYO: 1 = reweight sample losses by entropy (Eqn. 10)')
    parser.add_argument('--reweight_plpd', default=1, type=int, help='DeYO: 1 = reweight sample losses by PLPD (Eqn. 10)')
    parser.add_argument('--patch_len', default=4, type=int, help='The number of patches per row/column')
    return parser.parse_args()


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    initial_state = copy.deepcopy(model.state_dict())

    # reset
    def reset():
        model.load_state_dict(initial_state)
        model.eval()
    model.reset = reset
    return model


def compute_fishers(subnet, args, configure_model, collect_params):
    """Compute the EATA Fisher-information regularizer on the clean ('original')
    set. Returns the configured subnet, its trainable params, and the fisher dict."""
    args.corruption = 'original'
    fisher_dataset, fisher_loader = prepare_test_data(args)
    fisher_dataset.set_dataset_size(args.fisher_size)
    fisher_dataset.switch_mode(True, False)

    subnet = configure_model(subnet)
    params, param_names = collect_params(subnet)
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    for iter_, (images, targets) in enumerate(fisher_loader, start=1):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            targets = targets.cuda(args.gpu, non_blocking=True)
        outputs = subnet(images)
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        for name, param in subnet.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    logger.info("compute fisher matrices finished")
    del ewc_optimizer
    print(f"Found {len(params)} trainable params")
    return subnet, params, fishers


if __name__ == '__main__':

    args = get_args()

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.arch == 'vit':
        subnet = timm.create_model('vit_base_patch16_224', pretrained=True)
        subnet = subnet.cuda()
        subnet = subnet.eval()

    elif args.arch == 'convnext':
        subnet = timm.create_model('convnext_tiny.in12k_ft_in1k', pretrained=True)
        subnet = subnet.cuda()
        subnet = subnet.eval()
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    logger = get_logger(name="project", output_directory=args.output,
                        log_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + "-log.txt", debug=False)

    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                          'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
                          'pixelate', 'jpeg_compression']
    logger.info(args)

    if args.exp_type == 'continual':
        common_corruptions = [[item, 'original'] for item in common_corruptions]
        common_corruptions = [subitem for item in common_corruptions for subitem in item]
    elif args.exp_type == 'each_shift_reset':
        print("continue")
    else:
        assert False, NotImplementedError

    if "imagenet-r" in args.data_corruption:
        common_corruptions = ["imagenet-r"]

    _only = os.environ.get('ONLY_CORRUPTION')
    if _only:
        common_corruptions = _only.split(',')

    logger.info(common_corruptions)

    # Scale the DeYO margins by log(num_class) and use deyo_margin_e0 as the
    # reweighting margin, matching new_settings.py exactly. ImageNet-C uses 1000
    # classes, ImageNet-R uses 200.
    num_class = 200 if "imagenet-r" in args.data_corruption else 1000
    deyo_margin = args.deyo_margin * math.log(num_class)
    deyo_margin_e0 = args.deyo_margin_e0 * math.log(num_class)

    if args.algorithm == 'tent':
        subnet = tent.configure_model(subnet)
        params, param_names = tent.collect_params(subnet)
        print(f"Found {len(params)} trainable params")
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = tent.Tent(subnet, optimizer, args.lr)
    elif args.algorithm == 'tent_dp':
        subnet = tent_dp.configure_model(subnet)
        params, param_names = tent_dp.collect_params(subnet)
        print(f"Found {len(params)} trainable params")
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = tent_dp.Tent(args, subnet, optimizer, args.lr, args.max_norm, args.noise)
    elif args.algorithm == 'tent_clip':
        subnet = tent_clip.configure_model(subnet)
        params, param_names = tent_clip.collect_params(subnet)
        print(f"Found {len(params)} trainable params")
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = tent_clip.Tent(args, subnet, optimizer, args.lr, args.max_norm, args.noise)
    elif args.algorithm == 'sar':
        subnet = sar.configure_model(subnet)
        params, param_names = sar.collect_params(subnet)
        logger.info(param_names)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, args.noise, args.max_norm, lr=args.lr, momentum=0.9)
        adapt_model = sar.SAR(subnet, optimizer, margin_e0=args.sar_margin_e0)
    elif args.algorithm == 'sar_clip':
        subnet = sar_clip.configure_model(subnet)
        params, param_names = sar_clip.collect_params(subnet)
        logger.info(param_names)
        base_optimizer = torch.optim.SGD
        # clip-only variant: DP noise is pinned to 0
        optimizer = DPSAM(params, base_optimizer, 0.0, args.max_norm, batch_size=args.batch_size, lr=args.lr, momentum=0.9)
        adapt_model = sar_clip.SARClip(subnet, optimizer, margin_e0=args.sar_margin_e0)
    elif args.algorithm == 'deyo':
        subnet = deyo.configure_model(subnet)
        params, param_names = deyo.collect_params(subnet)
        logger.info(param_names)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = deyo.DeYO(subnet, args, optimizer, deyo_margin=deyo_margin, margin_e0=deyo_margin_e0)
    elif args.algorithm == 'deyo_dp':
        subnet = deyo_dp.configure_model(subnet)
        params, param_names = deyo_dp.collect_params(subnet)
        logger.info(param_names)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = deyo_dp.DPDeYO(subnet, args, optimizer, deyo_margin=deyo_margin, margin_e0=deyo_margin_e0)
    elif args.algorithm == 'deyo_clip':
        subnet = deyo_clip.configure_model(subnet)
        params, param_names = deyo_clip.collect_params(subnet)
        logger.info(param_names)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = deyo_clip.DeYOClip(subnet, args, optimizer, deyo_margin=deyo_margin, margin_e0=deyo_margin_e0)
    elif args.algorithm == 'sar_dp':
        subnet = sar_dp.configure_model(subnet)
        params, param_names = sar_dp.collect_params(subnet)
        logger.info(param_names)
        base_optimizer = torch.optim.SGD
        optimizer = DPSAT(params, base_optimizer, noise_multiplier=args.noise, max_grad_norm=args.max_norm,
                          batch_size=args.batch_size, lr=args.lr, momentum=0.9)
        adapt_model = sar_dp.SARNoFilter(subnet, optimizer, margin_e0=args.sar_margin_e0)
    elif args.algorithm == 'deyo_come':
        subnet = deyo_come.configure_model(subnet)
        params, param_names = deyo_come.collect_params(subnet)
        logger.info(param_names)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = deyo_come.DeYO_COME(subnet, args, optimizer, deyo_margin=deyo_margin, margin_e0=deyo_margin_e0)
    elif args.algorithm == 'deyo_come_dp':
        subnet = deyo_come_dp.configure_model(subnet)
        params, param_names = deyo_come_dp.collect_params(subnet)
        logger.info(param_names)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = deyo_come_dp.DeYO_COME_DP(subnet, args, optimizer, deyo_margin=deyo_margin, margin_e0=deyo_margin_e0)
    elif args.algorithm == 'deyo_come_clip':
        subnet = deyo_come_clip.configure_model(subnet)
        params, param_names = deyo_come_clip.collect_params(subnet)
        logger.info(param_names)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = deyo_come_clip.DeYO_COME_Clip(subnet, args, optimizer, deyo_margin=deyo_margin, margin_e0=deyo_margin_e0)
    elif args.algorithm == 'eata':
        subnet, params, fishers = compute_fishers(subnet, args, eata.configure_model, eata.collect_params)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = eata.EATA(args, subnet, optimizer, args.lr, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)
    elif args.algorithm == 'eata_dp':
        subnet, params, fishers = compute_fishers(subnet, args, eata.configure_model, eata.collect_params)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = eata_dp.DPEATA(args, subnet, optimizer, args.lr, args.max_norm, args.noise, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)
    elif args.algorithm == 'eata_clip':
        subnet, params, fishers = compute_fishers(subnet, args, eata_clip.configure_model, eata_clip.collect_params)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = eata_clip.EATA(args, subnet, optimizer, args.lr, args.max_norm, args.noise, fishers, args.fisher_alpha,  e_margin=args.e_margin, d_margin=args.d_margin)
    elif args.algorithm == 'source':
        adapt_model = setup_source(subnet)
    else:
        assert False, NotImplementedError
    acc = 0.0
    num = 0
    print("Parameters:",args.lr,args.max_norm,args.noise)
    for corrupt in common_corruptions:
        if args.exp_type == 'each_shift_reset' and args.reset == 0:
            print("no reset")
        elif args.exp_type == 'each_shift_reset' and args.reset == 1:
            adapt_model.reset()
            print("reset")
        else:
            assert False, NotImplementedError

        args.corruption = corrupt
        logger.info(args.corruption)
        if args.corruption == 'imagenet-r':
            adapt_model.imagenet_mask = imagenet_r_mask
        elif args.corruption == 'adversial':
            adapt_model.imagenet_mask = imagenet_a_mask
        else:
            adapt_model.imagenet_mask = None
        val_dataset, val_loader = prepare_test_data(args)
        val_dataset.switch_mode(True, False)

        top1, top5 = validate(val_loader, adapt_model, None, args, mode='eval')
        acc += top1
        num += 1
        logger.info(
            f"Under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {top1:.5f} and Top-5 Accuracy: {top5:.5f}")
        if args.algorithm in ['eata', 'eta']:
            logger.info(
                f"num of reliable samples is {adapt_model.num_samples_update_1}, num of reliable+non-redundant samples is {adapt_model.num_samples_update_2}")
            adapt_model.num_samples_update_1, adapt_model.num_samples_update_2 = 0, 0
    acc = acc / num
    print(f"The Accuracy = {float(acc):.5f}")