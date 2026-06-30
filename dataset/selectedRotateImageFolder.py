import os
import random
import math

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import timm
import torch
from timm.data import resolve_data_config, create_transform
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
tr_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])

te_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])



te_transforms_imageC = transforms.Compose([transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize])

ad_transforms_imageC = transforms.Compose([transforms.CenterCrop(224),
                                           transforms.ToTensor(), ])

rotation_tr_transforms = tr_transforms
rotation_te_transforms = te_transforms

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


class ImagePathFolder(datasets.ImageFolder):
    def __init__(self, traindir, train_transform):
        super(ImagePathFolder, self).__init__(traindir, train_transform)

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        path, pa = os.path.split(path)
        path, pb = os.path.split(path)
        return img, 'val/%s/%s' % (pb, pa)


def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)


def tensor_rot_180(x):
    return x.flip(2).flip(1)


def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


def rotate_single_with_label(img, label):
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    return img


def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        img = rotate_single_with_label(img, label)
        images.append(img.unsqueeze(0))
    return torch.cat(images)


def rotate_batch(batch, label='rand'):
    if label == 'rand':
        labels = torch.randint(4, (len(batch),), dtype=torch.long)
    else:
        assert isinstance(label, int)
        labels = torch.zeros((len(batch),), dtype=torch.long) + label
    return rotate_batch_with_labels(batch, labels), labels


class SelectedRotateImageFolder(datasets.ImageFolder):
    def __init__(self, root, train_transform, original=True, rotation=True, rotation_transform=None):
        super(SelectedRotateImageFolder, self).__init__(root, train_transform)
        self.original = original
        self.rotation = rotation
        self.rotation_transform = rotation_transform

        self.original_samples = self.samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        img_input = self.loader(path)

        if self.transform is not None:
            if isinstance(self.transform, list):
                img = self.transform[1](img_input)
                img_aug = self.transform[0](img_input)
            else:
                img = self.transform(img_input)
        else:
            img = img_input

        results = []
        if self.original:
            results.append(img)
            results.append(target)
            if isinstance(self.transform, list):
                results.append(img_aug)
        if self.rotation:
            if self.rotation_transform is not None:
                img = self.rotation_transform(img_input)
            target_ssh = np.random.randint(0, 4, 1)[0]
            img_ssh = rotate_single_with_label(img, target_ssh)
            results.append(img_ssh)
            results.append(target_ssh)
        return results

    def switch_mode(self, original, rotation):
        self.original = original
        self.rotation = rotation

    def set_target_class_dataset(self, target_class_index, logger=None):
        self.target_class_index = target_class_index
        self.samples = [(path, idx) for (path, idx) in self.original_samples if idx in self.target_class_index]
        self.targets = [s[1] for s in self.samples]

    def set_dataset_size(self, subset_size):
        num_train = len(self.targets)
        indices = list(range(num_train))
        random.shuffle(indices)
        self.samples = [self.samples[i] for i in indices[:subset_size]]
        self.targets = [self.targets[i] for i in indices[:subset_size]]
        return len(self.targets)

    def set_specific_subset(self, indices):
        self.samples = [self.original_samples[i] for i in indices]
        self.targets = [s[1] for s in self.samples]


def reset_data_sampler(sampler, dset_length, dset):
    sampler.dataset = dset
    if dset_length % sampler.num_replicas != 0 and False:
        sampler.num_samples = math.ceil((dset_length - sampler.num_replicas) / sampler.num_replicas)
    else:
        sampler.num_samples = math.ceil(dset_length / sampler.num_replicas)
    sampler.total_size = sampler.num_samples * sampler.num_replicas


def prepare_train_dataset(args, use_transforms=True):
    print('Preparing training data (ori imagenet train)...')
    tr_transforms_local = tr_transforms if use_transforms else None
    traindir = os.path.join(args.data, 'train')
    trset = SelectedRotateImageFolder(traindir, tr_transforms_local, original=True, rotation=args.rotation,
                                      rotation_transform=rotation_tr_transforms)
    return trset


def prepare_train_dataloader(args, trset=None, sampler=None):
    if sampler is None:
        trloader = torch.utils.data.DataLoader(trset, batch_size=args.train_batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=False)
        train_sampler = None
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trset)
        trloader = torch.utils.data.DataLoader(
            trset, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    return trloader, train_sampler


def obtain_train_loader(args):
    args.corruption = 'original'
    train_dataset, train_loader = prepare_test_data(args)
    train_dataset.switch_mode(True, False)
    return train_dataset, train_loader


_CANONICAL_MODEL_NAME = {
    'vit': 'vit_base_patch16_224',
    'convnext': 'convnext_tiny',
}
_data_config_cache = {}


def _resolve_arch(args):
    arch = getattr(args, 'arch', None)
    if arch in _CANONICAL_MODEL_NAME:
        return arch
    model = getattr(args, 'model', '') or ''
    if 'vit_base_patch16_224' in model:
        return 'vit'
    if 'convnext_tiny' in model:
        return 'convnext'
    return None


def _get_data_config(arch):
    if arch not in _data_config_cache:
        m = timm.create_model(_CANONICAL_MODEL_NAME[arch], pretrained=False)
        _data_config_cache[arch] = resolve_data_config({}, model=m)
    return _data_config_cache[arch]


_printed_transform = False


def get_test_transform(args, corruption):
    global _printed_transform
    arch = _resolve_arch(args)
    if arch is None:
        return te_transforms_imageC if corruption in common_corruptions else te_transforms

    cfg = _get_data_config(arch)
    transform = create_transform(**cfg, is_training=False)
    if not _printed_transform:
        print(f"[test-transform] model={_CANONICAL_MODEL_NAME[arch]}\n{transform}")
        _printed_transform = True
    return transform


def prepare_test_data(args, use_transforms=True):
    g = None
    if getattr(args, "seed", None) is not None:
        g = torch.Generator()
        g.manual_seed(args.seed)

    if not use_transforms:
        te_transforms_local = None
    elif args.corruption in (['original', 'imagenet-r'] + common_corruptions):
        te_transforms_local = get_test_transform(args, args.corruption)
    else:
        assert False, NotImplementedError

    if args.exp_type == 'adversial_attack':
        te_transforms_local = ad_transforms_imageC

    if not hasattr(args, 'corruption') or args.corruption == 'original':
        print('Test on the original test set')
        validdir = os.path.join(args.data, 'val')
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                          rotation_transform=rotation_te_transforms)
    elif not hasattr(args, 'corruption') or args.corruption == 'imagenet-r':
        print('Test on the imagenet-r test set')
        validdir = os.path.join(args.data_corruption)
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                                    rotation_transform=rotation_te_transforms)
    elif args.corruption in common_corruptions:
        print('Test on %s level %d' % (args.corruption, args.level))
        validdir = os.path.join(args.data_corruption, args.corruption, str(args.level))
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                          rotation_transform=rotation_te_transforms)
    else:
        raise Exception('Corruption not found!')

    if not hasattr(args, 'workers'):
        args.workers = 0
    teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True,generator=g, drop_last=True)

    return teset, teloader


te_transforms_inc = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])

def prepare_test_data_for_train(args, use_transforms=True):
    te_transforms_local = tr_transforms if use_transforms else None
    if args.corruption in common_corruptions:
        print('Test on %s level %d' % (args.corruption, args.level))
        validdir = os.path.join(args.data_corruption, args.corruption, str(args.level))
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                          rotation_transform=rotation_te_transforms)
    else:
        raise Exception('Corruption not found!')

    if not hasattr(args, 'workers'):
        args.workers = 1
    teloader = torch.utils.data.DataLoader(teset, batch_size=64, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)
    return teset, teloader