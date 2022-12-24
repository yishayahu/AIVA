import argparse
import dataclasses
import json
import random
import shutil
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import wandb
from dpipe.io import load
from torch.utils import data

from configs import *
from dataset.cc359_dataset import CC359Ds
from dataset.msm_dataset import MultiSiteMri
from model.unet import UNet2D
from utils import load_model
from postprocess import postprocess
from pretrain import pretrain
from clustering import clustering
from pseudo_labeling import pseudo_labels_iterations
from cotraining import cotraining

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    parser.add_argument("--num-workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    # lr params
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")

    parser.add_argument("--random-seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")


    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes to predict (including background).")

    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=2)

    parser.add_argument('--exp_name', default='')
    parser.add_argument('--msm', action='store_true')
    parser.add_argument("--mode", type=str, default='pretrain', help='pretrain or clustering_finetune')

    parser.add_argument("--pp_epochs", type=int, default=1000)
    parser.add_argument("--target_model_path", type=str, default='')
    parser.add_argument("--pl_iterations", type=int, default=10)
    parser.add_argument("--pl_epochs", type=int, default=100)
    parser.add_argument('--soft', action='store_true')

    return parser.parse_args()

def main():
    args = get_arguments()
    if args.msm:
        args.source = args.target
        if args.mode == 'clustering_finetune':
            config = MsmConfigFinetuneClustering()
        elif args.mode == 'pretrain':
            config = MsmPretrainConfig()
        else:
            config = MsmConfigFinetuneClustering()

    else:
        if 'debug' in args.exp_name:
            config = DebugConfigCC359()
        elif args.mode == 'clustering_finetune' or args.mode == "pseudo_labeling" or args.mode == "pseudo_no_cluster":
            config = CC359ConfigFinetuneClustering()
        elif args.mode == 'pretrain':
            config = CC359ConfigPretrain()
        else:
            config = CC359ConfigFinetuneClustering()
    if args.exp_name == '':
        args.exp_name = args.mode

    cudnn.enabled = True
    model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
    if args.soft:
        args.mode = args.mode + "_soft"
    if args.mode != 'pretrain':
        if args.exp_name != '':
            config.exp_dir = Path(config.base_res_path) / f'source_{args.source}_target_{args.target}' / args.exp_name
        else:
            config.exp_dir = Path(config.base_res_path) / f'source_{args.source}_target_{args.target}' / args.mode

        ckpt_path = Path(config.base_res_path) / f'source_{args.source}_target_{args.target}' / 'clustering_finetune' / 'best_model.pth'
        if args.mode == "adaBN":
            # ckpt_path = Path(config.base_res_path) / f'source_{args.source}_target_{args.target}' / args.mode / "pseudo_labeling_best_model.pth"
            # ckpt_path = f"/home/dsi/shaya/unsup_resres_zoom/source_2_target_1/adaBN/model.pth"
            ckpt_path = f"/home/dsi/shaya/unsup_resres_zoom/source_{args.source}_target_{args.target}/adaBN/model.pth"
        if args.mode == "jdot":
            # ckpt_path = Path(config.base_res_path) / f'source_{args.source}_target_{args.target}' / args.mode / "pseudo_labeling_best_model.pth"
            ckpt_path = f"/home/dsi/shaya/unsup_resres_zoom/source_{args.source}_target_{args.target}/jdot/best_model.pth"
        if args.mode == "adaSegNet":
            ckpt_path = f"/home/dsi/shaya/unsup_resres_zoom/source_{args.source}_target_{args.target}/their/best_model.pth"
        if args.mode == "pseudo_no_cluster":
            ckpt_path = Path(config.base_res_path) / f'source_{args.source}' / 'pretrain' / 'best_model.pth'
        model = load_model(model, ckpt_path, config.msm)
        if config.msm:
            optimizer = optim.Adam(model.parameters(),
                                   lr=config.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=config.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        if args.exp_name != '':
            config.exp_dir = Path(config.base_res_path) / f'source_{args.source}' / args.exp_name
        else:
            config.exp_dir = Path(config.base_res_path) / f'source_{args.source}' / args.mode

        if config.msm:
            optimizer = optim.Adam(model.parameters(),
                                   lr=config.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=config.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    if config.sched:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones,
                                                         gamma=config.sched_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)
    if config.exp_dir.exists():
        shutil.rmtree(config.exp_dir)

    config.exp_dir.mkdir(parents=True, exist_ok=True)
    json.dump(dataclasses.asdict(config), open(config.exp_dir / 'config.json', 'w'))
    model.train()
    if not torch.cuda.is_available():
        print('training on cpu')
        args.gpu = 'cpu'
        config.parallel_model = False
        torch.cuda.manual_seed_all(args.random_seed)

    model.to(args.gpu)
    if config.parallel_model:
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if config.msm:
        assert args.source == args.target
        source_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/train_ids.json'))
        target_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/train_ids.json'))
        val_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/val_ids.json'), yield_id=True,
                              test=True)
        val_ds_source = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/val_ids.json'), yield_id=True,
                                     test=True)
        test_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'), yield_id=True,
                               test=True)
        project = 'adaptSegUNetMsm'
    else:
        source_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/train_ids.json')[:config.data_len],
                            site=args.source)
        target_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/train_ids.json')[:config.data_len],
                            site=args.target)
        val_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'), site=args.target,
                         yield_id=True, slicing_interval=1)
        val_ds_source = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/val_ids.json'), site=args.source,
                                yield_id=True, slicing_interval=1)
        test_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'), site=args.target,
                          yield_id=True, slicing_interval=1)
        project = 'adaptSegUNet'
    project += args.mode
    if config.debug:
        wandb.init(
            project='spot3',
            id=wandb.util.generate_id(),
            name=args.exp_name,
            dir='../debug_wandb'
        )
    else:
        wandb.init(
            project=project,
            id=wandb.util.generate_id(),
            name=args.exp_name + '_' + str(args.source) + '_' + str(args.target),
            dir='..'
        )
    trainloader = data.DataLoader(source_ds, batch_size=config.source_batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=config.drop_last)
    targetloader = data.DataLoader(target_ds, batch_size=config.target_batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=config.drop_last)

    optimizer.zero_grad()

    if args.mode == 'pretrain':
        pretrain(model, optimizer, scheduler, trainloader, config, args)
    elif args.mode == 'clustering_finetune':
        clustering(model, optimizer, scheduler, trainloader, targetloader, val_ds, test_ds, val_ds_source, config, args)
    elif args.mode == "postprocess":
        souce_model_path = ckpt_path
        postprocess(souce_model_path, config, trainloader, targetloader, val_ds, test_ds, val_ds_source, args)
    # elif args.mode == "pseudo_labeling" or "pseudo_no_cluster":
    elif args.mode == "cotraining":
        ref_path = f"/home/dsi/shaya/unsup_resres_zoom/source_{args.source}_target_{args.target}/jdot/best_model.pth"
        org_path = f"/home/dsi/shaya/unsup_resres_zoom/source_{args.source}_target_{args.target}/adaBN/model.pth"
        cotraining(org_path, ref_path, trainloader, targetloader, val_ds, test_ds, val_ds_source, args, config)
    else:
        model_path = ckpt_path
        pseudo_labels_iterations(ckpt_path, trainloader, targetloader, val_ds, test_ds, val_ds_source, args, config)

if __name__ == '__main__':
    # import subprocess
    # print(subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode("utf8"))
    main()

# 0 = siemens 15
# 1 = siemens 3
# 2 = ge 15
# 3 = ge3
# 4 = philips 15
#  5 = philips 3