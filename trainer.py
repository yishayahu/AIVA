import argparse
import dataclasses
import json
import pickle
import random
import shutil
import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import wandb
from dpipe.io import load
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from configs import *
from dataset.cc359_dataset import CC359Ds
from dataset.msm_dataset import MultiSiteMri
from metric_utils import get_sdice, get_dice
from model.unet import UNet2D
from utils import adjust_learning_rate, loss_calc
from utils import freeze_model, include_patterns, tensor_to_image


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
    return parser.parse_args()


args = get_arguments()
if args.msm:
    args.source = args.target
    if args.mode == 'clustering_finetune':
        config = MsmConfigFinetuneClustering()
    elif args.mode == 'pretrain':
        config = MsmPretrainConfig()
    else:
        raise Exception(f'mode {args.mode} not exists')

else:
    if 'debug' in args.exp_name:
        config = DebugConfigCC359()
    elif args.mode == 'clustering_finetune':
        config = CC359ConfigFinetuneClustering()
    elif args.mode == 'pretrain':
        config = CC359ConfigPretrain()
    else:
        raise Exception(f'mode {args.mode} not exists')
if args.exp_name == '':
    args.exp_name = args.mode
best_metric = -1
low_source_metric = 1.1


def after_step(num_step, val_ds, test_ds, model, val_ds_source):
    global best_metric
    global low_source_metric
    if num_step % config.save_pred_every == 0 and num_step != 0:
        if config.msm:
            dice1, sdice1 = get_dice(model, val_ds, args.gpu, config)
            main_metric = dice1
        else:
            dice1, sdice1 = get_sdice(model, val_ds, args.gpu, config)
            main_metric = sdice1
        if val_ds_source is not None:
            if config.msm:
                dice_source, sdice_source = get_dice(model, val_ds_source, args.gpu, config)
                main_metric_source = dice_source
            else:
                dice_source, sdice_source = get_sdice(model, val_ds_source, args.gpu, config)
                main_metric_source = sdice_source
            if main_metric_source < low_source_metric:
                low_source_metric = main_metric_source
                torch.save(model.state_dict(), config.exp_dir / f'low_source_model.pth')
            wandb.log({f'dice/val_source': dice_source, f'sdice/val_source': sdice_source}, step=num_step)
        wandb.log({f'dice/val': dice1, f'sdice/val': sdice1}, step=num_step)
        print(f'dice is ', dice1)
        print(f'sdice is ', sdice1)
        print('taking snapshot ...')

        if main_metric > best_metric:
            best_metric = main_metric
            torch.save(model.state_dict(), config.exp_dir / f'best_model.pth')

        torch.save(model.state_dict(), config.exp_dir / f'model.pth')
    if num_step == config.num_steps - 1 or num_step == 0:
        title = 'end' if num_step != 0 else 'start'
        scores = {}
        if config.msm:
            dice_test, sdice_test = get_dice(model, test_ds, args.gpu, config)
        else:
            dice_test, sdice_test = get_sdice(model, test_ds, args.gpu, config)
        scores[f'dice_{title}/test'] = dice_test
        scores[f'sdice_{title}/test'] = sdice_test
        if num_step != 0:
            model.load_state_dict(torch.load(config.exp_dir / f'best_model.pth', map_location='cpu'))
            if config.msm:
                dice_test_best, sdice_test_best = get_dice(model, test_ds, args.gpu, config)
            else:
                dice_test_best, sdice_test_best = get_sdice(model, test_ds, args.gpu, config)
            scores[f'dice_{title}/test_best'] = dice_test_best
            scores[f'sdice_{title}/test_best'] = sdice_test_best
            if val_ds_source is not None:
                model.load_state_dict(torch.load(config.exp_dir / f'low_source_model.pth', map_location='cpu'))
                if config.msm:
                    dice_test_low_source, sdice_test_low_source = get_dice(model, test_ds, args.gpu, config)
                else:
                    dice_test_low_source, sdice_test_low_source = get_sdice(model, test_ds, args.gpu, config)
                scores[f'dice_{title}/test_low_source_on_target'] = dice_test_low_source
                scores[f'sdice_{title}/test_low_source_on_target'] = sdice_test_low_source

        wandb.log(scores, step=num_step)
        json.dump(scores, open(config.exp_dir / f'scores_{title}.json', 'w'))


def train_pretrain(model, optimizer, scheduler, trainloader):
    if config.msm:
        val_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/val_ids.json'), yield_id=True,
                              test=True)
        test_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/test_ids.json'), yield_id=True,
                               test=True)
    else:
        val_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/val_ids.json'), site=args.source,
                         yield_id=True, slicing_interval=1)
        test_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/test_ids.json'), site=args.source,
                          yield_id=True, slicing_interval=1)
    trainloader_iter = iter(trainloader)
    for i_iter in range(config.num_steps):
        model.train()
        loss_seg_value = 0
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, config, args)

        # train with source
        try:
            batch = trainloader_iter.next()
        except StopIteration:
            trainloader_iter = iter(trainloader)
            batch = trainloader_iter.next()

        images, labels = batch
        images = Variable(images).to(args.gpu)

        _, pred = model(images)
        loss_seg = loss_calc(pred, labels, args.gpu)
        loss = loss_seg
        # proper normalization

        loss.backward()
        loss_seg_value += loss_seg.data.cpu().numpy()

        optimizer.step()
        scheduler.step()

        print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f}'.format(
                i_iter, config.num_steps, loss_seg_value))
        after_step(i_iter, model=model, val_ds=val_ds, test_ds=test_ds, val_ds_source=None)


def get_best_match_aux(distss):
    n_clusters = len(distss)
    print('n_clusterss', n_clusters)
    res = linear_sum_assignment(distss)[1].tolist()
    targets = [None] * n_clusters
    for x, y in enumerate(res):
        targets[y] = x
    return targets


def get_best_match(sc, tc):
    dists = np.full((sc.shape[0], tc.shape[0]), fill_value=np.inf)
    for i in range(sc.shape[0]):
        for j in range(tc.shape[0]):
            dists[i][j] = np.mean((sc[i] - tc[j]) ** 2)
    best_match = get_best_match_aux(dists.copy())

    return best_match


def train_clustering(model, optimizer, scheduler, trainloader, targetloader, val_ds, test_ds, val_ds_source):
    freeze_model(model, exclude_layers=['init_path', 'down', 'bottleneck.0', 'bottleneck.1', 'bottleneck.2',
                                        'bottleneck.3.conv_path.0', 'out_path'])
    trainloader.dataset.yield_id = True
    targetloader.dataset.yield_id = True
    trainloader_iter = iter(trainloader)
    targetloader_iter = iter(targetloader)
    dist_loss_lambda = config.dist_loss_lambda
    dist_loss_normalization = None
    n_clusters = config.n_clusters
    slice_to_cluster = None
    source_clusters = None
    target_clusters = None
    best_matchs = None
    best_matchs_indexes = None
    accumulate_for_loss = None
    if config.use_accumulate_for_loss:
        accumulate_for_loss = []
        for _ in range(n_clusters):
            accumulate_for_loss.append([])
    slice_to_feature_source = {}
    slice_to_feature_target = {}
    id_to_num_slices = load(config.id_to_num_slices)
    epoch_seg_loss = []
    epoch_dist_loss = []
    optimizer.zero_grad()
    for i_iter in tqdm(range(config.num_steps)):
        if config.use_adjust_lr:
            adjust_learning_rate(optimizer, i_iter, config, args)
        if i_iter == 0:
            if config.parallel_model:
                model.module.get_bottleneck = False
            else:
                model.get_bottleneck = False
            after_step(i_iter, val_ds, test_ds, model, val_ds_source)
            continue
        if config.parallel_model:
            model.module.get_bottleneck = True
        else:
            model.get_bottleneck = True
        if best_matchs is None:
            model.eval()
        else:
            model.train()
        if i_iter % config.epoch_every == 0 and i_iter != 0:
            trainloader_iter = iter(trainloader)
            targetloader_iter = iter(targetloader)
            source_clusters = []
            target_clusters = []
            if config.use_accumulate_for_loss:
                accumulate_for_loss = []
                for _ in range(n_clusters):
                    accumulate_for_loss.append([])
            for i in range(n_clusters):
                source_clusters.append([])
                target_clusters.append([])
            p = PCA(n_components=20, random_state=42)
            t = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42)
            points = []
            slices = []
            for id_slc, feat in slice_to_feature_source.items():
                points.append(feat)
                id1, slc_num = id_slc.split('_')
                if config.msm:
                    slc_num = int(slc_num) / id_to_num_slices[id1]
                else:
                    slc_num = int(slc_num) / id_to_num_slices[id1]
                slices.append(slc_num)
            for id_slc, feat in slice_to_feature_target.items():
                points.append(feat)
                id1, slc_num = id_slc.split('_')
                if config.msm:
                    slc_num = int(slc_num) / id_to_num_slices[id1]
                else:
                    slc_num = int(slc_num) / id_to_num_slices[id1]
                slices.append(slc_num)
            points = np.array(points)
            points = points.reshape(points.shape[0], -1)
            print('doing tsne')
            points = p.fit_transform(points)
            if config.use_slice_num:
                slices = np.expand_dims(np.array(slices), axis=1)
                points = np.concatenate([points, slices], axis=1)
            points = t.fit_transform(points)
            source_points, target_points = points[:len(slice_to_feature_source)], points[len(slice_to_feature_source):]
            # source_points,target_points = points[:max(len(slice_to_feature_source),n_clusters)],points[-max(len(slice_to_feature_target),n_clusters):]
            k1 = KMeans(n_clusters=n_clusters, random_state=42)
            print('doing kmean 1')
            sc = k1.fit_predict(source_points)
            k2 = KMeans(n_clusters=n_clusters, random_state=42, init=k1.cluster_centers_)
            print('doing kmean 2')
            tc = k2.fit_predict(target_points)
            print('getting best match')
            best_matchs_indexes = get_best_match(k1.cluster_centers_, k2.cluster_centers_)
            slice_to_cluster = {}
            items = list(slice_to_feature_source.items())
            for i in range(len(slice_to_feature_source)):
                source_clusters[sc[i]].append(items[i][1])
                slice_to_cluster[items[i][0]] = sc[i]
            items = list(slice_to_feature_target.items())
            for i in range(len(slice_to_feature_target)):
                slice_to_cluster[items[i][0]] = tc[i]
            for i in range(len(source_clusters)):
                source_clusters[i] = np.mean(source_clusters[i], axis=0)
            best_matchs = []
            for i in range(len(best_matchs_indexes)):
                best_matchs.append(torch.tensor(source_clusters[best_matchs_indexes[i]]))

            colors = ['black', 'blue', 'cyan', 'red', 'orange'
                , 'tomato', 'lime', 'gold', 'magenta', 'dodgerblue'
                , 'peru', 'grey', 'brown', 'olive', 'navy'
                , 'blueviolet', 'darkgreen', 'maroon', 'yellow', 'cadetblue']
            im_path_source = str(config.exp_dir / f'{i_iter}_source.png')
            fig = plt.figure()
            ax = fig.add_subplot()
            curr_colors = []
            curr_points_x = []
            curr_points_y = []
            for i, slc_name in enumerate(slice_to_feature_source.keys()):
                curr_points_x.append(source_points[i][0])
                curr_points_y.append(source_points[i][1])
                curr_colors.append(colors[slice_to_cluster[slc_name]])
            ax.scatter(curr_points_x, curr_points_y, marker='.', c=curr_colors)
            plt.savefig(im_path_source)
            plt.cla()
            plt.clf()
            plt.close()
            im_path_target = str(config.exp_dir / f'{i_iter}_target.png')

            fig = plt.figure()
            ax = fig.add_subplot()
            curr_colors = []
            curr_points_x = []
            curr_points_y = []
            for i, slc_name in enumerate(slice_to_feature_target.keys()):
                curr_points_x.append(target_points[i][0])
                curr_points_y.append(target_points[i][1])
                curr_colors.append(colors[best_matchs_indexes[slice_to_cluster[slc_name]]])
            ax.scatter(curr_points_x, curr_points_y, marker='.', c=curr_colors)
            plt.savefig(im_path_target)
            plt.cla()
            plt.clf()
            plt.close()
            im_path_clusters = str(config.exp_dir / f'{i_iter}_clusters.png')
            fig = plt.figure()
            ax = fig.add_subplot()
            pickle.dump(k1.cluster_centers_, open(config.exp_dir / f'source_cluster_centers_{i_iter}.p', 'wb'))
            pickle.dump(k2.cluster_centers_, open(config.exp_dir / f'target_cluster_centers_{i_iter}.p', 'wb'))
            for i, (p, marker) in enumerate([(k1.cluster_centers_, '.'), (k2.cluster_centers_, '^')]):
                if i == 0:
                    ax.scatter(p[:, 0], p[:, 1], marker=marker, c=colors[:len(p)])
                else:
                    ax.scatter(p[:, 0], p[:, 1], marker=marker,
                               c=[colors[best_matchs_indexes[i]] for i in range(len(p))])
            plt.savefig(im_path_clusters)
            plt.cla()
            plt.clf()
            plt.close()
            slice_to_feature_source = {}
            slice_to_feature_target = {}
            vizviz = {}
            log_log = {f'figs/source': wandb.Image(im_path_source), f'figs/target': wandb.Image(im_path_target),
                       f'figs/cluster': wandb.Image(im_path_clusters)}
            wandb.log(log_log, step=i_iter)
        log_log = {}

        try:
            train_batch = trainloader_iter.next()
        except StopIteration:
            trainloader_iter = iter(trainloader)
            train_batch = trainloader_iter.next()

        train_images, labels, train_ids, train_slice_nums = train_batch
        train_images = Variable(train_images).to(args.gpu)

        _, pred, train_features = model(train_images)
        train_features = train_features.detach().cpu().numpy()
        for train_id1, train_slc_num, train_feature, train_img in zip(train_ids, train_slice_nums, train_features, train_images):
            slice_to_feature_source[f'{train_id1}_{train_slc_num}'] = train_feature
            if best_matchs is not None and f'{train_id1}_{train_slc_num}' in slice_to_cluster:
                src_cluster = slice_to_cluster[f'{train_id1}_{train_slc_num}']
                if f'source_{src_cluster}' not in vizviz or len(vizviz[f'source_{src_cluster}']) < 4:
                    if f'source_{src_cluster}' not in vizviz:
                        vizviz[f'source_{src_cluster}'] = []
                    vizviz[f'source_{src_cluster}'].append(None)
                    im_path = str(
                        config.exp_dir / f'source_{src_cluster}_{i_iter}_{len(vizviz[f"source_{src_cluster}"])}.png')
                    if train_img.shape[0] == 3:
                        plt.imsave(im_path, np.array(train_img[1].detach().cpu()), cmap='gray')
                    else:
                        train_img = tensor_to_image(train_img)
                        train_img.save(im_path)
                    log_log[f'{src_cluster}/source_{len(vizviz[f"source_{src_cluster}"])}'] = wandb.Image(im_path)
            loss_seg = loss_calc(pred, labels, args.gpu)
            loss = loss_seg
            # proper normalization

            try:
                traget_batch = targetloader_iter.next()
            except StopIteration:
                targetloader_iter = iter(targetloader)
                traget_batch = targetloader_iter.next()
            traget_images, _, ids, slice_nums = traget_batch
            traget_images = Variable(traget_images).to(args.gpu)

            _, __, traget_features = model(traget_images)
            # features = features.mean(1)
            dist_loss = torch.tensor(0.0, device=args.gpu)
            for id1, slc_num, traget_feature, traget_img in zip(ids, slice_nums, traget_features, traget_images):
                slice_to_feature_target[f'{id1}_{slc_num}'] = traget_feature.detach().cpu().numpy()
                if best_matchs is not None and f'{id1}_{slc_num}' in slice_to_cluster:
                    if config.use_accumulate_for_loss:
                        accumulate_for_loss[slice_to_cluster[f'{id1}_{slc_num}']].append(traget_feature)
                    else:
                        dist_loss += torch.mean(
                            torch.abs(traget_feature - best_matchs[slice_to_cluster[f'{id1}_{slc_num}']].to(args.gpu)))
                    src_cluster = best_matchs_indexes[slice_to_cluster[f'{id1}_{slc_num}']]
                    if f'target_{src_cluster}' not in vizviz or len(vizviz[f'target_{src_cluster}']) < 4:
                        if f'target_{src_cluster}' not in vizviz:
                            vizviz[f'target_{src_cluster}'] = []
                        vizviz[f'target_{src_cluster}'].append(None)
                        im_path = str(
                            config.exp_dir / f'target_{src_cluster}_{i_iter}_{len(vizviz[f"target_{src_cluster}"])}.png')
                        if traget_img.shape[0] == 3:
                            plt.imsave(im_path, np.array(traget_img[1].detach().cpu()), cmap='gray')
                        else:
                            traget_img = tensor_to_image(traget_img)
                            traget_img.save(im_path)
                        log_log[f'{src_cluster}/target_{len(vizviz[f"target_{src_cluster}"])}'] = wandb.Image(im_path)
            if accumulate_for_loss is not None:
                use_dist_loss = False
                lens1 = [len(x) for x in accumulate_for_loss]
                if np.sum(lens1) > config.acc_amount:
                    use_dist_loss = True
                if use_dist_loss:
                    total_amount = 0
                    dist_losses = [0] * len(accumulate_for_loss)
                    for i, accu_features in enumerate(accumulate_for_loss):
                        if len(accu_features) > 0:
                            curr_amount = len(accu_features)
                            total_amount += curr_amount
                            accu_features = torch.mean(torch.stack(accu_features), dim=0)
                            dist_losses[i] = torch.mean((accu_features - best_matchs[i].to(args.gpu)) ** 2) * curr_amount
                            accumulate_for_loss[i] = []
                    for l in dist_losses:
                        if l > 0:
                            dist_loss += l
                    dist_loss /= total_amount
            if dist_loss_normalization is not None:
                dist_loss /= dist_loss_normalization
                dist_loss *= dist_loss_lambda
            if float(dist_loss) > 0:
                epoch_dist_loss.append(float(dist_loss))
                if dist_loss_normalization is None and len(epoch_dist_loss) > 5:
                    dist_loss_normalization = np.mean(epoch_dist_loss)
                    epoch_dist_loss = []
                    print(f'dist loss n is :{dist_loss_normalization}')
            epoch_seg_loss.append(float(loss))
            losses_dict = {'seg_loss': loss, 'dist_loss': dist_loss, 'total': loss + dist_loss}
            if accumulate_for_loss is None:
                losses_dict['total'].backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pred.detach()
                _, pred, _ = model(train_images)
            else:
                if use_dist_loss:
                    losses_dict['total'].backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    pred.detach()
                    _, pred, _ = model(train_images)
                elif best_matchs is None:
                    pass
                else:
                    losses_dict['seg_loss'].backward(retain_graph=True)
                    scheduler.step()
            log_log['seg_loss'] = float(np.mean(epoch_seg_loss))
            if epoch_dist_loss:
                log_log['dist_loss'] = float(np.mean(epoch_dist_loss))
            log_log['lr'] = float(scheduler.get_last_lr()[0])
            wandb.log(log_log, step=i_iter)

        if config.parallel_model:
            model.module.get_bottleneck = False
        else:
            model.get_bottleneck = False
        after_step(i_iter, val_ds, test_ds, model, val_ds_source)


def main():
    """Create the model and start the training."""
    cudnn.enabled = True
    model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)

    if args.mode != 'pretrain':
        if args.exp_name != '':
            config.exp_dir = Path(config.base_res_path) / f'source_{args.source}_target_{args.target}' / args.exp_name
        else:
            config.exp_dir = Path(config.base_res_path) / f'source_{args.source}_target_{args.target}' / args.mode

        ckpt_path = Path(config.base_res_path) / f'source_{args.source}' / 'pretrain' / 'best_model.pth'
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if config.msm:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                new_state_dict[k] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
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
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
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
        train_pretrain(model, optimizer, scheduler, trainloader)
    elif args.mode == 'clustering_finetune':
        train_clustering(model, optimizer, scheduler, trainloader, targetloader, val_ds, test_ds, val_ds_source)


if __name__ == '__main__':
    main()
