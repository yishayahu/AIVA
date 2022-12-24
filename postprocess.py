import torch
import torch.optim as optim
from torch.autograd import Variable
import json
import math
from tqdm import tqdm
import numpy as np
import wandb
from model.unet import UNet2D
from utils import load_model, loss_calc, get_batch
from metric_utils import get_sdice, get_dice

best_metric = -1
low_source_metric = 1.1


def after_step(model, config, step_num, epochs, test_ds, val_ds_source, val_ds, args):
    global best_metric
    global low_source_metric
    if step_num % 200 == 0 and step_num != 0:
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
                torch.save(model.state_dict(), config.exp_dir / f'postprocess_low_source_model.pth')
            wandb.log({f'postprocess_dice/val_source': dice_source, f'postprocess_sdice/val_source': sdice_source},
                      step=step_num)
        wandb.log({f'postprocess_dice/val': dice1, f'postprocess_sdice/val': sdice1}, step=step_num)
        print(f'postprocess_dice is ', dice1)
        print(f'postprocess_sdice is ', sdice1)
        print('postprocess taking snapshot ...')

        if main_metric > best_metric:
            best_metric = main_metric
            torch.save(model.state_dict(), config.exp_dir / f'postprocess_best_model.pth')

        torch.save(model.state_dict(), config.exp_dir / f'postprocess_model.pth')
    if step_num == epochs - 1 or step_num == 0:
        title = 'end' if step_num != 0 else 'start'
        scores = {}
        if config.msm:
            dice_test, sdice_test = get_dice(model, test_ds, args.gpu, config)
        else:
            dice_test, sdice_test = get_sdice(model, test_ds, args.gpu, config)
        scores[f'postprocess_dice_{title}/test'] = dice_test
        scores[f'postprocess_sdice_{title}/test'] = sdice_test
        if step_num != 0:
            model.load_state_dict(torch.load(config.exp_dir / f'postprocess_best_model.pth', map_location='cpu'))
            if config.msm:
                dice_test_best, sdice_test_best = get_dice(model, test_ds, args.gpu, config)
            else:
                dice_test_best, sdice_test_best = get_sdice(model, test_ds, args.gpu, config)
            scores[f'postprocess_dice_{title}/test_best'] = dice_test_best
            scores[f'postprocess_sdice_{title}/test_best'] = sdice_test_best
            if val_ds_source is not None:
                model.load_state_dict(
                    torch.load(config.exp_dir / f'postprocess_low_source_model.pth', map_location='cpu'))
                if config.msm:
                    dice_test_low_source, sdice_test_low_source = get_dice(model, test_ds, args.gpu, config)
                else:
                    dice_test_low_source, sdice_test_low_source = get_sdice(model, test_ds, args.gpu, config)
                scores[f'postprocess_dice_{title}/test_low_source_on_target'] = dice_test_low_source
                scores[f'postprocess_sdice_{title}/test_low_source_on_target'] = sdice_test_low_source

        wandb.log(scores, step=step_num)
        json.dump(scores, open(config.exp_dir / f'postprocess_scores_{title}.json', 'w'))


def postprocess(source_model_path, config, trainloader, targetloader, val_ds, test_ds, val_ds_source, args):
    print("starting postprocess")
    target_model_path = args.target_model_path
    epochs = args.pp_epochs

    source_model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
    source_model = load_model(source_model, source_model_path, config.msm)
    source_model.train()
    source_model.to(args.gpu)
    if config.parallel_model:
        source_model = torch.nn.DataParallel(source_model, device_ids=[2, 0, 1, 3])

    target_model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
    target_model = load_model(target_model, target_model_path, config.msm)
    target_model.eval()

    trainloader.dataset.yield_id = True
    targetloader.dataset.yield_id = True
    trainloader_iter = iter(trainloader)
    targetloader_iter = iter(targetloader)
    epoch_seg_loss = []
    optimizer = optim.SGD(source_model.parameters(),
                          lr=1e-3, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    for epoch in tqdm(range(epochs)):
        log_log = {}

        optimizer.zero_grad()
        source_train_batch = get_batch(trainloader, trainloader_iter)
        target_batch = get_batch(targetloader, targetloader_iter)

        n_target = math.ceil((epoch / epochs) * config.source_batch_size)
        n_source = config.source_batch_size - n_target

        print(f"n_target: {n_target}")
        print(f"n_source: {n_source}")

        source_train_images, source_train_labels, source_train_ids, source_train_slice_nums = source_train_batch
        source_train_images = source_train_images.to(args.gpu)
        target_images, _, target_ids, target_slice_nums = target_batch

        selected_target_images = target_images[:n_target]
        __, target_labels = target_model(selected_target_images)
        target_labels = torch.argmax(target_labels, dim=1)
        target_labels = target_labels[:n_target].to(args.gpu)
        selected_target_images = selected_target_images.to(args.gpu)

        selected_source_train_images = source_train_images[:n_source]
        selected_source_train_labels = source_train_labels[:n_source]

        new_images = torch.cat((selected_source_train_images, selected_target_images))
        new_images = Variable(new_images).to(args.gpu)
        print(f"source_train_images: {source_train_images.size()}")
        print(f"selected_source_train_labels: {selected_source_train_labels.size()}")
        print(f"target_labels: {target_labels.size()}")

        del source_train_labels
        del source_train_images
        del target_images

        selected_source_train_labels = selected_source_train_labels.to(args.gpu)
        target_labels = target_labels.to(args.gpu)
        labels = torch.cat((selected_source_train_labels, target_labels))

        _, preds = source_model(new_images)
        loss_seg = loss_calc(preds, labels, args.gpu)
        epoch_seg_loss.append(float(loss_seg))
        loss_seg.backward()
        optimizer.step()

        del preds
        del labels
        del new_images
        del target_labels
        del selected_source_train_labels

        after_step(source_model, config, epoch, epochs, test_ds, val_ds_source, val_ds, args)

        log_log['postprocess_seg_loss'] = float(np.mean(epoch_seg_loss))
        wandb.log(log_log, step=epoch)
