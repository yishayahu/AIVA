from utils import load_model, get_batch, loss_calc
from model.unet import UNet2D
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import json
from metric_utils import get_sdice, get_dice
import os
from print_seg import print_seg

best_metric = -1
low_source_metric = 1.1

prev_d_score = 0


def cotraining(model_path, ref_model_path, train_loader, target_loader, val_ds, test_ds, val_ds_source, args,
               config):
    print("cotraining")
    model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
    model = load_model(model, model_path, config.msm)
    ref_model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
    ref_model = load_model(ref_model, ref_model_path, config.msm)
    torch.save(model.state_dict(), config.exp_dir / f'cotraining_best_model.pth')
    torch.save(ref_model.state_dict(), config.exp_dir / f'cotraining_best_ref_model.pth')
    model.eval()
    model.to(args.gpu)
    ref_model.eval()
    ref_model.to(args.gpu)

    if config.parallel_model:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        ref_model = torch.nn.DataParallel(ref_model, device_ids=[0, 1, 2, 3])

    train_loader.dataset.yield_id = True
    target_loader.dataset.yield_id = True
    train_loader_iter = iter(train_loader)
    target_loader_iter = iter(target_loader)
    if config.msm:
        optimizers = {"model": optim.Adam(model.parameters(),
                                          lr=1e-6, weight_decay=args.weight_decay),
                      "ref_model": optim.Adam(ref_model.parameters(),
                                              lr=1e-6, weight_decay=args.weight_decay)}
    else:
        optimizers = {"model": optim.SGD(model.parameters(), lr=1e-6),
                      "ref_model": optim.SGD(ref_model.parameters(), lr=1e-6)}
    iterations = 50
    epochs = 50
    alpha = 0
    print(f"alpha is {alpha}")
    for i in tqdm(range(iterations), "iteration", position=0, leave=True):
        print_seg(ref_model, model, config, os.getcwd(), 1, i)

        for epoch in tqdm(range(epochs), desc="epoch", position=0, leave=True):
            optimizers["model"].zero_grad()
            optimizers["ref_model"].zero_grad()

            source_train_batch = get_batch(train_loader, train_loader_iter)
            target_batch = get_batch(target_loader, target_loader_iter)

            source_train_images, source_train_labels, source_train_ids, _ = source_train_batch
            target_images, _, _, _ = target_batch

            _, org_preds = model(target_images.to(args.gpu))
            ref_labels = torch.argmax(org_preds, dim=1)
            _, ref_preds = ref_model(target_images.to(args.gpu))
            org_labels = torch.argmax(ref_preds, dim=1)

            org_loss = loss_calc(org_preds, org_labels, gpu=args.gpu)
            org_loss.backward()
            optimizers["model"].step()

            ref_loss = loss_calc(ref_preds, ref_labels, gpu=args.gpu)
            ref_loss.backward()
            optimizers["ref_model"].step()

            del source_train_images
            del target_images
            del ref_preds
            del org_preds
            del org_loss
            del ref_loss


def get_alpha(i, iterations):
    return 1 - (0.5 + ((i + 1) / iterations) if ((i + 1) / iterations) < 0.5 else 1)


def pseudo_labeling_after_step(model, config, step_num, epochs, test_ds, val_ds_source, val_ds, args):
    global best_metric
    global low_source_metric
    global prev_d_score
    if step_num % 1 == 0 and step_num != 0:
        if config.msm:
            dice1, sdice1 = get_dice(model, val_ds, args.gpu, config)
            main_metric = dice1
        else:
            dice1, sdice1 = get_sdice(model, val_ds, args.gpu, config)
            main_metric = sdice1
        if False:
            if config.msm:
                dice_source, sdice_source = get_dice(model, val_ds_source, args.gpu, config)
                main_metric_source = dice_source
            else:
                dice_source, sdice_source = get_sdice(model, val_ds_source, args.gpu, config)
                main_metric_source = sdice_source
            if main_metric_source < low_source_metric:
                low_source_metric = main_metric_source
                torch.save(model.state_dict(), config.exp_dir / f'pseudo_labeling_low_source_model.pth')
            wandb.log(
                {f'pseudo_labeling_dice/val_source': dice_source, f'pseudo_labeling_sdice/val_source': sdice_source},
                step=step_num)
        improvement = main_metric / prev_d_score
        print(f"improvement is {improvement}")
        wandb.log({f'pseudo_labeling_dice/val': dice1, f'pseudo_labeling_sdice/val': sdice1,
                   'pseudo_labels/improvement': improvement}, step=step_num)
        prev_d_score = improvement
        print(f'pseudo_labeling_dice is ', dice1)
        print(f'pseudo_labeling_sdice is ', sdice1)
        print('pseudo_labeling taking snapshot ...')

        if main_metric > best_metric:
            best_metric = main_metric
            print("new best metric!")
            torch.save(model.state_dict(), config.exp_dir / f'pseudo_labeling_best_model.pth')

        torch.save(model.state_dict(), config.exp_dir / f'pseudo_labeling_model.pth')
    if step_num == 0 or step_num == epochs - 1:

        title = 'end' if step_num != 0 else 'start'
        scores = {}
        if config.msm:
            dice_test, sdice_test = get_dice(model, test_ds, args.gpu, config)
        else:
            dice_test, sdice_test = get_sdice(model, test_ds, args.gpu, config)

        prev_d_score = sdice_test

        scores[f'pseudo_labeling_dice_{title}/test'] = dice_test
        scores[f'pseudo_labeling_sdice_{title}/test'] = sdice_test
        print(f"dice {title} is: {dice_test}")
        print(f"sdice {title} is: {sdice_test}")
        if step_num != 0:
            model.load_state_dict(torch.load(config.exp_dir / f'pseudo_labeling_best_model.pth', map_location='cpu'))
            if config.msm:
                dice_test_best, sdice_test_best = get_dice(model, test_ds, args.gpu, config)
            else:
                dice_test_best, sdice_test_best = get_sdice(model, test_ds, args.gpu, config)
            scores[f'pseudo_labeling_dice_{title}/test_best'] = dice_test_best
            scores[f'pseudo_labeling_sdice_{title}/test_best'] = sdice_test_best
            if val_ds_source is not None:
                model.load_state_dict(
                    torch.load(config.exp_dir / f'pseudo_labeling_low_source_model.pth', map_location='cpu'))
                if config.msm:
                    dice_test_low_source, sdice_test_low_source = get_dice(model, test_ds, args.gpu, config)
                else:
                    dice_test_low_source, sdice_test_low_source = get_sdice(model, test_ds, args.gpu, config)
                scores[f'pseudo_labeling_dice_{title}/test_low_source_on_target'] = dice_test_low_source
                scores[f'pseudo_labeling_sdice_{title}/test_low_source_on_target'] = sdice_test_low_source

        wandb.log(scores, step=step_num)
        json.dump(scores, open(config.exp_dir / f'pseudo_labeling_scores_{title}.json', 'w'))
