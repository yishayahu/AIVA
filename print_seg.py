import os

import torch
from torch.autograd import Variable
from dpipe.io import load
from dataset.cc359_dataset import CC359Ds
from dataset.msm_dataset import MultiSiteMri
from utils import load_model
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from configs import CC359ConfigPretrain
from model.unet import UNet2D
import seaborn as sns
import matplotlib.colors as mcolors


def print_seg(model, pre_model, config, save_path, site, ind=0):
    i = 0
    save_gt = ind == 0
    ind += 42
    if ind < 100:
        ind = "00" + str(ind) if ind < 10 else "0" + str(ind)
    if config.msm:
        test_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{site}t/test_ids.json'), yield_id=True,
                               test=True)
    else:
        test_ds = CC359Ds(load(f'{config.base_splits_path}/site_{site}/train_ids.json'), site=site,
                          yield_id=True, slicing_interval=1)
    test_loader = data.DataLoader(test_ds, batch_size=115, shuffle=False)
    test_loader.dataset.yield_id = True
    test_iter = iter(test_loader)
    for i_iter in range(30):
        # model.eval()
        pre_model.eval()
        model.eval()
        if i > 200:
            gpu = 5
            pre_model.to(gpu)
            model.to(gpu)
            break

        try:
            batch = test_iter.next()
        except StopIteration:
            test_iter = iter(test_loader)
            batch = test_iter.next()

        images, labels, _, __ = batch
        images = Variable(images)

        pre_model.to("cpu")
        model.to("cpu")
        _, pre_pred = pre_model(images)
        _, pred = model(images)

        pre_res = torch.argmax(pre_pred, dim=1)
        res = torch.argmax(pred, dim=1)

        for image, segmentation, pre_segmentation, label in zip(images, res, pre_res, labels):

            p = str(i)
            if i < 100:
                if i < 10:
                    p = "00" + p
                else:
                    p = "0" + p
            parent_path = f'{save_path}/covisual/{p}'
            if not os.path.exists(parent_path):
                os.mkdir(parent_path)

            ref_path = f"{parent_path}/ref"
            org_path = f"{parent_path}/org"

            im = np.array(image[0].detach().cpu())
            if save_gt:
                lab = np.array(label[0].detach().cpu()).astype(bool)
                lab = np.where(lab == 1, 1, np.nan)
                fig, ax = plt.subplots()
                plt.title(f"{i}")
                plt.axis("off")
                ax.imshow(im, cmap='gray')
                ax.imshow(lab, cmap='summer', alpha=0.60)
                fig.show()
                if not os.path.exists(parent_path):
                    os.mkdir(parent_path)
                fig.savefig(f'{parent_path}/gt.png', bbox_inches="tight")
                plt.clf()
            print(i)

            pre_seg = np.array(pre_segmentation.detach().cpu()).astype(bool)
            pre_seg = np.where(pre_seg == 1, 1, np.nan)
            fig, ax = plt.subplots()
            ax.imshow(im, cmap='gray')
            fig.patch.set_alpha(0.0)
            plt.title(f"{i}")
            plt.axis("off")
            ax.imshow(pre_seg, cmap='winter', alpha=0.60)
            fig.show()
            if not os.path.exists(org_path):
                os.mkdir(org_path)
            fig.savefig(f'{org_path}/{ind}.png', bbox_inches="tight")
            plt.clf()


            seg = np.array(segmentation.detach().cpu()).astype(bool)
            seg = np.where(seg == 1, 1, np.nan)
            fig, ax = plt.subplots()
            ax.imshow(im, cmap='gray')
            fig.patch.set_alpha(0.0)
            plt.title(f"{i}")
            plt.axis("off")
            ax.imshow(seg, cmap='autumn', alpha=0.60)
            fig.show()
            if not os.path.exists(ref_path):
                os.mkdir(ref_path)
            fig.savefig(f'{ref_path}/{ind}.png', bbox_inches="tight")
            plt.clf()
            i += 1


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    print(os.getcwd())
    config = CC359ConfigPretrain()
    site = 1
    # model_path = "/home/dsi/shaya/tomer/CC359_results/source_2_target_1/pseudo_labeling/pseudo_labeling_best_model.pth"
    # pre_model_path = "/home/dsi/shaya/tomer/CC359_results/source_2_target_1/clustering_finetune/best_model.pth"
    pre_model_path = f"/home/dsi/shaya/unsup_resres_zoom/source_2_target_1/adaBN/model.pth"
    model_path = "/home/dsi/shaya/tomer/CC359_results/source_2_target_1/adaBN/pseudo_labeling_best_model.pth"
    save_path = os.getcwd()
    model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
    pre_model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
    model = load_model(model, model_path, config.msm)
    pre_model = load_model(pre_model, pre_model_path, config.msm)
    print_seg(model, pre_model, config, save_path, site)
