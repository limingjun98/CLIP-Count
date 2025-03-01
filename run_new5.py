# -*- coding : utf-8 -*-
# @FileName  : run.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Aug 13, 2023
# @Github    : https://github.com/songrise
# @Description: script to train and test CLIP-Count
#supress torchvision warnings
# try to add teacher-student EMA
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import numpy as np
import os
import random
from pathlib import Path
import math
from PIL import Image
from models.contrastive_loss import ContrastiveLoss
import torch
import torch.nn.functional as F
from typing import List, Dict, Any
import clip

import util.misc as misc
from util.FSC147 import  FSC147
from util.CARPK import CARPK
from util.ShanghaiTech import ShanghaiTech
import util
from models import  clip_count
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
import einops
import cv2 
import gradio as gr
import torchvision.transforms.functional as TF
from util.constant import SCALE_FACTOR

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP-Count', add_help=False)
    parser.add_argument("--mode",type = str, default = "train", choices = ["train", "test", "app"], help = "train or test or an interactive application")
    parser.add_argument("--exp_name",type = str, default = "exp", help = "experiment name")
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    # Model parameters

    parser.add_argument('--backbone', default="b16", choices=["b16", "b32", "l14"], 
                    type=str, help = "backbone of clip")
    parser.add_argument('--decoder_depth', default=4, type=int, help='Number of FIM layers')
    parser.add_argument('--decoder_head', default=8, type=int, help='Number of attention heads for FIM')

    parser.add_argument('--use_mixed_fim', default=True, type = misc.str2bool, help = "whether to use hierarchical patch-text interaction")
    parser.add_argument('--unfreeze_vit', default=False, type = misc.str2bool, help = "whether to unfreeze CLIP vit i.e., finetune CLIP")
    parser.add_argument('--use_fim', default=False, type = misc.str2bool, help = "whether to use naive interaction")
    
    #contrastive loss related
    parser.add_argument('--use_coop',  default=True, type = misc.str2bool,
                        help='whether to perform context learning for text prompts.')
    parser.add_argument('--coop_width', default = 2, type = int, help = "width of context (how many token to be learned)")
    parser.add_argument('--coop_require_grad', default = False, type = misc.str2bool, help = "whether to require grad for context learning")
    parser.add_argument('--use_vpt', default=True, type = misc.str2bool,
                        help='whether to perform visual prompt learning.')
    parser.add_argument('--vpt_width', default = 20, type = int, help = "width of visual prompt (how many token each layer)")
    parser.add_argument('--vpt_depth', default = 10, type = int, help = "depth of visual prompt (how many layer)")

    parser.add_argument("--use_contrast", default=True, type = misc.str2bool, help = "whether to use contrasitive loss")
    parser.add_argument("--w_contrast", default = 1.0, type = float, help = "weight of contrastive loss")
    parser.add_argument("--noise_text_ratio", default = 0.0, type = float, help = "ratio of noise text")
    parser.add_argument('--normalize_contrast',default=False, type = misc.str2bool, help = "whether to normalize contrastive loss")
    parser.add_argument('--contrast_pos', default = "pre", choices = ["pre", "post"], type = str, help = "Use contrastive loss before or after the interaction")
    parser.add_argument('--contrast_pre_epoch', default = 20, type = int, help = "how many epoch to use contrastive pretraining")

    # self supervise loss related -lmj
    parser.add_argument('--test_method', default = 0, type = int,
                        help="0 means normal, 1 means crop, 2 means crop and classify")
    parser.add_argument('--use_self_supervised', default=True, type=misc.str2bool,
                        help = "whether to use self supervised")
    parser.add_argument('--self_supervised_epoch', default = 20, type = int,
                        help = "how many epoch to use self supervised loss finetuned")
    parser.add_argument('--resume_checkpoint', default=False, type=misc.str2bool,
                        help="whether to resume checkpoint.If resuming from mid-epoch checkpoint, training will start from the beginning of the next epoch")
    parser.add_argument("--consistency_factor", default=2.0, type=float, help="the consistency factor of no people block")
    parser.add_argument("--teacher_decay", default=0.99, type=float,
                        help="the teacher decay rate during each step")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')


    # Dataset parameters
    parser.add_argument('--data_path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--dataset_type', default="ShanghaiTech", type = str, choices=["FSC","CARPK", "COCO", "ShanghaiTech"])

    parser.add_argument('--output_dir', default='./out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=1, type=int)


    parser.add_argument('--ckpt', default='epoch=209-val_mae=16.60.ckpt', type = str,
                        help='path of resume from checkpoint')
    # parser.add_argument('--ckpt', default=None, type=str,
    #                     help='path of resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    # log related
    parser.add_argument('--log_dir', default='./out',
                        help='path where to tensorboard log')
    parser.add_argument('--log_test_img', default=True, type=bool, help="whehter to log overlaied density map when validation and testing.")
    parser.add_argument('--dont_log', action='store_true', help='do not log to tensorboard')
    parser.add_argument('--val_freq', default=1, type=int, help='check validation every val_freq epochs')




    #log setup
    parser.add_argument('--exp_note', default = "", type = str, help = "experiment note")
    return parser


class Model(LightningModule):
    def __init__(self, args, all_classes:List[str] = None):
        super().__init__()
        self.args = args

        # if args is a dictionary, convert to Namespace
        if self.args is not None and type(self.args) is dict:
            self.args = argparse.Namespace(**self.args)
        self.all_classes = all_classes

        self.save_hyperparameters(args)
        self.model = clip_count.CLIPCount(
                        fim_depth=self.args.decoder_depth,
                        fim_num_heads=self.args.decoder_head,
                        use_coop=self.args.use_coop, 
                        use_vpt=self.args.use_vpt,
                        coop_width=self.args.coop_width,
                        vpt_width=self.args.vpt_width,
                        vpt_depth= self.args.vpt_depth,
                        backbone = self.args.backbone,
                        use_fim = self.args.use_fim,
                        use_mixed_fim = self.args.use_mixed_fim,
                        unfreeze_vit = self.args.unfreeze_vit,
                        )
        self.model.to('cuda')
        self.loss = F.mse_loss
        self.contrastive_loss = ContrastiveLoss(0.07,self.args.noise_text_ratio, self.args.normalize_contrast)
        self.neg_prompt_embed = None
        # self.consistency_factor = 2.
        self.shadow = {} # record teacher average weight parameter
        self.backup = {} # record student backup parameter

    def register_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                pass

    def update_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.args.teacher_decay) * param.data + self.args.teacher_decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


    def training_step(self, batch, batch_idx):
        if self.args.use_self_supervised:
            self.update_params()
            assert self.args.dataset_type == "ShanghaiTech", "should use ShanghaiTech when self-supervising"
            samples, gt_cnt, origin_img_tensor, im_name = batch
            # image_crop = samples[:,:,:,:384]
            # gt_cnt = gt_cnt.item()
            prompt = ["people"] # [1]
            pseudo_list = []
            inference_list = []
            for h_ in range(origin_img_tensor.shape[0]): # batch_size(32) loop
                slides, _, mini_patches = misc.sliding_window_origin_image(origin_img_tensor[h_:h_+1])
                slides = torch.from_numpy(slides).float().to(self.device)  # [N, 3, 384, 384]/[N, 3, 1536, 1536]
                prompt_big_img = np.repeat(prompt, slides.shape[0], axis=0)  # [N]
                output_big_img = self.model(slides, prompt_big_img, coop_require_grad =  True)  # [N, 384, 384]
                mini_patches = torch.from_numpy(mini_patches).float().to(self.device)
                avg_pooling = torch.nn.AvgPool2d(2)
                prompt_mini_patch = np.repeat(prompt, mini_patches.shape[1], axis=0) # [16]
                pool_tensor_cat_list = []
                bigimg_crop_pool_tensor_cat_list = []

                self.apply_shadow_params()
                self.model.eval()

                for i in range(mini_patches.shape[0]):  # [N, 16, 3, 384, 384]
                    with torch.no_grad():
                        mini_patch_output, extra_out = self.model(mini_patches[i], prompt_mini_patch,
                                                                  return_extra=True)  # [16, 384, 384]
                    img_embedding = extra_out['x_cls']  # [16, 1, 512]
                    classify_prompt = ['heads', 'background', 'tree', 'leaves', 'sky', 'building']
                    classify_prompt = [f"a photo of {p}" for p in classify_prompt]
                    with torch.no_grad():
                        text_token = clip.tokenize(classify_prompt).to(samples.device)
                        text_embedding = self.model.text_encoder_classfiy(text_token).float()  # [6, 1, 512]
                    text_embedding.squeeze_(1)  # [6, 512]
                    text_embedding.unsqueeze_(0)  # [1, 6, 512]
                    sim_map = F.cosine_similarity(img_embedding, text_embedding, dim=-1)  # [16, 6]
                    sim_map_max_index = sim_map.argmax(dim=1)  # [16]
                    pool_tensor_list = [] # [[1, 96, 96]]
                    bigimg_crop_pool_tensor_list = [] # [[1, 96, 96]]
                    for j in range(mini_patch_output.shape[0]): # 16 loop
                        mini_patch_pool_tensor = mini_patch_output[j].detach().unsqueeze(0)
                        mini_patch_pool_tensor = avg_pooling(mini_patch_pool_tensor) * 4
                        mini_patch_pool_tensor = avg_pooling(mini_patch_pool_tensor) * 4
                        mini_patch_pool_tensor.squeeze_(0)  # [96, 96]
                        bigimg_crop_pool_tensor = output_big_img[i, (j // 4) * 96:(j // 4 + 1) * 96,
                                                  (j % 4) * 96:(j % 4 + 1) * 96]
                        bigimg_crop_pool_tensor_detach = bigimg_crop_pool_tensor.detach().clone()
                        mini_patch_cnt = torch.sum(mini_patch_pool_tensor / SCALE_FACTOR).item()
                        bigimg_crop_cnt = torch.sum(bigimg_crop_pool_tensor_detach / SCALE_FACTOR).item()
                        # the classify result of the small pieces is the head
                        if sim_map_max_index[j] == 0:
                            if 2.5 < mini_patch_cnt / bigimg_crop_cnt < 10:
                                pool_tensor = mini_patch_pool_tensor
                                pool_tensor_list.append(pool_tensor.unsqueeze(0))
                                bigimg_crop_pool_tensor_list.append(bigimg_crop_pool_tensor.unsqueeze(0))
                        else:
                            if 1 < mini_patch_cnt / bigimg_crop_cnt and bigimg_crop_cnt <= 8:
                            # if 2.5 < mini_patch_cnt / bigimg_crop_cnt < 15 and bigimg_crop_cnt <= 3.5:
                                pool_tensor = torch.zeros(96,96).to(self.device)
                                pool_tensor_list.append(pool_tensor.unsqueeze(0))
                                bigimg_crop_pool_tensor_list.append(bigimg_crop_pool_tensor.unsqueeze(0) * self.args.consistency_factor)
                    if len(pool_tensor_list) == 0:
                        continue
                    pool_tensor_cat = torch.cat(pool_tensor_list, 0) # [S, 96, 96]
                    bigimg_crop_pool_tensor_cat = torch.cat(bigimg_crop_pool_tensor_list, 0)  # [S, 96, 96]
                    pool_tensor_cat_list.append(pool_tensor_cat)
                    bigimg_crop_pool_tensor_cat_list.append(bigimg_crop_pool_tensor_cat)

                self.model.train()
                self.restore_params()

                if len(pool_tensor_cat_list) == 0:
                    continue
                pool_tensor_cat_cat = torch.cat(pool_tensor_cat_list, 0)
                bigimg_crop_pool_tensor_cat_cat = torch.cat(bigimg_crop_pool_tensor_cat_list, 0)
                pseudo_list.append(pool_tensor_cat_cat)
                inference_list.append(bigimg_crop_pool_tensor_cat_cat)
            if len(pseudo_list) != 0:
                pseudo_tensor = torch.cat(pseudo_list, 0)
                inference_tensor = torch.cat(inference_list, 0)
                loss = self.loss(inference_tensor, pseudo_tensor) # [1]
            else:
                # loss = torch.randn(1).to(self.device)
                loss = torch.tensor([0.]).to(self.device)
                loss.requires_grad_(True)
            # with torch.no_grad():
            #     output, extra_out = self.model(image_crop, prompt, return_extra=True, coop_require_grad=True)

        else:
            samples, gt_density, boxes, m_flag, prompt_gt, prompt_add = batch

            if not self.args.use_contrast:
                prompt_gt = [f"a photo of {p}" for p in prompt_gt]

            output, extra_out = self.model(samples, prompt_gt, return_extra=True, coop_require_grad =  True)
            loss = self.loss(output, gt_density) # [1]

        # Compute loss function
        mask = np.random.binomial(n=1, p=0.8, size=[384,384])
        masks = np.tile(mask,(samples.shape[0],1))
        masks = masks.reshape(samples.shape[0], 384, 384)
        masks = torch.from_numpy(masks).to(self.device)

        loss = (loss * masks / (384*384)).sum() / samples.shape[0]
        if not self.args.use_self_supervised and self.args.use_contrast and self.current_epoch <= self.args.contrast_pre_epoch:
            text_embedding = extra_out['text_embedding'] # [B,1, 512]
            if self.args.contrast_pos == "pre":
                patch_embedding = extra_out['patch_embedding_contrast'] # [B, 196, 512]
            elif self.args.contrast_pos == "post":
                patch_embedding = extra_out['pixel_text_matching_map']
            img_embedding = extra_out['x_cls'] # [B, 1, 512]
            contrast_loss = self.contrastive_loss(patch_embedding, img_embedding, text_embedding, self.neg_prompt_embed,  gt_density.detach().clone())
            loss = args.w_contrast * contrast_loss
            self.log('train_loss_contrast', contrast_loss)


        self.log('train_loss', loss)

        if not self.args.use_self_supervised:
            # Update information of MAE and RMSE
            batch_mae = 0

            batch_rmse = 0
            gt_sum = 0
            for h_ in range(output.shape[0]):
                pred_cnt = torch.sum(output[h_]/SCALE_FACTOR).item()
                gt_cnt = torch.sum(gt_density[h_]/SCALE_FACTOR).item()
                cnt_err = abs(pred_cnt - gt_cnt)
                gt_sum += gt_cnt
                batch_mae += cnt_err
                batch_rmse += cnt_err ** 2
            batch_mae /= output.shape[0]
            batch_rmse /= output.shape[0]
            batch_rmse = math.sqrt(batch_rmse)
            # loss = loss / gt_sum
            self.log('train_mae', batch_mae)
            self.log('train_rmse', batch_rmse)
    
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.args.use_self_supervised:
            assert self.args.dataset_type == "ShanghaiTech", "should use ShanghaiTech when self-supervising"
            image, _, origin_img_tensor, gt_density, im_name, prompt = batch
            samples = image[:,:,:,:384]
            gt_density = gt_density[:,:,:384]
        else:
            samples, gt_density, _, _, prompt, _ = batch

        if not self.args.use_contrast:
            prompt = [f"a photo of {p}" for p in prompt]

        output = self.model(samples, prompt)

        
        # Update information of MAE and RMSE
        batch_mae = []
        batch_rmse = []
        pred_cnts = []
        gt_cnts = []
        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i]/SCALE_FACTOR).item() # SCALE_FACTOR is the scaling factor as CounTR uses
            gt_cnt = torch.sum(gt_density[i]/SCALE_FACTOR).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            batch_mae.append(cnt_err)
            batch_rmse.append(cnt_err ** 2)
            pred_cnts.append(pred_cnt)
            gt_cnts.append(gt_cnt)


        #log the image
        img_log = samples[0].detach().cpu().numpy()
        pred_density = output[0].detach().cpu().numpy()
        pred_log_rgb = cv2.applyColorMap(np.uint8(255*pred_density), cv2.COLORMAP_JET)
        pred_log_rgb = np.transpose(pred_log_rgb, (2,0,1))
        gt_density_log = gt_density[0].detach().cpu().numpy()
        gt_log_rgb = cv2.applyColorMap(np.uint8(255*gt_density_log), cv2.COLORMAP_JET)
        gt_log_rgb = np.transpose(gt_log_rgb, (2,0,1))


        pred_density = einops.repeat(pred_density, 'h w -> c h w', c=3)
        pred_density = pred_density / pred_density.max() #normalize
        heatmap_pred = 0.33 * img_log + 0.67 * pred_density
        gt_density_log = einops.repeat(gt_density_log, 'h w -> c h w', c=3)
        heatmap_gt = 0.33 * img_log + 0.67 * gt_density_log

        return {"mae": batch_mae, "rmse": batch_rmse, "img": img_log, "pred": pred_log_rgb, "gt": gt_log_rgb, "heatmap_pred": heatmap_pred, "heatmap_gt": heatmap_gt, "prompt": prompt[0], "pred_cnts": pred_cnts, "gt_cnts": gt_cnts}
    
    def validation_epoch_end(self, outputs):
        all_mae = []
        all_rmse = []

        for output in outputs:
            all_mae += output["mae"]
            all_rmse += output["rmse"]
        val_mae = np.mean(all_mae)
        val_rmse = np.sqrt(np.mean(all_rmse))
        self.log('val_mae', val_mae)
        self.log('val_rmse', val_rmse)

        # log the image
        idx = random.randint(0, len(outputs)-1)
        img = outputs[idx]["img"]
        pred = outputs[idx]["pred"]
        gt = outputs[idx]["gt"]
        heatmap_pred = outputs[idx]["heatmap_pred"]
        heatmap_gt = outputs[idx]["heatmap_gt"]
        prompt = outputs[idx]["prompt"]
        pred_cnts = outputs[idx]["pred_cnts"]
        gt_cnts = outputs[idx]["gt_cnts"]
        pred_gt = "pred: {:.2f} gt: {:.2f}".format(pred_cnts[0], gt_cnts[0])
        self.logger.experiment.add_image("val_img", img, self.current_epoch)
        self.logger.experiment.add_image("density_pred", pred, self.current_epoch)
        self.logger.experiment.add_image("density_gt", gt, self.current_epoch)
        self.logger.experiment.add_image("overlay_pred", heatmap_pred, self.current_epoch)
        self.logger.experiment.add_image("overlay_gt", heatmap_gt, self.current_epoch)
        self.logger.experiment.add_text("prompt", prompt, self.current_epoch)
        self.logger.experiment.add_text("count", pred_gt, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        if self.args.dataset_type=='FSC' or self.args.dataset_type == "COCO":
            image, gt_density, boxes, m_flag, prompt = batch
        elif self.args.dataset_type == "CARPK":
            image, gt_cnt = batch
            gt_cnt = gt_cnt.item()
            prompt = ["car" for _ in range(image.shape[0])]
            gt_density = torch.zeros(image.shape[0], image.shape[2], image.shape[3]) 
        elif self.args.dataset_type == "ShanghaiTech":
            image, gt_cnt, origin_img_tensor, im_name = batch
            gt_cnt = gt_cnt.item()
            prompt = ["people" for _ in range(image.shape[0])] # [1]
            gt_density = torch.zeros(image.shape[0], image.shape[2], image.shape[3])


        assert image.shape[0] == 1 , "only support inference one image at a time"
        raw_h, raw_w = image.shape[2:]

        if self.args.test_method == 2:
            # composite mini-batch
            # origin_img_tensor = origin_img_tensor.unsqueeze(0)
            slides, _, mini_patches = misc.sliding_window_origin_image(origin_img_tensor)

            slides = torch.from_numpy(slides).float().to(self.device)  # [N, 3, 384, 384]/[N, 3, 1536, 1536]
            prompt_big_img = np.repeat(prompt, slides.shape[0], axis=0)  # [N]
            output_big_img = self.model(slides, prompt_big_img) # [N, 384, 384]

            mini_patches = torch.from_numpy(mini_patches).float().to(self.device)
            avg_pooling = torch.nn.AvgPool2d(2)
            output = []
            prompt = np.repeat(prompt, mini_patches.shape[1], axis=0) # [16]
            # n slice. n = mini_patches.shape[0]
            for i in range(mini_patches.shape[0]):  # [N, 16, 3, 384, 384]
                mini_patch_output, extra_out = self.model(mini_patches[i], prompt, return_extra = True)  # [16, 384, 384]
                img_embedding = extra_out['x_cls']  # [16, 1, 512]
                classify_prompt = ['heads','background','tree','leaves','sky','building']
                classify_prompt = [f"a photo of {p}" for p in classify_prompt]
                text_token = clip.tokenize(classify_prompt).to(image.device)
                with torch.no_grad():
                    text_embedding = self.model.text_encoder_classfiy(text_token).float() # [6, 1, 512]
                text_embedding.squeeze_(1) # [6, 512]
                # for index in range(4):
                #     print(f'{index}: ',text_embedding[index].detach().equal(text_embedding[index+1].detach()))
                text_embedding.unsqueeze_(0)  # [1, 6, 512]
                # [16, 1, 512] and [1, 6, 512] => [16, 6]
                sim_map = F.cosine_similarity(img_embedding, text_embedding, dim=-1)  # [16, 6]
                sim_map_max_index = sim_map.argmax(dim=1) # [16]

                pool_tensor_list = []
                for j in range(mini_patch_output.shape[0]):
                    # torch.sum(output[0] / SCALE_FACTOR).item()
                    mini_patch_pool_tensor = mini_patch_output[j].detach().unsqueeze(0)
                    mini_patch_pool_tensor = avg_pooling(mini_patch_pool_tensor) * 4
                    mini_patch_pool_tensor = avg_pooling(mini_patch_pool_tensor) * 4
                    mini_patch_pool_tensor.squeeze_(0)  # [96, 96]
                    bigimg_crop_pool_tensor = output_big_img[i, (j // 4) * 96:(j // 4 + 1) * 96,
                                          (j % 4) * 96:(j % 4 + 1) * 96].detach().clone()
                    mini_patch_cnt = torch.sum(mini_patch_pool_tensor / SCALE_FACTOR).item()
                    bigimg_crop_cnt = torch.sum(bigimg_crop_pool_tensor / SCALE_FACTOR).item()
                    # the classify result of the small pieces is the head
                    if sim_map_max_index[j] == 0 and 2.5 < mini_patch_cnt/bigimg_crop_cnt:
                        pool_tensor = mini_patch_pool_tensor
                    else:
                        pool_tensor = bigimg_crop_pool_tensor
                    pool_tensor_list.append(pool_tensor)
                # concat all patches
                results = []
                for i_ in range(4):
                    result = torch.cat(pool_tensor_list[i_*4:i_*4+4], 1)
                    results.append(result)
                results = torch.cat(results, 0) # [384, 384]
                output.append(results.unsqueeze(0)) # [1, 384, 384]
            # output = torch.Tensor(output)
            output = torch.cat(output, 0) # [N, 384, 384]
        elif self.args.test_method == 1:
            # composite mini-batch
            # origin_img_tensor = origin_img_tensor.unsqueeze(0)
            slides, _, mini_patches = misc.sliding_window_origin_image(origin_img_tensor)
            mini_patches = torch.from_numpy(mini_patches).float().to(self.device)
            avg_pooling = torch.nn.AvgPool2d(2)
            output = []
            prompt = np.repeat(prompt, mini_patches.shape[1], axis=0)
            # n slice. n = mini_patches.shape[0]
            for i in range(mini_patches.shape[0]):  # [N, 16, 3, 384, 384]
                mini_patch_output = self.model(mini_patches[i], prompt)  # [16, 384, 384]
                pool_tensor_list = []
                for j in range(mini_patch_output.shape[0]):
                    pool_tensor = mini_patch_output[j].detach().unsqueeze(0)
                    pool_tensor = avg_pooling(pool_tensor) * 4
                    pool_tensor = avg_pooling(pool_tensor) * 4
                    pool_tensor.squeeze_(0) # [96, 96]
                    pool_tensor_list.append(pool_tensor)
                results = []
                for i_ in range(4):
                    result = torch.cat(pool_tensor_list[i_*4:i_*4+4], 1)
                    results.append(result)
                results = torch.cat(results, 0) # [384, 384]
                output.append(results.unsqueeze(0)) # [1, 384, 384]
            # output = torch.Tensor(output)
            output = torch.cat(output, 0) # [N, 384, 384]
        elif self.args.test_method == 0:
            patches, _ = misc.sliding_window(image, stride=128)
            # covert to batch
            patches = torch.from_numpy(patches).float().to(self.device) # [N, 3, 384, 384]
            prompt = np.repeat(prompt, patches.shape[0], axis=0)
            output = self.model(patches, prompt)

        # Looks like adding a channel dimension
        output.unsqueeze_(1) # [N, 1, 384, 384]
        output = misc.window_composite(output, stride=128) # [1, 1, W+, 384]
        output = output.squeeze(1) # [1, W+, 384]
        # crop to original width
        output = output[:, :, :raw_w] # [1, W, 384]

        # Update information of MAE and RMSE
        batch_mae = []
        batch_rmse = []
        pred_cnts = []
        gt_cnts = []

        pred_cnt = torch.sum(output[0]/SCALE_FACTOR).item()
        if self.args.dataset_type == "FSC" or self.args.dataset_type == "COCO":
            gt_cnt = torch.sum(gt_density[0]/SCALE_FACTOR).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        batch_mae.append(cnt_err)
        batch_rmse.append(cnt_err ** 2)
        pred_cnts.append(pred_cnt)
        gt_cnts.append(gt_cnt)
 

        #log the image
        img_log = image[0].detach().cpu().numpy()
        pred_density = output[0].detach().cpu().numpy()
        pred_log_rgb = cv2.applyColorMap(np.uint8(255*pred_density), cv2.COLORMAP_JET)
        pred_log_rgb = np.transpose(pred_log_rgb, (2,0,1))
        gt_density_log = gt_density[0].detach().cpu().numpy()
        gt_log_rgb = cv2.applyColorMap(np.uint8(255*gt_density_log), cv2.COLORMAP_JET)
        gt_log_rgb = np.transpose(gt_log_rgb, (2,0,1))


        pred_density = einops.repeat(pred_density, 'h w -> c h w', c=3)
        pred_density_max = pred_density.max()
        tmp_sum = pred_density[0].sum()
        # pred_density = pred_density / pred_density_max #normalize
        heatmap_pred = img_log 
        heatmap_pred = 0.33 * img_log + 0.67 * pred_density
        gt_density_log = einops.repeat(gt_density_log, 'h w -> c h w', c=3)
        heatmap_gt = img_log 

        # log qualitative results
        if self.args.log_test_img:
            # if cnt_err < 5:
            if 1 == 1:
                #log density
                log_dir = "out/good_density/"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                name = "{}_{}_{:.2f}_gt_{:.2f}.jpg".format(im_name[0], prompt[0], pred_cnt, gt_cnt)
                # name = "good_{}_{:.2f}_gt_{:.2f}.jpg".format(prompt[0], pred_cnt, gt_cnt)
                pred_density_write = 1. - pred_density[0]
                pred_density_write = cv2.applyColorMap(np.uint8(255*pred_density_write), cv2.COLORMAP_JET)
                img = Image.fromarray(np.uint8(pred_density_write))
                img.save(log_dir + name)

                log_dir = "out/good_pred/"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                #log overlay
                name = "{}_{}_{:.2f}_gt_{:.2f}.jpg".format(im_name[0],prompt[0], pred_cnt, gt_cnt)
                # name = "good_{}_{:.2f}_gt_{:.2f}.jpg".format(prompt[0], pred_cnt, gt_cnt)
                pred_density_write = pred_density_write / 255.
                img_write = 0.33 * np.transpose(img_log,(1,2,0)) + 0.67 * pred_density_write
                img = Image.fromarray(np.uint8(255*img_write))
                img.save(log_dir + name)

            # if cnt_err > 100:
            if cnt_err > 100000:
                #save image, overlaied
                #log density
                name = "good_{}_{:.2f}_gt_{:.2f}.jpg".format(prompt[0], pred_cnt, gt_cnt)
                pred_density_write = 1. - pred_density[0]
                pred_density_write = cv2.applyColorMap(np.uint8(255*pred_density_write), cv2.COLORMAP_JET)

                log_dir = "debug/bad_pred/"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                name = "bad_{}_{:.2f}_gt_{:.2f}.jpg".format(prompt[0], pred_cnt, gt_cnt)
                pred_density_write = pred_density_write / 255.
                img_write = 0.33 * np.transpose(img_log,(1,2,0)) + 0.67 * pred_density_write
                img = Image.fromarray(np.uint8(255*img_write))
                img.save(log_dir + name)

        return {"mae": batch_mae, "rmse": batch_rmse, "img": img_log, "pred": pred_log_rgb, "gt": gt_log_rgb, "heatmap_pred": heatmap_pred, "heatmap_gt": heatmap_gt, "prompt": prompt[0], "pred_cnts": pred_cnts, "gt_cnts": gt_cnts}
    
    def test_epoch_end(self, outputs):
        all_mae = []
        all_rmse = []


        for output in outputs:
            all_mae += output["mae"]
            all_rmse += output["rmse"]
        test_mae = np.mean(all_mae)
        test_rmse = np.sqrt(np.mean(all_rmse))
        self.log('test_mae', test_mae)
        self.log('test_rmse', test_rmse)

    def forward(self, img, prompt):
        """
        img: (1, 3, H, W)
        prompt: List[str]
        """
        return self.model(img, prompt)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )

        schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.33)
        return {"optimizer": optimizer, "lr_scheduler": schedular, "monitor": "val_mae"}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # delete frozen clip parameters
        if not self.args.unfreeze_vit :
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith("model.clip") or k.startswith("model.img_encoder.clip") or k.startswith("model.text_encoder.clip") or k.startswith("model.img_encoder.vit"):
                    del checkpoint["state_dict"][k]

    def overwrite_args(self, args):
        """Avoid the exception caused by lighting when loading incompatible args from model ckpt."""
        self.args = args
        self.shadow = {}  # record teacher average weight parameter
        self.backup = {}  # record student backup parameter

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    seed = args.seed
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if not args.use_self_supervised:
        dataset_train = FSC147(split = "train")
        all_classes_train = dataset_train.all_classes
    else:
        dataset_train = ShanghaiTech(None, split="train",
                                                      part = "A", preserve_the_original_image=True)
        all_classes_train = None

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    # the val set for training.
    if not args.use_self_supervised:
        dataset_val = FSC147( split = "val")
    else:
        dataset_val = ShanghaiTech(None, split="test",
                                                      part = "A", preserve_all=True)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    val_dataloader =  torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )



    save_callback = pl.callbacks.ModelCheckpoint(monitor='val_mae', save_top_k=4, mode='min',  filename='{epoch}-{val_mae:.2f}')
    model = Model(args,all_classes=all_classes_train)
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=args.exp_name)
    max_epochs = args.self_supervised_epoch if args.use_self_supervised else args.epochs+args.contrast_pre_epoch
    trainer = Trainer(
        accelerator="gpu", 
        callbacks=[save_callback],
        accumulate_grad_batches = args.accum_iter,
        precision=16, 
        max_epochs=max_epochs,
        logger=logger,
        check_val_every_n_epoch=args.val_freq,
    )
    if args.mode == "train":
        if args.ckpt is not None:
            model = Model.load_from_checkpoint(args.ckpt, strict=False)
            model.overwrite_args(args)
        model.register_params()
        # model = Model.load_from_checkpoint('epoch=209-val_mae=16.60.ckpt', strict=False)
        # checkpoint = torch.load('epoch=209-val_mae=16.60.ckpt')
        # epp = checkpoint['epoch']
        if args.resume_checkpoint:
            # automatically restores model, epoch, step, LR schedulers, apex, etc...
            trainer.fit(model, train_dataloader, val_dataloader)
        else:
            trainer.fit(model, train_dataloader, val_dataloader)
    elif args.mode == "test":
        if args.dataset_type == "FSC":
            dataset_val = FSC147(split = "val", resize_val=False)
            dataset_test = FSC147(split = "test")
        elif args.dataset_type == "COCO":
            dataset_val = FSC147(split = "val_coco", resize_val=False)
            dataset_test = FSC147(split = "test_coco")

        elif args.dataset_type == "CARPK":
            dataset_val = dataset_test = CARPK(None, split="test")
        elif args.dataset_type == "ShanghaiTech":
            dataset_val = dataset_test = ShanghaiTech(None, split="test",
                                                      part = "A", preserve_the_original_image=True)


        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        # when inference, batch size is always 1
        val_dataloader =  torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        if args.ckpt is None:
            raise ValueError("Please specify a checkpoint to test")
        model = Model.load_from_checkpoint(args.ckpt,strict=False)
        model.overwrite_args(args)
        model.eval()
        if args.dataset_type == "FSC" or args.dataset_type == "COCO": #CARPK and ShanghaiTech do not have val set
            print("====Metric on val set====")
            trainer.test(model, val_dataloader)
        print("====Metric on test set====")
        trainer.test(model, test_dataloader)



    elif args.mode == "app":
        if args.ckpt is None:
            raise ValueError("Please specify a checkpoint to test")
        model = Model.load_from_checkpoint(args.ckpt,strict=False)
        model.eval()
        def infer(img, prompt):
            model.eval()
            model.model = model.model.cuda()
            with torch.no_grad():
                # reshape height to 384, keep aspect ratio
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
                img = TF.resize(img, (384))
                
                img = img.float()/255.
                img = torch.clamp(img, 0, 1)
                prompt = [prompt]
                with torch.cuda.amp.autocast():
                    raw_h, raw_w = img.shape[2:]
                    patches, _ = misc.sliding_window(img,stride=128)
                    #covert to batch
                    patches = torch.from_numpy(patches).float().to(img.device)
                    prompt = np.repeat(prompt, patches.shape[0], axis=0)
                    output = model.forward(patches, prompt)
                    output.unsqueeze_(1)
                    output = misc.window_composite(output, stride=128)
                    output = output.squeeze(1)
                    #crop to original width
                    output = output[:, :, :raw_w]
                pred_cnt = torch.sum(output[0]/SCALE_FACTOR).item()
                pred_density = output[0].detach().cpu().numpy()
                # normalize
                pred_density = pred_density/pred_density.max()
                pred_density_write = 1. - pred_density
                pred_density_write = cv2.applyColorMap(np.uint8(255*pred_density_write), cv2.COLORMAP_JET)
                pred_density_write = pred_density_write/255.
                # pred_rgb = cv2.applyColorMap(np.uint8(255*pred_density), cv2.COLORMAP_JET)
                img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

                
                heatmap_pred = 0.33 * img + 0.67 * pred_density_write
                heatmap_pred = heatmap_pred/heatmap_pred.max()
            return heatmap_pred, pred_cnt
        demo = gr.Interface(
            fn=infer,
            inputs=[
                # height = 384, keep aspect ratio
                gr.inputs.Image(label="Image"),
                gr.inputs.Textbox(lines=1, label="Prompt (What would you like to count)"),
            ],
            outputs= ["image", "number"],
            interpretation="default",
            title="CLIP-Count",
            description="A unified counting model to count them all.",
            
        )
        demo.launch(share=True)