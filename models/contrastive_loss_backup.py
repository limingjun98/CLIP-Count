import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, noise_text_ratio=0.0, normalize=False):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.noise_text_ratio = noise_text_ratio
        self.normalize = normalize




    def forward(self, patch_embedding, img_embedding, gt_text_embedding_map, noise_text_embeddings, gt_density):
        """
        Args:
            patch_embedding: (B, 196, 512) embedding of image patch feature
            img_embedding: (B, 1, 512) embedding of image feature
            text_embedding: (B, 1, 512), ground truth text embedding
            noise_text_embeddings: (N, 1, 512), noise text embeddings
            gt_density: (B, 384, 384), ground truth density map
        """
        gt_density = F.interpolate(gt_density.unsqueeze_(1), size=(224, 224), mode='nearest')
        density_mask = F.max_pool2d(gt_density, kernel_size=16, stride=16, padding=0) #same as ViT conv1 
        density_mask = density_mask > 0.
        density_mask = density_mask.permute(0, 2, 3 ,1) # (B, 14, 14, 1)

        gt_text_embedding_map = gt_text_embedding_map.unsqueeze(1).expand(-1, 14, 14, -1) 

        # [B, 14, 14, 512], contains both gt and noise text embedding
        fused_text_embedding_map =  gt_text_embedding_map
        pos_mask = density_mask.squeeze_(-1) # (B, 14, 14, 1)
        
        patch_embeddings = patch_embedding.reshape(-1, 14, 14, 512) # (B, 14, 14, 512)
        # img_normlen = patch_embeddings.norm(dim=-1, keepdim=True)
        # text_normlen = fused_text_embedding_map.norm(dim=-1, keepdim=True)

        #batch cosine similarity, this function automatically normalizes the vectors
        sim_map = F.cosine_similarity(patch_embeddings, fused_text_embedding_map , dim=-1) # (B, 14, 14)
        # sim_global = F.cosine_similarity(img_embedding, fused_text_embedding_map , dim=-1) # (B, 1)
        n_pos = torch.sum(pos_mask, dim=(1, 2)) # (B) how many positive samples in each batch
        # if n_pos == 0, set to 1 to avoid nan
        n_pos = torch.where(n_pos == 0, torch.ones_like(n_pos), n_pos)
        #infoNCE 

        sim_map = torch.exp(sim_map / self.temperature)
        pos_sum = torch.sum(torch.where(pos_mask, sim_map, torch.zeros_like(sim_map)), dim=(1, 2)) + 1e-5
        neg_sum = torch.sum(torch.where(~pos_mask, sim_map, torch.zeros_like(sim_map)), dim=(1, 2)) + 1e-5

        loss = -torch.log(pos_sum / (pos_sum + neg_sum))
        if self.normalize:
            loss = loss / n_pos            
        return loss.mean()


class ContrastiveLossBoost(nn.Module):
    def __init__(self, temperature=0.07, noise_text_ratio=0.0, normalize=False, text_encoder = None):
        super(ContrastiveLossBoost, self).__init__()
        self.temperature = temperature
        self.noise_text_ratio = noise_text_ratio
        self.normalize = normalize
        self.criterion_loss_fn = nn.CrossEntropyLoss()
        self.avg_pooling = nn.AvgPool2d(16)
        self.text_encoder = text_encoder

    def forward(self, patch_embedding, img_embedding, gt_text_embedding_map, noise_text_embeddings, gt_density):
        """
        Args:
            patch_embedding: (B, 196, 512) embedding of image patch feature
            img_embedding: (B, 1, 512) embedding of image feature
            text_embedding: (B, 1, 512), ground truth text embedding
            noise_text_embeddings: (N, 1, 512), noise text embeddings
            gt_density: (B, 384, 384), ground truth density map
        """

        gt_density = F.interpolate(gt_density.unsqueeze_(1), size=(224, 224), mode='nearest')
        my_sum = torch.sum(gt_density)
        gt_t = self.avg_pooling(gt_density) * 256  # [B, 14, 14]
        gt_t = gt_t.reshape(gt_t.shape[0], -1)  # [B, 196]
        digital_text_emb_list = []
        digital_text_list = []
        for i in range(gt_t.shape[0]):
            for j in range(gt_t.shape[1]):
                text = str(round(gt_t[i,j].item()))
                digital_text_list.append(text)
                # text_token = clip.tokenize(text).cuda()
                # text_embedding = self.text_encoder(text_token).float()
                # digital_text_emb_list.append(text_embedding)
        text_token = clip.tokenize(digital_text_list).cuda()
        # for i in range(gt_t.shape[1]):
        #     text_embedding = self.text_encoder(text_token[i*196:(i+1)*196, :]).float()
        #     digital_text_list.append(text_embedding)


        text_embedding = self.text_encoder(text_token[:100,:])
        # digital_text_emb = torch.stack(digital_text_emb_list, dim=0)
        my_sum2 = torch.sum(gt_t)
        density_mask = F.max_pool2d(gt_density, kernel_size=16, stride=16, padding=0)  # same as ViT conv1
        density_mask = density_mask > 0.
        density_mask = density_mask.permute(0, 2, 3, 1)  # (B, 14, 14, 1)

        gt_text_embedding_map = gt_text_embedding_map.unsqueeze(1).expand(-1, 14, 14, -1)

        # [B, 14, 14, 512], contains both gt and noise text embedding
        fused_text_embedding_map = gt_text_embedding_map
        pos_mask = density_mask.squeeze_(-1)  # (B, 14, 14)

        patch_embeddings = patch_embedding.reshape(-1, 14, 14, 512)  # (B, 14, 14, 512)

        digital_text_embedding_map = fused_text_embedding_map.reshape(-1, 196, 512)  # (B, 196, 512)
        patch_emb = patch_embedding.unsqueeze(2) # (B, 196, 1, 512)
        digital_text_embed = digital_text_embedding_map.unsqueeze(1)  # (B, 1, 196, 512)
        res = F.cosine_similarity(patch_emb, digital_text_embed, dim=-1)  # (B, 196, 196) （B, N, C）
        res = res.permute(0, 2, 1)  # (B, C, N)
        gt_tensor = torch.arange(196).cuda()
        gt_tensor = gt_tensor.expand(res.shape[0], -1)
        res2 = res / self.temperature
        the_loss = self.criterion_loss_fn(res, gt_tensor)
        the_loss2 = self.criterion_loss_fn(res2, gt_tensor)
        pass
        out1 = F.softmax(res, dim=-1)
        out2 = F.softmax(res2, dim=-1)
        img_normlen = patch_embeddings.norm(dim=-1, keepdim=True)
        # text_normlen = fused_text_embedding_map.norm(dim=-1, keepdim=True)

        '''
        # batch cosine similarity, this function automatically normalizes the vectors
        sim_map = F.cosine_similarity(patch_embeddings, fused_text_embedding_map, dim=-1)  # (B, 14, 14)
        # sim_global = F.cosine_similarity(img_embedding, fused_text_embedding_map , dim=-1) # (B, 1)
        n_pos = torch.sum(pos_mask, dim=(1, 2))  # (B) how many positive samples in each batch
        # if n_pos == 0, set to 1 to avoid nan
        n_pos = torch.where(n_pos == 0, torch.ones_like(n_pos), n_pos)
        # infoNCE

        sim_map = torch.exp(sim_map / self.temperature)
        pos_sum = torch.sum(torch.where(pos_mask, sim_map, torch.zeros_like(sim_map)), dim=(1, 2)) + 1e-5
        neg_sum = torch.sum(torch.where(~pos_mask, sim_map, torch.zeros_like(sim_map)), dim=(1, 2)) + 1e-5

        loss = -torch.log(pos_sum / (pos_sum + neg_sum))
        if self.normalize:
            loss = loss / n_pos
        return loss.mean()
        '''

