import torch
import torch.nn as nn
import torch.nn.functional as F
from RecStudio.recstudio.model.scorer import CosineScorer, EuclideanScorer, InnerProductScorer

class BPRLoss(nn.Module):
    def __init__(self, score_fn='cos', neg_count=1, dns=False, reduction='mean'):
        super().__init__()
        if score_fn == 'cos':
            self.score_fn = CosineScorer()
            self.mode = 'max'
        elif score_fn == 'inner':
            self.score_fn = InnerProductScorer()
            self.mode = 'max'
        elif score_fn == 'l2':
            self.score_fn = EuclideanScorer()
            self.mode = 'min'
        self.neg_count = neg_count
        self.dns = dns
        self.reduction = reduction

    def forward(self, text_emb, ctr_emb):
        pos_score = self.score_fn(text_emb, ctr_emb)
        neg_ctr_emb = self.sample(ctr_emb)
        neg_score = self.score_fn(text_emb, neg_ctr_emb)
        if self.mode == min:
            pos_score = -pos_score
            neg_score = -neg_score
        if not self.dns:
            loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
            weight = F.softmax(torch.ones_like(neg_score), -1)
            final_loss = -(loss * weight).sum(-1)
        else:
            final_loss = -F.logsigmoid(pos_score - torch.max(neg_score, dim=-1))

        if self.reduction == 'mean':
            return torch.mean(final_loss)
        else:
            return final_loss
        
    def sample(self, emb):
        bs = emb.shape[0]
        neg_idx = torch.randint(bs - 1, size=(bs, self.neg_count))
        pos_idx = torch.arange(bs).unsqueeze(-1)
        neg_idx[neg_idx >= pos_idx] += 1
        neg_emb = emb[neg_idx]
        return neg_emb
    

class InfoNCELoss(nn.Module):
    def __init__(self, score_fn='cos', temperature=1.0, simcse=False, reduction='mean'):
        super().__init__()
        if score_fn == 'cos':
            self.score_fn = CosineScorer()
            self.mode = 'max'
        elif score_fn == 'inner':
            self.score_fn = InnerProductScorer()
            self.mode = 'max'
        else:
            raise ValueError('Wrong scorn_fn.')
        self.temperature = temperature
        # self.text_simcse = simcse
        self.reduction = reduction
    
    def forward(self, text_emb, ctr_emb):
    # def forward(self, text_emb, ctr_emb, text_emb_2=None):
        bs = text_emb.shape[0]
        sim_ii = self.score_fn(text_emb, ctr_emb) / self.temperature
        if self.mode == 'min':
            sim_ii *= -1

        # Query: text, Key: ctr
        all_ctr_emb = ctr_emb.tile((bs, 1, 1))      # B x B x D
        sim_ij_1 = self.score_fn(text_emb, all_ctr_emb) / self.temperature
        if self.mode == 'min':
            sim_ij_1 *= -1
        loss1 = torch.logsumexp(sim_ij_1, dim=-1) - sim_ii

        # Query: ctr, Key: text
        all_text_emb = text_emb.tile((bs, 1, 1))
        sim_ij_2 = self.score_fn(ctr_emb, all_text_emb) / self.temperature
        if self.mode == 'min':
            sim_ij_2 *= -1
        loss2 = torch.logsumexp(sim_ij_2, dim=-1) - sim_ii

        loss = loss1 + loss2

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss

        # if self.text_simcse:
        #     # Query: text, Key: text
        #     all_text_emb = torch.cat([text_emb, text_emb_2], -1).reshape(2 * bs, -1)    # 2B x D
        #     idxs = torch.arange(0, 2 * bs, device=text_emb.device)
        #     y_true = idxs + 1 - idxs % 2 * 2
        #     sim_ij = self.score_fn(all_text_emb, all_text_emb.tile(2 * bs, 1, 1))   # 2B x D; 2B x 2B x D -> 2B x 2B
        #     sim_ij -= torch.eye(2 * bs, device=text_emb.device) * 1e12
        #     sim_ij /= self.temperature
        #     loss_simcse = torch.mean(F.cross_entropy(sim_ij, y_true))

        #     loss += loss_simcse
        return loss

        