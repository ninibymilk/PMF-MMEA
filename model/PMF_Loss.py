import torch
from torch import nn
import torch.nn.functional as F


class CustomMultiLossLayer(nn.Module):
    """
    Custom layer for automatically weighted loss.
    
    Inspired by
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    """
    def __init__(self, loss_num):
        """
        Initializes the layer with the number of losses.
        
        Args:
            loss_num (int): Number of loss functions to be combined.
        """
        self.loss_num = loss_num
        super(CustomMultiLossLayer, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        """
        Combines multiple losses using learned weights.

        Args:
            loss_list (list): List of individual loss values.
        """
        self.loss_num = len(loss_list)
        # Calculate precision (inverse variance) for each loss
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            # Weight each loss by its precision and add a regularization term
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss


class CKG_Loss(nn.Module):
    """
    Cross-KG (Knowledge Graph) Alignment Loss.
    
    """
    def __init__(self, tau=0.05, ab_weight=0.5, n_view=2, intra_weight=1.0):
        
        """
        Args:
            tau (float): Temperature parameter for contrastive loss.
            ab_weight (float): Weight between two views.
            n_view (int): Number of views.
            intra_weight (float): Weight for intra-graph alignment.
        """
        super(CKG_Loss, self).__init__()
        self.tau = tau
        self.weight = ab_weight 
        self.n_view = n_view
        self.intra_weight = intra_weight 
        
    def softXEnt(self, target, logits, mask=None):
        """
        Soft cross-entropy loss function.

        Args:
            target (torch.Tensor): Target probabilities.
            logits (torch.Tensor): Logits from the model.
            mask (torch.Tensor, optional): Stop gradient flag.
        """
        logprobs = F.log_softmax(logits, dim=1)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1,target.shape[1])
            loss = -(target * logprobs* mask).sum() / logits.shape[0]
        else:
            loss = -(target * logprobs).sum() / logits.shape[0]
                 
        return loss


    def forward(self, emb, train_links, mask=None):
       
        emb = F.normalize(emb, dim=1)
      
        zis = emb[train_links[:, 0]]  # Embeddings for the first KG of nodes
        zjs = emb[train_links[:, 1]]  # Embeddings for the second KG of nodes

        temperature = self.tau
        alpha = self.weight
        n_view = self.n_view
        LARGE_NUM = 1e9
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]
        hidden1_large = hidden1
        hidden2_large = hidden2
        
        num_classes = batch_size * n_view
        
        # Create one-hot labels for the batch
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
        labels = labels.cuda()
        
        # Create masks to exclude self-contrast
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.cuda().float() 
        logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM # Mask out diagonal elements
        
        logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
    
       
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

       
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)
    
        
        if mask is not None:
            mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))
            loss_a = self.softXEnt(labels, logits_a, mask[train_links[:, 0]])
            loss_b = self.softXEnt(labels, logits_b, mask[train_links[:, 1]])
        else:
            loss_a = self.softXEnt(labels, logits_a)
            loss_b = self.softXEnt(labels, logits_b)
        
        return alpha * loss_a + (1 - alpha) * loss_b

'''
Cross-Modality Association Loss
'''       
class CM_Loss(nn.Module):
    """
    Cross-Modality (CM) Association Loss.
    
    """
    def __init__(self, tau=0.05, ab_weight=0.5, weight = 1.0):
        """
        Args:
            tau (float): Temperature parameter.
            ab_weight (float): Weight between two modalities.
            weight (float): Overall weight for the loss.
        """
        super(CM_Loss, self).__init__()
        self.tau = tau
        self.alpha = ab_weight
        self.weight = weight
    
    def softXEnt(self, target, logits, mask_l=None, mask_r=None):
        '''
        Args:
            target (torch.Tensor): Target probabilities.
            logits (torch.Tensor): Logits from the model.
            mask_l (torch.Tensor, optional): Left-side stop gradient flag (the link between l and r is unknown).
            mask_r (torch.Tensor, optional): Right-side stop gradient flag.
        '''  
        logprobs = F.log_softmax(logits, dim=1)
        if mask_l is not None:
            mask_l = mask_l.unsqueeze(1).repeat(1,target.shape[1])
            mask_r = mask_r.unsqueeze(1).repeat(1,target.shape[1])
            loss = (-(target * logprobs* mask_l*mask_r).sum()) / logits.shape[0]
        else:
            loss = -(target * logprobs).sum() / logits.shape[0]
        
        return loss
    
    def forward(self, m1_emb, m2_emb, train_links, loss_mask_l=None, loss_mask_r=None):
        '''
         Args:
            m1_emb (torch.Tensor): Embeddings from modality 1.
            m2_emb (torch.Tensor): Embeddings from modality 2.
            train_links (torch.Tensor): Links between the modalities(note: in test set, the link is unknown).
            loss_mask_l (torch.Tensor, optional): Left-side loss mask.
            loss_mask_r (torch.Tensor, optional): Right-side loss mask.

        '''
        temperature = self.tau
        m1_emb = F.normalize(m1_emb, dim=1)
        m2_emb = F.normalize(m2_emb, dim=1)

        m1_emb_l = m1_emb[train_links[:, 0]]
        m1_emb_r = m1_emb[train_links[:, 1]]
        m2_emb_l = m2_emb[train_links[:, 0]]
        m2_emb_r = m2_emb[train_links[:, 1]]
        batch_size = m1_emb_l.shape[0]
      
        
        label = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        label = label.cuda()

        mask = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        mask = mask.cuda().float()
        
        logits_m1_m2_ll = torch.matmul(m1_emb_l, torch.transpose(m2_emb_l, 0, 1)) / temperature
      
        logits_m1_m2_rr = torch.matmul(m1_emb_r, torch.transpose(m2_emb_r, 0, 1)) / temperature
       
        if loss_mask_l is not None:
            loss_mask_l = torch.where(loss_mask_l > 0, torch.ones_like(loss_mask_l), torch.zeros_like(loss_mask_l))
            loss_mask_r = torch.where(loss_mask_r > 0, torch.ones_like(loss_mask_r), torch.zeros_like(loss_mask_r))
            loss_l = self.softXEnt(label, logits_m1_m2_ll,loss_mask_l[train_links[:, 0]], loss_mask_r[train_links[:, 0]])
            loss_r = self.softXEnt(label, logits_m1_m2_rr,loss_mask_l[train_links[:, 1]], loss_mask_r[train_links[:, 1]])
        else:
            loss_l = self.softXEnt(label, logits_m1_m2_ll)
            loss_r = self.softXEnt(label, logits_m1_m2_rr)
        loss = loss_l + loss_r
        
        return loss
