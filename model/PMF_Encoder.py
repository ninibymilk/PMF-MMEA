from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .Tool_model import GAT, GCN


class PMFusion(nn.Module):
    '''
    PMFusion class for multi-modal feature fusion
    '''
    
    def __init__(self, args, ent_num, modal_num, with_weight = 1):
        
        """
        Args:
            args: Model arguments containing configurations.
            ent_num: Number of entities.
            modal_num: Number of modalities.
            with_weight: Whether to apply weights in normalization (default is 1).
        """
        super().__init__()
        self.args = args
        self.ENT_NUM = ent_num
        self.modal_num = modal_num

    def forward(self, embs_dict,input_idx,adj,mask):
     
        embs = [emb for key,emb in embs_dict.items() if emb is not None]
        modal_num = len(embs)

        hidden_states = torch.stack(embs, dim=1)
        weight_norm = None
        
        if self.args.with_weight :
            embs = [F.normalize(emb*mask[key].unsqueeze(1).repeat(1,emb.shape[1])) for key,emb in embs_dict.items() if emb is not None]
        else:           
            embs = [F.normalize(emb) for key,emb in embs_dict.items() if emb is not None]
        
        joint_emb = torch.cat(embs, dim=1)
       
        return joint_emb, hidden_states, weight_norm

class PMF_Encoder(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units
    """

    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 use_project_head=False,
                 attr_input_dim=1000,
                 img_features=None):
        super(PMF_Encoder, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        name_dim = self.args.name_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        '''
        Entity Embedding
        '''
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True
        
        '''
        Modal Encoder
        '''
        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(attr_input_dim, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
       
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)
        
        # structure encoder
        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)
        '''
        Fusion Encoder
        '''
        self.fusion = PMFusion(args, ent_num, modal_num=self.args.inner_view_num,
                                    with_weight=self.args.with_weight)

    def forward(self,
                input_idx,
                adj,
                mask,
                img_features=None,
                rel_features=None,
                att_features=None,
                name_features=None,
                char_features=None,
                ):

        if self.args.w_gcn:
            gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
        else:
            gph_emb = None
        
        if self.args.w_img:
            img_emb = self.img_fc(img_features)
        else:
            img_emb = None
        
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None
        
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None
        
        if self.args.w_name and name_features is not None:
            name_emb = self.name_fc(name_features)
        else:
            name_emb = None
        
        if self.args.w_char and char_features is not None:
            char_emb = self.char_fc(char_features)
        else:
            char_emb = None
        
        emb_dict =  {"structure": gph_emb,  "relation":rel_emb, "attribute": att_emb,"image":img_emb, "name": name_emb,\
                    "char": char_emb}    
        joint_emb, hidden_states, weight_norm = self.fusion(emb_dict,input_idx,adj,mask)

        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb,  joint_emb, hidden_states, weight_norm
