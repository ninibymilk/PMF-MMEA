from ast import parse
import os.path as osp
import numpy as np
import random
import torch
from easydict import EasyDict as edict
import argparse



class cfg():
    
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        # change
        self.data_root = osp.abspath(osp.join(self.this_dir, '..', 'data', ''))
    

    def get_args(self):
        parser = argparse.ArgumentParser()
        # basewith_tf
        parser.add_argument('--gpu', default=0, type=int)
        parser.add_argument('--batch_size', default=3500, type=int)
        parser.add_argument('--epoch', default=100, type=int)
        parser.add_argument("--save_model", default=0, type=int, choices=[0, 1])
        parser.add_argument("--only_test", default=0, type=int, choices=[0, 1])
        
        parser.add_argument("--distance_choise", default="all", type=str, choices=["ii_1", "ii_2", "tt_1", "tt_2","all"])
        parser.add_argument("--m_csls", default=2, type=int, help="m for csls")
        parser.add_argument("--csls", action="store_true", default=True, help="use CSLS for inference")
        parser.add_argument("--csls_k", type=int, default=3, help="top k for csls")
        parser.add_argument("--nni", default=0, type=int, choices=[0, 1])
        #---------------cross-modal learing-------------------
        parser.add_argument("--cross_modal", default=0, type=int, choices=[0, 1])
        parser.add_argument("--cross_modal_weight", nargs='+', type=float, default=[10, 0.1, 0.1, 0.1, 0.1, 0.1], help="the weight of cross_modal Loss")
        parser.add_argument('--cross_modal_tau', type=float, default=0.05,help="the temperature factor of cross_modal Loss" )
        parser.add_argument("--flood_level", default=0, type=float, help="flood level")
        parser.add_argument("--cm_lr", default=1e-3, type=float, help="cross_modal lr")
         #---------------freeze-------------------
        parser.add_argument("--freeze", default=0, type=int, choices=[0, 1] )
        parser.add_argument("--freeze_theta", default=0.9, type=float, help=' threshold upperbound for modality freeze')
        parser.add_argument("--image_theta", default=0, type=float, help='absolute threshold for image freeze')
        parser.add_argument("--freeze_epochs", default=50,type=int, help="max image freeze epochs")
        parser.add_argument("--freeze_rule", default='abs_relu', type=str, choices=["abs", "abs_relu","rel","rel_relu",'rel_relu2','rel_sigmoid'], help="freeze rule")
        parser.add_argument("--freeze_start", default = 10, type=float, help="freeze start epoch")
        parser.add_argument("--growing_rate", default = 1.2, type=float, help="growing rate of freeze rate")
        parser.add_argument("--frozen",default=1, type=int, choices=[0, 1], help="whether to stop gradient")
        parser.add_argument("--with_weight", type=int, default=1, help="Whether to weight the fusion of different modality")
      
      
        parser.add_argument("--no_tensorboard", default=False, action="store_true")
        parser.add_argument("--exp_name", default="EA_exp", type=str, help="Experiment name")
        parser.add_argument("--dump_path", default="dump/", type=str, help="Experiment dump path")
        parser.add_argument("--exp_id", default="001", type=str, help="Experiment ID")
        parser.add_argument("--random_seed", default=3408, type=int)
        parser.add_argument("--data_path", default="mmkg", type=str, help="Experiment path")

    
        parser.add_argument("--data_choice", default="FBDB15K", type=str, choices=["DBP15K", "DWY", "FBYG15K", "FBDB15K", "OEA_EN_FR_15K_V1", "OEA_EN_FR_15K_V2", "OEA_D_W_15K_V2",
                                                                                  "OEA_EN_DE_15K_V1", "OEA_EN_DE_15K_V2", "OEA_D_W_15K_V1", "OEA_EN_FR_100K_V2", "OEA_EN_FR_100K_V1", "OEA_D_W_100K_V2", "OEA_D_W_100K_V1"], help="Experiment path")
        parser.add_argument("--modality_choice", default="joint", type=str, choices=["all", "joint","structure","relation","attribute", "image","name"], help="Modality Information for Alignment")
        parser.add_argument("--data_rate", type=float, default=0.3, help="training set rate")
        parser.add_argument("--model_name", default="PMF", type=str, choices=["PMF"], help="model name")
        parser.add_argument("--model_name_save", default="", type=str, help="model name for model load")
        parser.add_argument('--workers', type=int, default=12)
        parser.add_argument('--accumulation_steps', type=int, default=1)
        parser.add_argument("--scheduler", default="cos", type=str, choices=["linear", "cos", "fixed"])
        parser.add_argument("--optim", default="adamw", type=str, choices=["adamw", "adam"])
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument('--eval_epoch', default=1, type=int, help='evaluate each n epoch')
        parser.add_argument("--enable_sota", action="store_true", default=False)

        parser.add_argument('--margin', default=1, type=float, help='The fixed margin in loss function. ')
        parser.add_argument('--emb_dim', default=1000, type=int, help='The embedding dimension in KGE model.')
        parser.add_argument('--adv_temp', default=1.0, type=float, help='The temperature of sampling in self-adversarial negative sampling.')
        parser.add_argument("--contrastive_loss", default=0, type=int, choices=[0, 1])
        parser.add_argument('--clip', type=float, default=1.1, help='gradient clipping')

       
        parser.add_argument("--data_split", default="norm", type=str, help="Experiment split", choices=["dbp_wd_15k_V2", "dbp_wd_15k_V1", "zh_en", "ja_en", "fr_en", "norm", "test"])
        parser.add_argument("--hidden_units", type=str, default="300,300,300", help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for layers")
        parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")
        parser.add_argument("--distance", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')", choices=[1, 2])
        parser.add_argument("--il", type=int, default=0, choices=[0, 1],  help="Iterative learning?")
        parser.add_argument("--semi_learn_step", type=int, default=5, help="If IL, what's the update step?")
        parser.add_argument("--il_start", type=int, default=500, help="If Il, when to start?")
        parser.add_argument("--unsup", action="store_true", default=False)
        parser.add_argument("--unsup_k", type=int, default=1000, help="|visual seed|")

        
        parser.add_argument("--unsup_mode", type=str, default="img", help="unsup mode", choices=["img", "name", "char"])
        parser.add_argument("--tau", type=float, default=0.05, help="the temperature factor of contrastive loss")
        parser.add_argument("--alpha", type=float, default=0.2, help="the margin of InfoMaxNCE loss")
        parser.add_argument("--structure_encoder", type=str, default="gat", help="the encoder of structure view", choices=["gat", "gcn"])
        parser.add_argument("--ab_weight", type=float, default=0.5, help="the weight of NTXent Loss")
        parser.add_argument("--projection", action="store_true", default=False, help="add projection for model")
        parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
        parser.add_argument("--instance_normalization", action="store_true", default=False, help="enable instance normalization")
        parser.add_argument("--attr_dim", type=int, default=300, help="the hidden size of attr and rel features")
        parser.add_argument("--img_dim", type=int, default=300, help="the hidden size of img feature")
        parser.add_argument("--name_dim", type=int, default=300, help="the hidden size of name feature")
        parser.add_argument("--char_dim", type=int, default=300, help="the hidden size of char feature")

        parser.add_argument("--w_gcn", action="store_false", default=True, help="with gcn features")
        parser.add_argument("--w_rel", action="store_false", default=True, help="with rel features")
        parser.add_argument("--w_attr", action="store_false", default=True, help="with attr features")
        parser.add_argument("--w_name", action="store_false", default=True, help="with name features")
        parser.add_argument("--w_char", action="store_false", default=True, help="with char features")
        parser.add_argument("--w_img", action="store_false", default=True, help="with img features")
        parser.add_argument("--w_image_text", action="store_false", default=False, help="with img_text features")
        parser.add_argument("--use_surface", type=int, default=0, help="whether to use the surface")

        parser.add_argument("--inner_view_num", type=int, default=4, help="the number of inner view")
        parser.add_argument("--word_embedding", type=str, default="glove", help="the type of word embedding, [glove|fasttext]", choices=["glove", "bert"])
        # projection head
        parser.add_argument("--use_project_head", action="store_true", default=False, help="use projection head")
        parser.add_argument("--zoom", type=float, default=0.1, help="narrow the range of losses")
        parser.add_argument("--reduction", type=str, default="mean", help="[sum|mean]", choices=["sum", "mean"])

       
        parser.add_argument("--hidden_size", type=int, default=300, help="the hidden size of MEAformer")
        parser.add_argument("--intermediate_size", type=int, default=300, help="the hidden size of MEAformer")
        parser.add_argument("--num_attention_heads", type=int, default=1, help="the number of attention_heads of MEAformer")
        parser.add_argument("--num_hidden_layers", type=int, default=1, help="the number of hidden_layers of MEAformer")
        parser.add_argument("--position_embedding_type", default="absolute", type=str)
        parser.add_argument("--use_intermediate", type=int, default=1, help="whether to use_intermediate")
        parser.add_argument("--replay", type=int, default=0, help="whether to use replay strategy")
        parser.add_argument("--neg_cross_kg", type=int, default=0, help="whether to force the negative samples in the opposite KG")
        parser.add_argument("--with_tf", type=int, default =0 , help="whether to use transformer based network")

        parser.add_argument("--dim", type=int, default=100, help="the hidden size of MSNEA")
        parser.add_argument("--neg_triple_num", type=int, default=1, help="neg triple num")
        parser.add_argument("--use_bert", type=int, default=0)
        parser.add_argument("--use_attr_value", type=int, default=0)
       
        parser.add_argument('--rank', type=int, default=0, help='rank to dist')
        parser.add_argument('--dist', type=int, default=0, help='whether to dist')
        parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--world_size', default=3, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
        parser.add_argument("--local_rank", default=-1, type=int)

        self.cfg = parser.parse_args()
    
    '''
    根据不同的配置和约束条件，以及一些动态变量，调整配置参数，以满足具体的训练需求。
    '''

    def update_train_configs(self):
        # add some constraint for parameters
        assert not (self.cfg.save_model and self.cfg.only_test)

        # update some dynamic variable
        self.cfg.data_root = self.data_root
       
        if self.cfg.use_surface:
            self.cfg.w_name = True
            self.cfg.w_char = True
        else:
            self.cfg.w_name = False
            self.cfg.w_char = False

        if self.cfg.data_choice in ["FBYG15K", "FBDB15K"]:
            self.cfg.data_split = "norm"
            self.cfg.inner_view_num = 4
            self.cfg.w_name = False
            self.cfg.w_char = False
            self.cfg.use_surface = 0
            data_split_name = f"{self.cfg.data_rate}_"
        else:
            data_split_name = f"{self.cfg.data_split}_"
            if self.cfg.w_name and self.cfg.w_char:
                data_split_name = f"{data_split_name}with_surface_"
        
        if not self.cfg.freeze:
            self.cfg.freeze_epochs = 0
        
        self.cfg.exp_id = f"{self.cfg.exp_id}"
        self.cfg.data_path = osp.join(self.data_root, self.cfg.data_path)
        self.cfg.dump_path = osp.join(self.cfg.data_path, self.cfg.dump_path)
        
        if self.cfg.only_test == 1:
            self.save_model = 0
            self.dist = 0
        
        self.cfg.dim = self.cfg.attr_dim

        # use SOTA param
        if self.cfg.enable_sota:
            self.cfg.random_seed = 3408
            self.cfg.freeze = 1
            self.cfg.cross_modal = 1
            self.cfg.freeze_theta = 0.9
            self.cfg.lr = 1e-3
            if "EN_DE" in self.cfg.data_choice:
                self.cfg.freeze_theta = 0.1
            if not self.cfg.il:
                if 'DBP' or 'OEA' in self.cfg.data_choice:
                    self.cfg.epoch = 100
                    self.cfg.freeze_epochs = 50
                if 'FB' in self.cfg.data_choice:
                    print("FB in data choice")
                    self.cfg.epoch = 250
                    self.cfg.freeze_epochs = 150
            else:
                if 'DBP' in self.cfg.data_choice:
                    self.cfg.epoch = 600
                    self.cfg.freeze_epochs = 50
                    self.cfg.il_start = 100
                if 'OEA' in self.cfg.data_choice:
                    self.cfg.epoch = 750
                    self.cfg.freeze_epochs = 50
                    self.cfg.il_start = 250
                if 'FB' in self.cfg.data_choice:
                    self.cfg.epoch = 750
                    self.cfg.freeze_epochs = 150
                    self.cfg.il_start = 250
                                   
        return self.cfg
