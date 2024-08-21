import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .PMF_Loss import CustomMultiLossLayer, CKG_Loss, CM_Loss
from .PMF_Encoder import PMF_Encoder
from src.utils import pairwise_distances


class PMF(nn.Module):
    '''
    Progressivelt Modality Freezing for MMEA
    '''
    def __init__(self, kgs, args,train_set,test_set,logger):
        super().__init__()
        
        self.kgs = kgs
        self.args = args
        self.logger = logger
        self.test_ill = test_set
        self.train_ill = train_set
        self.cross_modal_key = {}
        self.transfer_label = {}
        
        self.missing_set = self.kgs['missing_img']
        self.img_features = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()
        self.image_text_features = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()
        self.input_idx = kgs["input_idx"].cuda()
        self.adj = kgs["adj"].cuda()
        self.rel_features = torch.Tensor(kgs["rel_features"]).cuda()
        self.att_features = torch.Tensor(kgs["att_features"]).cuda()
        self.name_features = None
        self.char_features = None
        
        if kgs["name_features"] is not None:
            self.name_features = kgs["name_features"].cuda()
            self.text_emb = self.name_features
            self.char_features = kgs["char_features"].cuda()
        
        img_dim = self._get_img_dim(kgs)
        char_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100

        self.modal_keys = [ 'image','structure','relation', 'attribute', 'name', 'char']
        self.loss_mask = {}
        self.memory_mask = {}
        self.ths = {}
        
        # Initialize relevance score and freezing thresholds for each modality
        for key in self.modal_keys:
            self.loss_mask[key] = torch.ones(kgs['ent_num'], dtype=torch.float).cuda()
            self.memory_mask[key] = torch.ones(kgs['ent_num'], dtype=torch.float).cuda()
            self.ths[key] = 0.1
            
        # set 0 weight for missing images features
        if self.args.freeze or self.args.cross_modal:
            for i in self.missing_set:
                self.loss_mask['image'][i] = 0
        
        self.multimodal_encoder = PMF_Encoder(args=self.args,
                                            ent_num=kgs["ent_num"],
                                            img_feature_dim=img_dim,
                                            char_feature_dim=char_dim,
                                            use_project_head=self.args.use_project_head,
                                            attr_input_dim=kgs["att_features"].shape[1],
                                            img_features = self.img_features)
        if self.args.use_surface:
            self.multi_loss_layer = CustomMultiLossLayer(6)
            self.multi_loss_layer_2 = CustomMultiLossLayer(15)
        else:
            self.multi_loss_layer = CustomMultiLossLayer(4)
            self.multi_loss_layer_2 = CustomMultiLossLayer(6)
       
        self.criterion_cl = CKG_Loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        self.criterion_cl_joint = CKG_Loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        
        if self.args.cross_modal:
            self.cross_modal_loss = CM_Loss(tau=self.args.cross_modal_tau, ab_weight=self.args.ab_weight)
        

    def forward(self, batch, epoch, batch_no):
        self.loss_dic = {}
        self.epoch = epoch
        
        emb_dict = self.joint_emb_generat()
        gph_emb = emb_dict.get('structure', None)
        img_emb = emb_dict.get('image', None)
        rel_emb = emb_dict.get('relation', None)
        att_emb = emb_dict.get('attribute', None)
        name_emb = emb_dict.get('name', None)
        char_emb = emb_dict.get('char_name', None)
        joint_emb = emb_dict.get('joint', None)

        loss_joi = 0.
        loss_joi = self.criterion_cl_joint(joint_emb, batch)
        
        emb_dict =  {"structure": gph_emb, "image":img_emb, "relation":rel_emb, "attribute": att_emb, "name": name_emb,\
                    "char": char_emb}
        emb_dict = {key: F.normalize(value) for key, value in emb_dict.items() if value is not None}     

        if self.args.freeze and batch_no == 0 and self.epoch < self.args.freeze_epochs and self.epoch >= self.args.freeze_start: 
            self.modal_freezing(self.train_ill, self.test_ill, emb_dict)    
            
        in_loss = 0.
        in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb,batch)

        cross_loss = 0.
        
        loss_all = in_loss + loss_joi  + cross_loss
        self.loss_dic.update({"loss_all": loss_all,"joint_loss": loss_joi, "intra_modal_loss": in_loss, "cross_modal_loss": cross_loss  })
        
        
        output = {"loss_dic": self.loss_dic,"loss_all": loss_all, "gph_emb": gph_emb, "img_emb":img_emb, "rel_emb": rel_emb, \
            "att_emb": att_emb, "name_emb": name_emb, "char_emb":char_emb,"joint_emb": joint_emb}
        
        return  loss_all, output

    def generate_hidden_emb(self, hidden):
        i = 0
        embs = []
        
        if not self.args.w_gcn:
            gph_emb = None
        else:
            gph_emb = F.normalize(hidden[:, i, :].squeeze(1))
            embs.append(gph_emb)
            i+=1
        if not self.args.w_rel:
            rel_emb = None
        else:
            rel_emb = F.normalize(hidden[:, i, :].squeeze(1))
            embs.append(rel_emb)
            i+=1
        if not self.args.w_attr:
            att_emb = None
        else:
            att_emb = F.normalize(hidden[:, i, :].squeeze(1))
            embs.append(att_emb)
            i+=1
        if not self.args.w_img:
            img_emb = None
        else:
            img_emb = F.normalize(hidden[:, i, :].squeeze(1))
            embs.append(img_emb)
            i+=1
        if not self.args.w_name:
            name_emb = None
        else:
            name_emb = F.normalize(hidden[:, i, :].squeeze(1))
            embs.append(name_emb)
            i+=1
        if not self.args.w_char:
            char_emb = None
        else:
            char_emb = F.normalize(hidden[:, i, :].squeeze(1))
            embs.append(char_emb)
            i+=1
        
        joint_emb = torch.cat(embs, dim=1)
        
        return gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, joint_emb
 
        
    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill):
        
        if self.args.frozen:
            # Compute Cross-KG alignment loss for each modality with relevance score(stop gradient flag)
            loss_GCN = self.criterion_cl(gph_emb, train_ill, self.loss_mask['structure']) if gph_emb is not None else 0
            loss_rel = self.criterion_cl(rel_emb, train_ill, self.loss_mask['relation']) if rel_emb is not None else 0
            loss_att = self.criterion_cl(att_emb, train_ill, self.loss_mask['attribute']) if att_emb is not None else 0
            loss_img = self.criterion_cl(img_emb, train_ill, self.loss_mask['image']) if img_emb is not None else 0
            loss_name = self.criterion_cl(name_emb, train_ill, self.loss_mask['name']) if name_emb is not None else 0
            loss_char = self.criterion_cl(char_emb, train_ill, self.loss_mask['char']) if char_emb is not None else 0
        else:    
            loss_GCN = self.criterion_cl(gph_emb, train_ill) if gph_emb is not None else 0
            loss_rel = self.criterion_cl(rel_emb, train_ill) if rel_emb is not None else 0
            loss_att = self.criterion_cl(att_emb, train_ill) if att_emb is not None else 0
            loss_img = self.criterion_cl(img_emb, train_ill) if img_emb is not None else 0
            loss_name = self.criterion_cl(name_emb, train_ill) if name_emb is not None else 0
            loss_char = self.criterion_cl(char_emb, train_ill) if char_emb is not None else 0
        
        if len(train_ill) > self.args.batch_size * 1:
            self.loss_dic.update({"test_gph_loss": loss_GCN, "test_rel_loss": loss_rel, "test_att_loss": loss_att, "test_img_loss": loss_img, 
                                  "test_name_loss": loss_name, "test_char_loss": loss_char})
        else:                          
            self.loss_dic.update({"gph_loss": loss_GCN, "rel_loss": loss_rel, "att_loss": loss_att, "img_loss": loss_img, \
                "name_loss": loss_name, "char_loss": loss_char})
        loss = [loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char]
        
        loss = [l for l in loss if l != 0]
        total_loss = self.multi_loss_layer(loss)
        # total_loss = sum(loss)
        
        return total_loss
    
    # Calculate contrastive loss between two modalitity graph
    def cross_view_loss(self,train_ill):
        self.embs = self.joint_emb_generat()
        
        gph_emb = self.embs.get('structure', None)
        img_emb = self.embs.get('image', None)
        rel_emb = self.embs.get('relation', None)
        att_emb = self.embs.get('attribute', None)
        name_emb = self.embs.get('name', None)
        char_emb = self.embs.get('char_name', None)
        
        embs = [img_emb,gph_emb,rel_emb,att_emb,name_emb,char_emb]
        keys = self.modal_keys
        embs = [emb for emb in embs if emb is not None]
        
        modal_weight = self.args.cross_modal_weight
        loss = []
        total_loss = 0
        
        # Compute pairwise cross-modal losses
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                if self.args.frozen:
                    loss_j = modal_weight[i]*modal_weight[j]*self.cross_modal_loss(embs[i], embs[j], train_ill,
                                                    self.loss_mask[keys[i]], self.loss_mask[keys[j]])
                else:
                    loss_j = modal_weight[i]*modal_weight[j]*self.cross_modal_loss(embs[i], embs[j], train_ill)
                loss.append(loss_j)
                total_loss+=loss_j
        total_loss = self.multi_loss_layer_2(loss)
        # total_loss = sum(loss)
        
        return total_loss
    
    
    def joint_emb_generat(self):
        gph_emb, img_emb, rel_emb, att_emb, \
            name_emb, char_emb, joint_emb, hidden_states, weight_norm = \
                                    self.multimodal_encoder(self.input_idx,
                                                            self.adj,
                                                            self.loss_mask,
                                                            self.img_features,
                                                            self.rel_features,
                                                            self.att_features,
                                                            self.name_features,
                                                            self.char_features)
       
        return {"joint":joint_emb, "structure": gph_emb, "image":img_emb, "relation":rel_emb, "attribute": att_emb, "name": name_emb,\
            "char_name": char_emb,"hidden_states":hidden_states,"weight_norm": weight_norm}
                

    def _get_img_dim(self, kgs):
        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        elif isinstance(kgs["images_list"], np.ndarray) or torch.is_tensor(kgs["images_list"]):
            img_dim = kgs["images_list"].shape[1]
        return img_dim

    def Iter_new_links(self, epoch, left_non_train, final_emb, right_non_train, new_links=[]):
        if len(left_non_train) == 0 or len(right_non_train) == 0:
            return new_links
        distance_list = []
        for i in np.arange(0, len(left_non_train), 1000):
            d = pairwise_distances(final_emb[left_non_train[i:i + 1000]], final_emb[right_non_train])
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        if (epoch + 1) % (self.args.semi_learn_step * 5) == self.args.semi_learn_step:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if preds_r[p] == i]
        else:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if (preds_r[p] == i) and ((left_non_train[i], right_non_train[p]) in new_links)]

        return new_links

    def data_refresh(self, logger, train_ill, test_ill_, left_non_train, right_non_train, new_links=[]):
        if len(new_links) != 0 and (len(left_non_train) != 0 and len(right_non_train) != 0):
            new_links_select = new_links
            train_ill = np.vstack((train_ill, np.array(new_links_select)))
            num_true = len([nl for nl in new_links_select if nl in test_ill_])
            # remove from left/right_non_train
            for nl in new_links_select:
                left_non_train.remove(nl[0])
                right_non_train.remove(nl[1])

            if self.args.rank == 0:           
                logger.info(f"#new_links_select:{len(new_links_select)}")
                logger.info(f"train_ill.shape:{train_ill.shape}")
                logger.info(f"#true_links: {num_true}")
                logger.info(f"true link ratio: {(100 * num_true / len(new_links_select)):.1f}%")
                logger.info(f"#entity not in train set: {len(left_non_train)} (left) {len(right_non_train)} (right)")

            new_links = []
        else:
            logger.info("len(new_links) is 0")

        return left_non_train, right_non_train, train_ill, new_links

    def modal_freezing(self, train_links, test_links, emb_dic):
        with torch.no_grad():
            self.logger.info(f"---------modality_freezing--------")
            # Freeze modalities based on dynamic thresholds
            for freeze_key in self.modal_keys:
                if freeze_key not in emb_dic:
                    continue
                tau = 1
                k = 1
                abs_th = self.ths[freeze_key]
                self.ths[freeze_key] = self.get_threshold(self.ths[freeze_key], freeze_key)

                freeze_emb = F.normalize(emb_dic[freeze_key])
                sim_mat = {}
                sim = []
                for i,j in train_links:
                    sim.append(torch.mm(freeze_emb[i].unsqueeze(0), freeze_emb[j].unsqueeze(0).t())/tau)
                sim_mat = torch.cat(sim, dim=0).squeeze()
                   
                test_dis = torch.matmul(freeze_emb[test_links[:,0]], freeze_emb[test_links[:,1]].transpose(0,1))/tau
               
                test_dis_l,_ = torch.sort(test_dis, dim=1, descending=True)
                test_dis_l = torch.mean(test_dis_l[:,:k], dim=1)
                test_dis_r,_ = torch.sort(test_dis, dim=0, descending=True)
                test_dis_r = torch.mean(test_dis_r[:k,:], dim=0)
                
                # Update loss masks based on freeze rule
                if self.args.freeze_rule  == 'abs':
                    self.loss_mask[freeze_key][test_links[:,0]] = (test_dis_l >=  abs_th).float() 
                    self.loss_mask[freeze_key][test_links[:,1]] = (test_dis_r >=  abs_th).float()
                elif self.args.freeze_rule  == 'abs_relu':
                    self.loss_mask[freeze_key][train_links[:,0]] = self.loss_mask[freeze_key][train_links[:,1]] = torch.relu((sim_mat - abs_th)/(torch.max(sim_mat) - abs_th))
                    self.loss_mask[freeze_key][test_links[:,0]] = torch.relu((test_dis_l - abs_th) / (torch.max(test_dis_l) -abs_th))
                    self.loss_mask[freeze_key][test_links[:,1]] = torch.relu((test_dis_r - abs_th) / (torch.max(test_dis_r) -abs_th))
                
                # Special handling for missing 'image' modality
                if freeze_key == 'image':
                    for i in self.missing_set:  
                        self.loss_mask[freeze_key][i] = 0
                
                # Apply memory mechanism to frozen modalities, frozen modality feature  will not be updated
                self.loss_mask[freeze_key] = self.loss_mask[freeze_key]*self.memory_mask[freeze_key]
                self.memory_mask[freeze_key] = (self.loss_mask[freeze_key] > 0)
                
                self.logger.info(f'---------{freeze_key}--------')
                self.logger.info(f'freeze_rule: {self.args.freeze_rule}')
                self.logger.info(f'abs_th: {abs_th}')
                self.logger.info(f'{freeze_key} mean_confidence :{torch.mean(self.loss_mask[freeze_key])}')    
                self.logger.info(f'filted ratio: {torch.sum(self.loss_mask[freeze_key] == 0)/len(self.loss_mask[freeze_key])}' )
                self.logger.info(f'filted ratio in train_set: {torch.sum(self.loss_mask[freeze_key][train_links[:,1]] == 0)/len(self.loss_mask[freeze_key][train_links[:,1]])}' )
                self.logger.info(f'filted ratio in test_set: {( torch.sum(self.loss_mask[freeze_key][test_links[:,0]] == 0) + torch.sum(self.loss_mask[freeze_key][test_links[:,1]] == 0))/(2*len(self.loss_mask[freeze_key][test_links[:,1]]))}' )

    
    def get_threshold(self, th, modal):
        # Adjust freezing threshold during training
        th = th*self.args.growing_rate
        th = th = min(self.args.freeze_theta, th) if modal == 'image' else min(0.1, th)
        return th
    