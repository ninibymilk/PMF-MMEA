import os
import os.path as osp
import torch
import numpy as np
from torch.serialization import save
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from easydict import EasyDict as edict
from tqdm import tqdm
import pdb
import json
from collections import defaultdict
import torch.nn as nn
import math
import csv

from torchlight.utils import set_seed
from torchlight.logger import get_dump_path, initialize_exp
from config import cfg
from src.data import load_data, Collator_base, EADataset
from src.utils import set_optim, Loss_log, pairwise_distances, csls_sim
from model import PMF
import matplotlib.pyplot as plt
import numpy as np

from src.distributed_utils import init_distributed_mode, dist_pdb, is_main_process, reduce_value, cleanup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import scipy
import gc
import copy


class Runner:
    def __init__(self, args, writer=None, logger=None, rank=0):
        self.datapath = edict()
        self.datapath.log_dir = get_dump_path(args)
        self.datapath.model_dir = os.path.join(self.datapath.log_dir, 'model')
        self.rank = rank
        self.args = args
        self.writer = writer
        self.logger = logger
        self.scaler = GradScaler()
        self.model_list = []
        set_seed(args.random_seed)
        self.data_init()
        self.model_choise()
        set_seed(args.random_seed)

        if self.args.only_test :
            self.dataloader_init(test_set=self.test_set)
        else:
            if self.args.cross_modal:
                self.dataloader_init(train_set=self.train_set, eval_set=self.eval_set, test_set=self.test_set,ent_set=self.ent_set)
            else:
                self.dataloader_init(train_set=self.train_set, eval_set=self.eval_set, test_set=self.test_set)
            if self.args.dist:
                self.model_sync()
            else:
                self.model_list = [self.model]
 
            if self.args.freeze:
                train_epoch_1_stage = self.args.freeze_epochs
            else:
                if self.args.il:
                    assert self.args.il_start < self.args.epoch
                    train_epoch_1_stage = self.args.il_start
                else:
                    train_epoch_1_stage = self.args.epoch
            self.optim_init(self.args, total_epoch=train_epoch_1_stage)
    
    def model_sync(self):
        folder = osp.join(self.args.data_path, "tmp")
        if not os.path.exists(folder):
            os.makedirs(folder)
        checkpoint_path = osp.join(folder, "initial_weights.pt")
        if self.rank == 0:
            torch.save(self.model.state_dict(), checkpoint_path)
        dist.barrier()
        self.model = self._model_sync(self.model, checkpoint_path)

    def _model_sync(self, model, checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.args.device))
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.args.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=True)
        self.model_list.append(model)
        model = model.module
        return model

    def model_choise(self):
        assert self.args.model_name in ["PMF"]
        
        if self.args.model_name == "PMF":
            self.model = PMF(self.KGs, self.args,self.train_set,self.test_set,self.logger)

        self.model = self._load_model(self.model, model_name=self.args.model_name_save)
      
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"total params num: {total_params}")
    
    def optim_init(self, 
                   opt, is_cross_modal = False,total_step=None, total_epoch=None, accumulation_step=None):
        step_per_epoch = len(self.train_dataloader)
        if is_cross_modal:
            step_per_epoch += len(self.test_dataloader)
        if total_epoch is not None:
            opt.total_steps = int(step_per_epoch * total_epoch)
        else:
            opt.total_steps = int(step_per_epoch * opt.epoch) if total_step is None else int(total_step)
        
        opt.warmup_steps = int(opt.total_steps * 0.15)
        
        if self.rank == 0 and total_step is None:
            self.logger.info(f"warmup_steps: {opt.warmup_steps}")
            self.logger.info(f"total_steps: {opt.total_steps}")
            self.logger.info(f"weight_decay: {opt.weight_decay}")
        freeze_part = []
        no_decay = []
        large_lr = []
           
        self.optimizer, self.scheduler = set_optim(opt, self.model_list, freeze_part,no_decay,large_lr, accumulation_step)

    def data_init(self):
        self.KGs, self.non_train, self.train_set, self.test_set, self.eval_set, self.test_ill_,self.ent_set = load_data(self.logger, self.args)
        self.train_ill = self.train_set.data
        self.train_left = torch.LongTensor(self.train_set[:, 0].squeeze()).cuda()
        self.train_right = torch.LongTensor(self.train_set[:, 1].squeeze()).cuda()
        self.eval_left = torch.LongTensor(self.test_set[:, 0].squeeze()).cuda()
        self.eval_right = torch.LongTensor(self.test_set[:, 1].squeeze()).cuda()
        if self.test_set is not None:
            self.test_left = torch.LongTensor(self.test_set[:, 0].squeeze()).cuda()
            self.test_right = torch.LongTensor(self.test_set[:, 1].squeeze()).cuda()

        self.eval_sampler = None
        if self.args.dist and not self.args.only_test:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
            self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.eval_set)
            self.ent_sampler = torch.utils.data.distributed.DistributedSampler(self.ent_set)
            if self.test_set is not None:
                self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_set)

    def dataloader_init(self, train_set=None, eval_set=None, test_set=None,ent_set = None):
        bs = self.args.batch_size
        collator = Collator_base(self.args)
        if self.args.dist and not self.args.only_test:
            self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
            if train_set is not None:
                self.train_dataloader = self._dataloader_dist(train_set, self.train_sampler, bs, collator)
            if test_set is not None:
                self.test_dataloader = self._dataloader_dist(test_set, self.test_sampler, bs, collator)
            if eval_set is not None:
                self.eval_dataloader = self._dataloader_dist(eval_set, self.eval_sampler, bs, collator)
            if ent_set is not None:
                self.ent_dataloader = self._dataloader_dist(ent_set, self.ent_sampler, bs, collator)
        else:
            self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
            if train_set is not None:
                self.train_dataloader = self._dataloader(train_set, bs, collator)
            if test_set is not None:
                self.test_dataloader = self._dataloader(test_set, bs, collator)
            if eval_set is not None:
                self.eval_dataloader = self._dataloader(eval_set, bs, collator)
            if ent_set is not None:
                self.ent_dataloader = self._dataloader(ent_set, bs, collator)

        
    def _dataloader_dist(self, train_set, train_sampler, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=self.args.workers,
            persistent_workers=True,  # True
            drop_last=True,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def _dataloader(self, train_set, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            num_workers=self.args.workers,
            persistent_workers=True,  # True
            shuffle=(self.args.only_test == 0),
            # drop_last=(self.args.only_test == 0),
            drop_last=False,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def run(self):
        self.loss_log = Loss_log()
        self.curr_loss = 0.
        self.lr = self.args.lr
        self.curr_loss_dic = defaultdict(float)
        self.weight = np.ones(self.args.inner_view_num, dtype=np.float32)
        self.loss_weight = [1, 1]
        self.loss_item = 99999.
        self.step = 1
        self.epoch = 0
        self.new_links = []
        self.best_model_wts = None

        self.best_mrr = 0
        self.early_stop = 0
        self.early_stop_init = 1000
        self.early_stop_count = self.early_stop_init
        self.stage = 0

        self.freeze_stop_count = 0
        self.freeze_ratio = 0.
        
        with tqdm(total=self.args.epoch) as _tqdm:
            for i in range(self.args.epoch):
                # _tqdm.set_description(f'Train | epoch {i} Loss {self.loss_log.get_loss():.5f} Acc {self.loss_log.get_acc()*100:.3f}%')
                if self.args.dist and not self.args.only_test:
                    self.train_sampler.set_epoch(i)
                # -------------------------------Begin iteration--------------------------------
                self.epoch = i
                if self.args.il and (self.epoch == self.args.il_start and self.stage == 0) or (self.early_stop_count <= 0 and self.epoch <= self.args.il_start):
                    if self.early_stop_count <= 0:
                        logger.info(f"Early stop in epoch {self.epoch}... Begin iteration....")
                    self.stage = 1
                    self.early_stop_init = 2000
                    self.early_stop_count = self.early_stop_init
                    self.eval_epoch = 1
                    self.step = 1
                    self.args.lr = 1e-4
                    # self.args.weight_decay = 0.0001
                    self.optim_init(self.args, total_epoch=(self.args.epoch - self.args.il_start) * 3)
                    if self.best_model_wts is not None:
                        self.logger.info("load from the best model before IL... ")
                        self.model.load_state_dict(self.best_model_wts)
                    name = self._save_name_define() 
                    self.test(save_name=f"{name}_test_ep{self.args.epoch}_no_iter")
                    if self.rank == 0:
                        if not self.args.only_test and self.args.save_model:
                            self._save_model(self.model, input_name=f"{name}_non_iter")
                # count when to stop freezing
                if self.args.freeze and self.epoch < self.args.freeze_epochs:
                    freeze_ratio = torch.sum(self.model.loss_mask['image'] == 0)/len(self.model.loss_mask['image'])
                    if (abs(freeze_ratio-self.freeze_ratio) < 1e-5):
                            self.freeze_stop_count += 1
                    else:
                        self.freeze_stop_count = 0
                    self.freeze_ratio = freeze_ratio
                # stop freezing and apply cross-modal loss
                if (self.freeze_stop_count >= 500  or self.epoch == self.args.freeze_epochs) and self.args.cross_modal:
                    if self.args.il:
                        train_epoch_2_stage = self.args.epoch - self.args.il_start - self.args.freeze_epochs
                    else:
                        train_epoch_2_stage = self.args.epoch - self.args.freeze_epochs
                    if self.args.freeze:
                        self.args.lr = self.args.cm_lr
                    self.step = 1
                    self.early_stop_count = self.early_stop_init
                    # self.args.weight_decay = 3
                    self.optim_init(self.args, total_epoch = train_epoch_2_stage, is_cross_modal=True)
                    if self.best_model_wts is not None:
                        self.logger.info("load from the best model after freezing... ")
                        self.model.load_state_dict(self.best_model_wts)
                        self.model.loss_mask = self.best_loss_mask
                        self.eval()
                        
                if self.stage == 1 and (self.epoch + 1) % self.args.semi_learn_step == 0 and self.args.il:
                    self.il_for_ea()

                if self.stage == 1 and (self.epoch + 1) % (self.args.semi_learn_step * 10) == 0 and len(self.new_links) != 0 and self.args.il:
                    self.il_for_data_ref()
                
                self.train(_tqdm)
                self.loss_log.update(self.curr_loss)
                self.loss_item = self.loss_log.get_loss()
                _tqdm.set_description(f'Train | Ep [{self.epoch}/{self.args.epoch}] Step [{self.step}/{self.args.total_steps}] LR [{self.lr:.5f}] Loss {self.loss_log.get_loss():.5f} ')
                self.update_loss_log()
                if (i + 1) % self.args.eval_epoch == 0:
                    self.eval()
                _tqdm.update(1)
                if self.stage == 1 and self.early_stop_count <= 0:
                    logger.info(f"Early stop in epoch {self.epoch}")
                    break

        name = self._save_name_define()
        if self.best_model_wts is not None:
            self.logger.info("load from the best model before final testing ... ")
            self.model.load_state_dict(self.best_model_wts)
            self.model.loss_mask = self.best_loss_mask
        self.test(save_name=f"{name}_test_ep{self.args.epoch}")

        if self.rank == 0:
            self.logger.info(f"min loss {self.loss_log.get_min_loss()}")
            if not self.args.only_test and self.args.save_model:
                self._save_model(self.model, input_name=name)
    
    def _load_model(self, model, model_name=None):
        if model_name is None:
            model_name = self.args.model_name_save
        save_path = osp.join(self.args.data_path, self.args.model_name, 'save')
        save_path = osp.join(save_path, f'{model_name}.pkl')
        if len(model_name) > 0 and not os.path.exists(save_path):
            print(f"not exists {model_name} !! ")
            pdb.set_trace()
        if (len(model_name) == 0 or not os.path.exists(save_path)) and self.rank == 0:
            if len(model_name) > 0:
                self.logger.info(f"{model_name}.pkl not exist!!")
            else:
                self.logger.info("Random init...")
            model.cuda()
            return model
        if 'Dist' in self.args.model_name:
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path, map_location=self.args.device).items()})
        else:
            if self.args.model_name == 'PMF':
                model.load_state_dict(torch.load(save_path, map_location=self.args.device)['model_state_dict'])
                model.loss_mask = torch.load(save_path, map_location=self.args.device)['loss_mask']
            else:
                model.load_state_dict(torch.load(save_path, map_location=self.args.device))

        model.cuda()
        if self.rank == 0:
            self.logger.info(f"loading model [{model_name}.pkl] done!")

        return model

    def _save_model(self, model, input_name=""):

        model_name = self.args.model_name

        save_path = osp.join(self.args.data_path, model_name, 'save')
        os.makedirs(save_path, exist_ok=True)

        if input_name == "":
            input_name = self._save_name_define()
        save_path = osp.join(save_path, f'{input_name}.pkl')

        if model is None:
            return
        if self.args.save_model:
            if model_name == 'PMF':
                save_dict = {
                'model_state_dict': model.state_dict(),
                'loss_mask': model.loss_mask
            }
                torch.save(save_dict, save_path)
            else:
                torch.save(model.state_dict(), save_path)
            self.logger.info(f"saving [{save_path}] done!")

        return save_path

    def il_for_ea(self):
        with torch.no_grad():
            if self.args.model_name in ["PMF"]:
                emb_dic = self.model.joint_emb_generat()
                final_emb = emb_dic.get('joint',None)
            final_emb = F.normalize(final_emb)
            self.new_links = self.model.Iter_new_links(self.epoch, self.non_train["left"], final_emb, self.non_train["right"], new_links=self.new_links)
            if (self.epoch + 1) % (self.args.semi_learn_step * 5) == 0:
                self.logger.info(f"[epoch {self.epoch}] #links in candidate set: {len(self.new_links)}")
   
    def il_for_data_ref(self):
        self.non_train["left"], self.non_train["right"], self.train_ill, self.new_links = self.model.data_refresh(
            self.logger, self.train_ill, self.test_ill_, self.non_train["left"], self.non_train["right"], new_links=self.new_links)
        set_seed(self.args.random_seed)
        self.train_set = EADataset(self.train_ill)
        self.dataloader_init(train_set=self.train_set)
        # one time train

    def _save_name_define(self):
        prefix = ""
        if self.args.dist:
            prefix = f"dist_{prefix}"
        if self.args.il:
            prefix = f"il{self.args.epoch-self.args.il_start}_b{self.args.il_start}_{prefix}"
        name = f'{self.args.exp_id}_{prefix}'
        return name

    def train(self, _tqdm):
        self.model.train()
        curr_loss = 0.
        self.loss_log.acc_init()
        accumulation_steps = self.args.accumulation_steps
        # torch.cuda.empty_cache()
        batch_no = 0
        def process_batch(batch, is_cross_modal=False):
            if is_cross_modal:
                loss = self.model.cross_view_loss(batch)
            else:
                loss, output = self.model(batch,self.epoch, batch_no)
            loss = loss / accumulation_steps
            self.scaler.scale(loss).backward()
            if self.args.dist:
                loss = reduce_value(loss, average=True)
            self.step += 1
            if not self.args.dist or is_main_process():
                if is_cross_modal:
                    self.curr_loss_dic['cross_modal_loss'] += loss
                    output = None
                self.output_statistic(loss, output)
            if self.step % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                for model in self.model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skip_lr_sched = (scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    self.scheduler.step()

                if not self.args.dist or is_main_process():
                    self.lr = self.scheduler.get_last_lr()[-1]
                    self.writer.add_scalars("lr", {"lr": self.lr}, self.step)
                for model in self.model_list:
                    model.zero_grad(set_to_none=True)

            if self.args.dist:
                torch.cuda.synchronize(self.args.device)
           
            return loss.item()
            
        for batch in self.train_dataloader:
             process_batch(batch)
             batch_no += 1
            
        if self.args.cross_modal and self.epoch >= self.args.freeze_epochs and self.epoch < self.args.il_start:
            for batch in self.ent_dataloader:
                process_batch(batch, is_cross_modal=True)
                batch_no += 1    
        
    def output_statistic(self, loss, output):
        self.curr_loss += loss.item()
        if output is None:
            return
   
        if 'loss_dic' in output:
            for key in output['loss_dic'].keys():
                self.curr_loss_dic[key] += output['loss_dic'][key]
        if 'weight' in output:          
          if output['weight'] is not None:
            self.weight = output['weight']
        if 'loss_weight' in output:
            print("loss_weight is not None")
            if output['loss_weight'] is not None:
                self.loss_weight = output['loss_weight']

    def update_loss_log(self):
        vis_dict = {"train_loss": self.curr_loss}
        vis_dict.update(self.curr_loss_dic)
        self.writer.add_scalars("loss", vis_dict, self.step)
        for key in self.curr_loss_dic:
            self.logger.info(f"loss/{key}: {self.curr_loss_dic[key]}")
    
        if self.loss_weight is not None and self.loss_weight != [1, 1]:
            weight_dic = {}
            weight_dic["mask"] = 1 / (self.loss_weight[0]**2)
            weight_dic["kpi"] = 1 / (self.loss_weight[1]**2)
            self.writer.add_scalars("loss_weight", weight_dic, self.step)

        self.curr_loss = 0.
        for key in self.curr_loss_dic:
            self.curr_loss_dic[key] = 0.

    def eval(self, last_epoch=False, save_name=""):
        test_left = self.eval_left
        test_right = self.eval_right
        train_left = self.train_left
        train_right = self.train_right
        self.model.eval() 
        
        self.logger.info(" --------------------- Eval result --------------------- ")
        self._test(test_left, test_right, last_epoch=last_epoch, save_name=save_name)

    # one time test
    def test(self, save_name="", last_epoch=True):
        if self.test_set is None:
            test_left = self.eval_left
            test_right = self.eval_right
        else:
            test_left = self.test_left
            test_right = self.test_right
        train_left = self.train_left
        train_right = self.train_right
        self.model.eval()
        self.logger.info(" --------------------- Test result --------------------- ")
        self._test(test_left, test_right, last_epoch=last_epoch, save_name=save_name)
            
                
    def _test(self, test_left, test_right, last_epoch=False, save_name="", loss=None):
        with torch.no_grad():
            w_normalized = None
            if self.args.model_name in ["PMF"]:
                embs_dic = self.model.joint_emb_generat()
                final_emb = embs_dic.get('joint',None)
                test_loss = self.model.criterion_cl_joint(final_emb,self.test_set,mask=None)
                self.writer.add_scalars("loss", {"test_loss": test_loss}, self.epoch)
                weight_norm = None
            else:
                final_emb = self.model.joint_emb_generat()
                weight_norm = None
        time_now = datetime.now().strftime("%m-%d %H:%M")       
            
        # pdb.set_trace()
        top_k = [1, 10, 50]
        acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
        acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
        test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
                
        
        if self.args.model_name in ["PMF"]:
            distance = torch.matmul(final_emb[test_left], final_emb[test_right].transpose(0,1))
        
        if self.args.csls is True:
           distance = csls_sim(distance, self.args.csls_k,self.args.m_csls)
                
        test_left_np = test_left.cpu().numpy()
        test_right_np = test_right.cpu().numpy()
        if last_epoch:
            to_write = []
            to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3", "v1", "v2", "v3"])
        
        aligned = []
        for idx in range(test_left.shape[0]):
            values, indices = torch.sort(distance[idx, :], descending=True)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()                               
            mean_l2r += (rank + 1)
            mrr_l2r += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_l2r[i] += 1
            if rank == 0:
                aligned.append(test_left_np[idx])
            if last_epoch:
                indices = indices.cpu().numpy()
                to_write.append([idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]], test_right_np[indices[1]],
                                test_right_np[indices[2]], round(values[0].item(), 4), round(values[1].item(), 4), round(values[2].item(), 4)])
                to_write.append([idx, rank, self.KGs['id2ent'][test_left_np[idx]], self.KGs['id2ent'][test_right_np[idx]],\
                    self.KGs['id2ent'][test_right_np[indices[0]]], self.KGs['id2ent'][test_right_np[indices[1]]],self.KGs['id2ent'][test_right_np[indices[2]]],\
                        round(values[0].item(), 4), round(values[1].item(), 4), round(values[2].item(), 4)])
        
        # write alignment results and modality weight to csv file
        if last_epoch:
            if save_name == "":
                save_name = self.args.model_name
            save_pred_path = osp.join(self.args.data_path, self.args.model_name, f"{save_name}_pred")
            os.makedirs(save_pred_path, exist_ok=True)
            import csv
            with open(osp.join(save_pred_path, f"{self.args.model_name}_{self.args.data_choice}_{self.args.data_split}_{time_now}_pred.txt"), "w") as f:
                wr = csv.writer(f, dialect='excel')
                wr.writerows(to_write)
            if w_normalized is not None:
                with open(osp.join(save_pred_path, f"{self.args.model_name}_{self.args.data_choice}_{self.args.data_split}_{time_now}_wight.json"), "w") as fp:
                    json.dump(w_normalized.cpu().tolist(), fp)
            if weight_norm is not None:
                wight= weight_norm.cpu().numpy()
                import csv
                with open(osp.join(save_pred_path, f"{self.args.model_name}_{self.args.data_choice}_{self.args.data_split}_{time_now}_wight_dic.csv"), "w") as fc:
                    writer = csv.writer(fc)
                    index = ['img', 'attr', 'rel', 'graph', 'name', 'char']
                    index = index[:wight.shape[1]]
                    writer.writerow(index)
                    for wi  in wight:
                        writer.writerow(wi)                   
            # report model performance to csv file
            results = []
            

                    
        for idx in range(test_right.shape[0]):
            _, indices = torch.sort(distance[:, idx], descending=True)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_r2l += (rank + 1)
            mrr_r2l += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_r2l[i] += 1
       
        mean_l2r /= test_left.size(0)
        mean_r2l /= test_right.size(0)
        mrr_l2r /= test_left.size(0)
        mrr_r2l /= test_right.size(0)
        
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / test_left.size(0), 4)
            acc_r2l[i] = round(acc_r2l[i] / test_right.size(0), 4)
        gc.collect()
        if not self.args.only_test:
            Loss_out = f", Loss = {self.loss_item:.4f}"
        else:
            Loss_out = ""
            self.epoch = "Test"
            self.early_stop_count = 1

        if self.rank == 0:
            self.logger.info(f"Ep {self.epoch} | l2r: acc of top {top_k} = {acc_l2r}, mr = {mean_l2r:.3f}, mrr = {mrr_l2r:.3f}{Loss_out}")
            self.logger.info(f"Ep {self.epoch} | r2l: acc of top {top_k} = {acc_r2l}, mr = {mean_r2l:.3f}, mrr = {mrr_r2l:.3f}{Loss_out}")
            self.early_stop_count -= 1
        
                   
        if not self.args.only_test and not last_epoch and mrr_l2r <= max(self.loss_log.acc):
            self.early_stop += 1
        
        if not self.args.only_test :
            
            self.writer.add_scalars("acc_l2r", {"hit@1":max(acc_l2r[0],acc_r2l[0])}, self.epoch)
            self.writer.add_scalars("mrr",{"mrr": max(mrr_l2r,mrr_r2l)}, self.epoch)
            self.writer.add_scalars("freeze_ratio",{"image_freeze_ratio":self.freeze_ratio}, self.epoch)
        
            
        if not self.args.only_test and max(mrr_r2l,mrr_l2r) > max(self.loss_log.acc) and not last_epoch:
            self.logger.info(f"Best model update in Ep {self.epoch}: MRR from [{max(self.loss_log.acc)}] --> [{max(mrr_r2l,mrr_l2r)}] ... ")
            self.loss_log.update_acc(max(mrr_r2l,mrr_l2r))
            self.early_stop_count = self.early_stop_init
            self.early_stop = 0
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            self.best_loss_mask = self.model.loss_mask
        
        # Model performance report
        if last_epoch:
            import csv
            with open(f"results/report.csv", "a") as f:
                wr = csv.writer(f, dialect='excel')
                results.append([self.args.exp_id,time_now,self.epoch,round(acc_l2r[0], 4), round(acc_l2r[1], 4), round(mrr_l2r, 4),
                                round(acc_r2l[0], 4), round(acc_r2l[1], 4), round(mrr_r2l, 4)])
                wr.writerows(results)
                self.logger.info(f"writing test results to csv file...")
                            
           
                         
    def _test_(self, test_left, test_right, last_epoch=False, save_name="", loss=None):
                
        import math
        from decimal import Decimal
        with torch.no_grad():
            embs_dic = self.model.joint_emb_generat() 
            embs_dic.pop('hidden_states', None)
            embs_dic.pop('weight_norm', None)
            # embs_dic.pop('name', None)
            # embs_dic.pop('char_name', None)
        if self.KGs['txt_features'] is not None:
            embs_dic.update({'txt':torch.tensor(self.KGs['txt_features'])})
        # self.cal_feature_sim()
        # embs_dic = {k:F.normalize(v) for k,v in embs_dic.items() if v is not None}
        # keys = ['image','attribute','relation','structure','name','txt']
        # for key in keys:
        #     if key in embs_dic.keys():
        #         self.gold_sim_anal(embs_dic[key],key,self.test_set)
        
        top_k = [1,10,50]
        # embs = [F.normalize(emb) for (modal,emb) in emb_dic.items() if modal != 'joint']
        # print(f"embs size:{len(embs)}")
        embs = F.normalize(embs_dic['joint'])
        dis = torch.matmul(embs[test_left], embs[test_right].transpose(0,1))
        if self.args.csls is True:
            for i in range(self.args.m_csls):
                dis =  csls_sim(dis, self.args.csls_k)
        aligned = []        
        distance = dis
        acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
        acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
        test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l, cross_acc = 0, 0., 0., 0., 0., 0.,0.                                                                                                                                                                                                                                                                    
        test_left_np = test_left.cpu().numpy()
        test_right_np = test_right.cpu().numpy()

        for idx in range(test_left.shape[0]):
            _,indices = torch.sort(distance[idx, :], descending=True)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()     
            mean_l2r += (rank + 1)
            mrr_l2r += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_l2r[i] += 1
            if rank == 0:
                aligned.append(test_left_np[idx])
                
        for idx in range(test_right.shape[0]):
            _, indices = torch.sort(distance[:, idx], descending=True)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_r2l += (rank + 1)
            mrr_r2l += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_r2l[i] += 1
        mean_l2r /= test_left.size(0)
        mean_r2l /= test_right.size(0)
        mrr_l2r /= test_left.size(0)
        mrr_r2l /= test_right.size(0)
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / test_left.size(0), 4)
            acc_r2l[i] = round(acc_r2l[i] / test_right.size(0), 4)
        gc.collect()
        if not self.args.only_test:
            Loss_out = f", Loss = {self.loss_item:.4f}"
        else:
            Loss_out = ""
            self.epoch = "Test"
            self.early_stop_count = 1
        if self.rank == 0:
            self.logger.info(f"Ep {self.epoch} | l2r: acc of top {top_k} = {acc_l2r}, mr = {mean_l2r:.3f}, mrr = {mrr_l2r:.3f}{Loss_out}")
            self.logger.info(f"Ep {self.epoch} | r2l: acc of top {top_k} = {acc_r2l}, mr = {mean_r2l:.3f}, mrr = {mrr_r2l:.3f}{Loss_out}")
            self.logger.info(f'cross_acc: {cross_acc:.3f}')
            self.early_stop_count -= 1
        save_file = f"iter_analy/{self.args.exp_id}.csv"
        with open(save_file, "w") as f:
            for a in aligned:
                f.write(str(a) + '\n')
               
   

        
if __name__ == '__main__':
    cfg = cfg()
    cfg.get_args()
    cfgs = cfg.update_train_configs()
    
    set_seed(cfgs.random_seed)
    # -----  Init ----------
    if cfgs.dist and not cfgs.only_test:
        init_distributed_mode(args=cfgs)
    else:
        torch.multiprocessing.set_sharing_strategy('file_system')
    rank = cfgs.rank
    # pprint.pprint(cfgs)

    writer, logger = None, None
    if rank == 0:
        logger = initialize_exp(cfgs)
        logger_path = get_dump_path(cfgs)
        cfgs.time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        comment = f'bath_size={cfgs.batch_size} exp_id={cfgs.exp_id}'
        if not cfgs.no_tensorboard and not cfgs.only_test:
            writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', cfgs.time_stamp), comment=comment)

    cfgs.device = torch.device(cfgs.device)

  
    runner = Runner(cfgs, writer, logger, rank)
    if cfgs.only_test:
        runner.test(last_epoch=False)
    else:
        runner.run()
    
    # -----  End ----------
    if not cfgs.no_tensorboard and not cfgs.only_test and rank == 0:
        writer.close()
        logger.info("done!")

    if cfgs.dist and not cfgs.only_test:
        dist.barrier()
        dist.destroy_process_group()
 