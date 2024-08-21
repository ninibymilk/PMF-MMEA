import os
import os.path as osp
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from easydict import EasyDict as edict
from tqdm import tqdm
import pdb
import pprint
import json
import pickle
from collections import defaultdict

from config import cfg
from torchlight import initialize_exp, set_seed, get_dump_path
from src.data import load_data, Collator_base, EADataset
from src.utils import set_optim, Loss_log, pairwise_distances, csls_sim
from model import MEAformer

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
        # self.model_choise()
        set_seed(args.random_seed)

        if self.args.only_test:
            self.dataloader_init(test_set=self.test_set)
        else:
            self.dataloader_init(train_set=self.train_set, eval_set=self.eval_set, test_set=self.test_set)
            # if self.args.dist:
            #     self.model_sync()
            # else:
            #     self.model_list = [self.model]
            # if self.args.il:
            #     assert self.args.il_start < self.args.epoch
            #     train_epoch_1_stage = self.args.il_start
            # else:
            #     train_epoch_1_stage = self.args.epoch
            # self.optim_init(self.args, total_epoch=train_epoch_1_stage)
        print("__")

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
        assert self.args.model_name in ["EVA", "MCLEA", "MSNEA", "MEAformer"]
        if self.args.model_name == "MEAformer":
            self.model = MEAformer(self.KGs, self.args)

        self.model = self._load_model(self.model, model_name=self.args.model_name_save)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"total params num: {total_params}")

    def optim_init(self, opt, total_step=None, total_epoch=None, accumulation_step=None):
        step_per_epoch = len(self.train_dataloader)
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

        self.optimizer, self.scheduler = set_optim(opt, self.model_list, freeze_part, accumulation_step)

    def data_init(self):
        self.KGs, self.non_train, self.train_set, self.eval_set, self.test_set, self.test_ill_ = load_data(self.logger, self.args)
        self.train_ill = self.train_set.data
        self.eval_left = torch.LongTensor(self.eval_set[:, 0].squeeze()).cuda()
        self.eval_right = torch.LongTensor(self.eval_set[:, 1].squeeze()).cuda()
        if self.test_set is not None:
            self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).cuda()
            self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).cuda()

        self.eval_sampler = None
        if self.args.dist and not self.args.only_test:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
            self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.eval_set)
            if self.test_set is not None:
                self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_set)

    def dataloader_init(self, train_set=None, eval_set=None, test_set=None):
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
        else:
            self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
            if train_set is not None:
                self.train_dataloader = self._dataloader(train_set, bs, collator)
            if test_set is not None:
                self.test_dataloader = self._dataloader(test_set, bs, collator)
            if eval_set is not None:
                self.eval_dataloader = self._dataloader(eval_set, bs, collator)

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







    def _save_name_define(self):
        prefix = ""
        if self.args.dist:
            prefix = f"dist_{prefix}"
        if self.args.il:
            prefix = f"il{self.args.epoch-self.args.il_start}_b{self.args.il_start}_{prefix}"
        name = f'{self.args.exp_id}_{prefix}'
        return name


def plot_LSA(X, Y):
    from sklearn.decomposition import TruncatedSVD
    import matplotlib.pyplot as plt
    lsa = TruncatedSVD(n_components=2)
    X_lsa = lsa.fit(X).transform(X)
    Y_lsa = lsa.fit(X).transform(Y)
    X_lsa = X_lsa[:30]
    Y_lsa = Y_lsa[:30]

    # colors = ['red', 'green', 'blue', 'olive', 'pink', 'purple', 'orange', 'brown', 'gray', 'cyan']
    # ms = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', '+']
    # labels = uni
    #
    # # plt.rcParams['font.family'] = ['Arial Unicode MS'] #正常显示中文
    # for color, i, m in zip(colors, labels, ms):
    plt.scatter(X_lsa[:, 0], X_lsa[:, 1], c='red')
    plt.scatter(Y_lsa[:,0], Y_lsa[:,1],  c='green')
    for i in range(len(X_lsa)):
        plt.plot([X_lsa[i, 0], Y_lsa[i, 0]], [X_lsa[i, 1], Y_lsa[i, 1]], c='black')

    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('step: '+str(step))
    plt.show()

    plt.close()

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

    # print("print c to continue...")
    # -----  Begin ----------
    torch.cuda.set_device(cfgs.gpu)
    runner = Runner(cfgs, writer, logger, rank)
    adj = runner.KGs['adj']
    imgs = runner.KGs['images_list']
    tra_ill = runner.train_ill
    test_ill = runner.test_ill_

    ans = []
    ans2 = []
    neg_ans = []
    neg_ans2 = []
    for i,j in tra_ill:
        img_i = imgs[i]
        img_j = imgs[j]
        neg = np.random.randint(0, 14951)
        s = img_i.dot(img_j)
        neg_img = imgs[neg]
        s2 = img_j.dot(neg_img)

        ans.append(s)
        neg_ans.append(s2)
        s /= (np.linalg.norm(img_i) * np.linalg.norm(img_j))
        ans2.append(s)
        s2 /= (np.linalg.norm(img_j) * np.linalg.norm(neg_img))
        neg_ans2.append(s2)

    for i,j in test_ill:
        img_i = imgs[i]
        img_j = imgs[j]
        neg = np.random.randint(0, 14951)
        s = img_i.dot(img_j)
        neg_img = imgs[neg]
        s2 = img_j.dot(neg_img)

        ans.append(s)
        neg_ans.append(s2)
        s /= (np.linalg.norm(img_i) * np.linalg.norm(img_j))
        ans2.append(s)
        s2 /= (np.linalg.norm(img_j) * np.linalg.norm(neg_img))
        neg_ans2.append(s2)
    # print(sum(ans)/len(ans))
    # print(ans)
    print(sum(ans2)/len(ans2))
    print(sum(neg_ans2)/len(neg_ans2))
    import matplotlib.pyplot as plt
    ans2.sort()
    neg_ans2.sort()
    ans2 = np.array(ans2)
    neg_ans2 = np.array(neg_ans2)
    plt.plot(ans2, label='pos')
    # plt.plot(neg_ans2, label='neg')
    plt.legend()
    plt.show()
    plt.close()
    plot_LSA(imgs[tra_ill[:, 0]], imgs[tra_ill[:, 1]])



    # print(ans2)




