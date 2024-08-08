
# PMF: Progressively Modality Freezing for Multi-Modal Entity Alignment
**The Offical Code for Our Paper：**[Progressively Modality Freezing for Multi-Modal Entity Alignment](https://arxiv.org/abs/2407.16168)**, ACL2024**.

## 📰Model Overview

We presented the Progressive Modality Freezing (PMF) model to advance Multi-Modal Entity Alignment. 

By measuring and evaluating the relevance of various modalities, PMF progressively freezes features deemed less critical, thereby facilitating the integration and consistency of multi-modal features. Furthermore, we introduced a unified training objective tailored to foster a harmonious contrast between KGs and modalities. 

Empirical evaluations on 9 sub-datasets affirm the superiority of PMF.

![Model](images/model.png)

## 🛠️Install

```bash
>> cd PMF-MMEA
>> pip install -r requirement.txt
```

### details

```
Python (>=3.7 )
Pytorch (>= 1.7.0)
numpy (>= 1.19.2)
easydict (>= 1.10)
unidecode (>= 1.3.7)
tensorboard (>= 2.11.2)
```

## 📂Dataset

- We assessed the effectiveness of our proposed method using three publicly available MMEA datasets: **DBP15K, MMKG, Multi-OpenEA**
- Download from [GoogleDrive](https://drive.google.com/file/d/1VIWcc3KDcLcRImeSrF2AyhetBLq_gsnx/view?usp=sharing) (1.26G) and unzip it to make those files **satisfy the following file hierarchy**:

```bash
ROOTs
├── data
│   └── datasets
└── PMF-MMEA
```

## ⛷️Train

### quick start

```bash

# DBP15K
>> bash run_dbp.sh 
# MMKG
>> bash run_mmkg.sh
# Multi-OpenEA
>> bash run_openea.sh
```

## 🥇Results

Model performance report can be found in the file `PMF-MMEA/results/reports.csv`

![results](images/results.png)

## 📝Cite

```
@article{huang2024progressively,
  title={Progressively Modality Freezing for Multi-Modal Entity Alignment},
  author={Huang, Yani and Zhang, Xuefeng and Zhang, Richong and Chen, Junfan and Kim, Jaein},
  journal={arXiv preprint arXiv:2407.16168},
  year={2024}
}
```

## 🫶**Acknowledgement**
This work was supported by CCSE, School of Computer Science and Engineering, Beihang University, Beijing, China.
Our codes are modified based on [MEAformer](https://github.com/zjukg/MEAformer), and we also appreciate [MCLEA](https://github.com/lzxlin/MCLEA), [MSNEA](https://github.com/liyichen-cly/MSNEA), [EVA](https://github.com/cambridgeltl/eva), [MMEA](https://github.com/liyichen-cly/MMEA) and many other related works for their open-source contributions.