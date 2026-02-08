These are the guidelines to run our proposed model, named MVCIB.

## Reproducibility

### Datasets 

We conducted experiments across four different chemical domains: Physiology (BBBP, Tox21, ToxCast,SIDER, ClinTox, and MUV), Physical Chemistry (ESOL, FreeSolv, and Lipo), Biophysics (mol-HIV and BACE), Quantum Mechanics (QM9).
For the pre-training dataset, we considered unlabeled molecules from the ChEMBL database.


### Requirements and Environment Setup

The source code was developed in Python 3.8.8. MVCIB is built using Torch-geometric 2.3.1 and DGL 1.1.0. Please refer to the official websites for installation and setup.
All the requirements are included in the ```environment.yml``` file.

```
# Conda installation

# Install python environment

conda env create -f environment.yml 
```
The source code contains both Self-Supervised Pre-training and Fine-tuning phases. 
We also provide our pre-trained model, named pre_trained_GIN_300_4_3.pt, in the folder outputs/. 
The pre-processed data for each dataset is stored in pts/ folder, which contains inputs prepared for direct use in the model.

### Self-supervised pre-training

#### pre-training
```
# Use the following command to run the pretrain task, the output will generate the pre-trained files in the folder outputs/.
python exp_pretraining.py --encoder GIN --k_transition 2 --device cuda:0
```

### Hyperparameters

The following options can be passed to the below commands for fine-tuning the model:

```--encoder:``` The graph encoder. For example: ```--encoder GIN```

```--lr:``` Learning rate for fine-tuning the model. For example: ```--lr 0.001```

```--dims:``` The dimension of hidden vectors. For example: ```--dims 64```

```--num_layers:``` Number of layers for model training. For example: ```--num_layers 5``` 

```--k_transition:``` The size of 2D molecular subgraphs. For example: ```--k_transition 3```

```--angstrom:``` The size of 3D molecular subgraphs. For example: ```--angstrom 1.5```

```--pretrained_ds:``` The file name of the pre-trained model. For example: ```--pretrained_ds pre_trained```

```--ft_epoches:```Number of epochs for fine-tuning the pre-trained model. For example: ```--ft_epoches 50```.

```--batch_size:``` The size of a batch. For example: ```--batch_size 128```.

```--device:``` The GPU id. For example: ```--device 0```.

### How to fine-tune MVCIB on downstream datasets

The following commands will run the fine-tuning the **MVCIB** on different datasets.
The model performance will be sent to the command console.

#### For BBBP and BACE Datasets
```
python exp_moleculenetBACE_BBBP.py  
``` 
#### For FreeSolv,  ESOL, and Lipo Datasets
```
python exp_molsolv.py  --dataset ESOL 
``` 
#### For MUV, SIDER, Tox21, ClinTox, and ToxCast Datasets
```
python exp_moleculeSTCT.py  --dataset Tox21  
``` 
#### For ogbg-molhiv Dataset
```
python exp_molhiv.py   
``` 

#### For the QM9 Dataset with different properties (Table 2)
```
python exp_molqm9.py  --target_index 0 

# where the target_index presents which target property in the QM9 dataset to predict, indexed from 0 to 6 as shown in Table 2.
``` 
