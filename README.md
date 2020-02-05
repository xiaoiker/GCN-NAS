# GCN-NAS
PyTorch Source code for "[Learning Graph Convolutional Network for Skeleton-based Human Action Recognition by Neural Searching](https://arxiv.org/pdf/1911.04131.pdf)", AAAI2020

## Requirements
- python packages
  - pytorch = 0.4.1
  - torchvision>=0.2.1

  
## Data Preparation
 - Download the raw data from [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) and [Skeleton-Kinetics](https://github.com/yysijie/st-gcn). And pre-processes the data.
 
 - Preprocess the data with
  
    `python data_gen/ntu_gendata.py`
    
    `python data_gen/kinetics-gendata.py.`

 - Generate the bone data with: 
    
    `python data_gen/gen_bone_data.py`

## Model Training 
- Here, you can train the model searched by our method.
- Configure the config file for different settings. For example, training model under the corss-view protocal:

    `python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`
## Model Evaluation 
Change the config file depending on what you want.


    `python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`
```
./train.sh configs/config_regular_c109_n32.yaml
```

## Model searching 
- Devide the training data into trainging set and searching parts, as it is for a general NAS.
- run the gcn_search.py with corresponding configuration.
```
./train.sh configs/config_regular_c109_n32.yaml
```

## License
All materials in this repository are released under the  Apache License 2.0.

```
@article{peng2020learning,
  title={Learning Graph Convolutional Network for Skeleton-based Human Action Recognition by Neural Searching},
  author={Peng, Wei and Hong, Xiaopeng and Chen, Haoyu and Zhao, Guoying},
  journal={The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI},
  year={2020}
}
```
