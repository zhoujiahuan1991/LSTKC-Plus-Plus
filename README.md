# [IEEE TPAMI 2025] Long Short-Term Knowledge Decomposition and Consolidation for Lifelong Person Re-Identification

<div align="center">

<div>
      Kunlun Xu<sup>1</sup>&emsp; Zichen Liu<sup>1</sup>&emsp; Xu Zou<sup>2</sup>&emsp; Yuxin Peng<sup>1</sup>&emsp; Jiahuan Zhou<sup>1*</sup>
  </div>
<div>

  <sup>1</sup>Wangxuan Institute of Computer Technology, Peking University&emsp; <sup>2</sup>School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

</div>
</div>
<p align="center">
  <a href="https://github.com/zhoujiahuan1991/LSTKC-Plus-Plus"><img src="https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fzhoujiahuan1991%2FLSTKC-Plus-Plus&label=&icon=github&color=%233d8bfd"></a>
</p>

The *official* repository for  [Long Short-Term Knowledge Decomposition and Consolidation for Lifelong Person Re-Identification](https://ieeexplore.ieee.org/abstract/document/11010188).

## News
* 🔥[2024.02.05] The code for LSTKC (accepted by AAAI 2024) is released in [LSTKC Code](https://github.com/zhoujiahuan1991/AAAI2024-LSTKC)!
* 🔥[2024.03.24] The full paper for LSTKC is publicly available in [LSTKC Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29554)!
* 🔥[2025.05.19] Our improved verison LSTKC++ is accepted by IEEE TPAMI. The full paper is available in [LSTKC++ Paper](https://ieeexplore.ieee.org/abstract/document/11010188/)!
* 🔥[2025.06.12] The code for LSTKC++ is released in [LSTKC++ Code](https://github.com/zhoujiahuan1991/LSTKC-Plus-Plus).

![Framework](figs/framework.png)


## Installation
```shell
conda create -n IRL python=3.9
conda activate IRL
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
python setup.py develop
```
## Prepare Datasets
Download the person re-identification datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](http://www.pkuvmc.com/dataset.html), [CUHK03](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03), [SenseReID](https://drive.google.com/file/d/0B56OfSrVI8hubVJLTzkwV2VaOWM/view?resourcekey=0-PKtdd5m_Jatmi2n9Kb_gFQ). Other datasets can be prepared following [Torchreid_Datasets_Doc](https://kaiyangzhou.github.io/deep-person-reid/datasets.html) and [light-reid](https://github.com/wangguanan/light-reid).
Then unzip them and rename them under the directory like
```
PRID
├── CUHK01
│   └──..
├── CUHK02
│   └──..
├── CUHK03
│   └──..
├── CUHK-SYSU
│   └──..
├── DukeMTMC-reID
│   └──..
├── grid
│   └──..
├── i-LIDS_Pedestrain
│   └──..
├── MSMT17_V2
│   └──..
├── Market-1501
│   └──..
├── prid2011
│   └──..
├── SenseReID
│   └──..
└── viper
    └──..
```

## Quick Start
Reproduce the reported results
```shell
CUDA_VISIBLE_DEVICES=7 python continual_train.py -b 64 --num-instances 8 --data-dir path/to/PRID
(for example, CUDA_VISIBLE_DEVICES=7 python continual_train.py -b 64 --num-instances 8 --data-dir ../DATA/PRID)
```
or run the bash file
```shell
sh work_reproduce.sh
```



## Results
The following results were obtained with a single NVIDIA 4090 GPU:

![Results](figs/result.png)

## Citation
If you find this code useful for your research, please cite our paper.

@article{xu2025long,
  title={Long Short-Term Knowledge Decomposition and Consolidation for Lifelong Person Re-Identification},
  author={Xu, Kunlun and Liu, Zichen and Zou, Xu and Peng, Yuxin and Zhou, Jiahuan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}

### We have conducted a series of research in Lifelong Person Re-Identification as follows.

#### Imgae-level Distribution Modeling and Transfer:
@inproceedings{xu2025dask,
  title={Dask: Distribution rehearsing via adaptive style kernel learning for exemplar-free lifelong person re-identification},
  author={Xu, Kunlun and Jiang, Chenghao and Xiong, Peixi and Peng, Yuxin and Zhou, Jiahuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={9},
  pages={8915--8923},
  year={2025}
}

#### Feature-level Distribution Modeling and Prototyping:
@inproceedings{xu2024distribution,
  title={Distribution-aware Knowledge Prototyping for Non-exemplar Lifelong Person Re-identification},
  author={Xu, Kunlun and Zou, Xu and Peng, Yuxin and Zhou, Jiahuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16604--16613},
  year={2024}
}

#### Long Short-Term Knowledge Rectification and Consolidation:
@inproceedings{xu2024lstkc,
  title={LSTKC: Long Short-Term Knowledge Consolidation for Lifelong Person Re-identification},
  author={Xu, Kunlun and Zou, Xu and Zhou, Jiahuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={14},
  pages={16202--16210},
  year={2024}
}

#### Lifelong Learning with Label Noise: 
@inproceedings{xu2024mitigate,
  title={Mitigate Catastrophic Remembering via Continual Knowledge Purification for Noisy Lifelong Person Re-Identification},
  author={Xu, Kunlun and Zhang, Haozhuo and Li, Yu and Peng, Yuxin and Zhou, Jiahuan},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={5790--5799},
  year={2024}
}

#### Prompt-guided Adaptive Knowledge Consolidation:
@article{li2024exemplar,
  title={Exemplar-Free Lifelong Person Re-identification via Prompt-Guided Adaptive Knowledge Consolidation},
  author={Li, Qiwei and Xu, Kunlun and Peng, Yuxin and Zhou, Jiahuan},
  journal={International Journal of Computer Vision},
  pages={1--16},
  year={2024},
  publisher={Springer}
}

#### Compatible Lifelong Learning:
@inproceedings{cui2024learning,
  title={Learning Continual Compatible Representation for Re-indexing Free Lifelong Person Re-identification},
  author={Cui, Zhenyu and Zhou, Jiahuan and Wang, Xun and Zhu, Manyu and Peng, Yuxin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16614--16623},
  year={2024}
}

## Acknowledgement
Our code is based on the PyTorch implementation of [PatchKD](https://github.com/feifeiobama/PatchKD) and [PTKP](https://github.com/g3956/PTKP).

## Contact

For any questions, feel free to contact us (xkl@stu.pku.edu.cn).

Welcome to our Laboratory Homepage ([OV<sup>3</sup> Lab](https://zhoujiahuan1991.github.io/)) for more information about our papers, source codes, and datasets.

