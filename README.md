# PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models

This repository is the official implementation of the following paper:

[PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models](https://arxiv.org/abs/2102.03161) \
Chaoyang He (USC), Shen Li (Facebook AI Research), Mahdi Soltanolkotabi (USC), Salman Avestimehr (USC) \
Accepted to ICML 2021 (International Conference on Machine Learning 2021) 

## 1. Introduction

<img src="https://chaoyanghe.com/wp-content/uploads/2021/06/PipeTransformer-ICML2021.png" alt="PipeTransformer"/>

The size of Transformer models is growing at an unprecedented rate. It has taken less than one year to reach trillion-level parameters since the release of GPT-3 (175B). 
Training such models requires both substantial engineering efforts and enormous computing resources, which are luxuries most research teams cannot afford. 
In this paper, we propose PipeTransformer, which leverages automated elastic pipelining for efficient distributed training of Transformer models. 
In PipeTransformer, we design an adaptive on the fly freeze algorithm that can identify and freeze some layers gradually during training, and an elastic pipelining system that can dynamically allocate resources to train the remaining active layers.
More specifically, PipeTransformer  automatically excludes frozen layers from the pipeline, packs active layers into fewer GPUs, and forks more replicas to increase data-parallel width. 
We evaluate PipeTransformer using Vision Transformer (ViT) on ImageNet and BERT on SQuAD and GLUE datasets. Our results show that compared to the state-of-the-art baseline, PipeTransformer attains up to $2.83$-fold speedup without losing accuracy.
We also provide various performance analyses for a more comprehensive understanding of our algorithmic and system-wise design. Finally, we have modularized our training system with flexible APIs and made the source code publicly available.

## 2. Overall Design
<img src="https://chaoyanghe.com/wp-content/uploads/2021/02/PipeTransformer-overall-design.png" alt="PipeTransformer"/>


## 3. Slides
[https://docs.google.com/presentation/d/1t6HWL33KIQo2as0nSHeBpXYtTBcy0nXCoLiKd0EashY/edit?usp=sharing](https://docs.google.com/presentation/d/1t6HWL33KIQo2as0nSHeBpXYtTBcy0nXCoLiKd0EashY/edit?usp=sharing)

## 4. Understanding PipeTransformer by Animation 
https://videos.files.wordpress.com/3vsRzoiw/pipetransformer-animation_m4v_hd.mp4
<figure class="video_container">
  <video controls="true" allowfullscreen="true">
    <source src="https://videos.files.wordpress.com/3vsRzoiw/pipetransformer-animation_m4v_hd.mp4" type="video/mp4">
  </video>
</figure>

![Animation](https://chaoyanghe.com/wp-content/uploads/2021/06/PipeTransformer-Animation.gif)


## 5. Installation
Please follow `INSTALL-CONDA.md`.

## 6. Experiments
check README.md at 

examples/image_classification

examples/question_answering

examples/text_classification

## 7. Citation

If you use any part of this code in your research or any engineering project, please cite our paper: 
```
@article{he2021pipetransformer,
  title={Pipetransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models},
  author={He, Chaoyang and Li, Shen and Soltanolkotabi, Mahdi and Avestimehr, Salman},
  journal={Thirty-eighth International Conference on Machine Learning},
  year={2021}
}
```


## 8. Contacts

Chaoyang He \
https://chaoyanghe.com \
chaoyang.he@usc.edu \
chaoyanghe.com@gmail.com
