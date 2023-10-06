---  
layout: distill  
title: Diffusion Models - Application  
date: 2023-10-04 15:09:00  
description: introduction about the applications of diffusion models  
tags:  
  - diffusion-model  
giscus_comments: "true"  
authors:  
  - name: Wei Deng  
    url: "https://wdaicv.site"  
    affiliations:  
      name: nil  
toc:  
  - name: SR3  
  - name: SR3  
  - name: CDM  
  - name: SDEdit  
  - name: ILVR  
  - name: DiffusionCLIP  
  - name: Reference  
---  
# SR3  
超分，训练数据是LR和SR配对的图片，以LR图片作为condition，生成SR图片  
# CDM  
超分，级联的方式对小图进行超分，采用的方法就是SR3  
{% include figure.html path="assets/img/Pasted image 20230927200225.png" width="100%" %}  
# SDEdit  
{% include figure.html path="assets/img/Pasted image 20230927200246.png" width="100%" %}  
由于加噪过程是首先破坏高频信息，然后才破坏低频信息，所以加噪到一定程度之后，就就可以去掉不想要的细节纹理，但仍保留大体结构，于是生成出来的图像就既能遵循输入的引导，又显得真实。但是需要 realism-faithfulness trade-off  
# ILVR  
给定一个参考图像$$y$$，通过调整DDPM去噪过程，希望让模型生成的图像接近参考图像，作者定义的接近是让模型能够满足  
$$  
\phi_N(x_t)=\phi_N(y_t)  
$$  
$$\phi_N(\cdot)$$是一个低通滤波器（下采样之后再插值回来）。使用如下的算法：  
{% include figure.html path="assets/img/Pasted image 20230927201110.png" width="100%" %}  
即，对DDPM预测的$$x'_{t-1}$$加上bias：$$\phi_N(y_{t-1})-\phi_N(x'_{t-1})$$，可以证明，如果上/下采样采用的是最近邻插值，使用这种方法可以使得$$\phi_N(x_t)=\phi_N(y_t)$$.  
这种方法和classifier guidance很相似，甚至不需要训练一个外部模型，对算力友好。  
# DiffusionCLIP  
基于扩散模型的图像编辑，使用到的技术有DDIM Inversion，CLIP  
# Reference