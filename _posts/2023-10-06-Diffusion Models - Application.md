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
  - name: CDM  
  - name: SDEdit  
  - name: ILVR  
  - name: DiffusionCLIP  
  - name: Prompt-to-Prompt  
  - name: Imagic  
  - name: Pix2Pix-Zero  
  - name: Null-text Inversion  
  - name: Direct Inversion  
---  
这篇文章是扩散模型的应用论文记录，主要有以下类别：  
- Low-level任务，如：超分、去噪等  
- 图像编辑  
- DDIM反演  
# SR3  
SR3 <d-cite key="sr3"></d-cite> 是第一个使用DDPM做图像超分的工作，对DDPM主要有以下三个改动：  
- 将图像超分任务转化为扩散模型中的控制生成任务，将LR上采样后作为condition与带噪HR图像concatenate输入到UNet  
- 不再使用$$\bar{\alpha}_t$$，而是从均匀分布$$\mathcal{U}(\bar{\alpha}_{t-1}, \bar{\alpha}_{t})$$中采样  
- UNet的输入不再是$$t$$，而是直接输入噪声强度。如果不做这样的改动，那么在采样时输入的$$t$$必须和训练时保持一致，即采样时必须按照$$T\to T-1\to \cdots \to 2 \to 1$$的方式采样。在改动后，相比输入时间步$$t$$，输入噪声强度更加鲁棒，可以随意改变采样步数。  
# CDM  
超分，级联的方式对小图进行超分，采用的方法就是SR3  
{% include figure.html path="assets/img/Pasted image 20230927200225.png" width="100%" %}  
# SDEdit  
{% include figure.html path="assets/img/Pasted image 20230927200246.png" width="100%" %}  
由于加噪过程是首先破坏高频信息，然后才破坏低频信息，所以加噪到一定程度之后，就可以去掉不想要的细节纹理，但仍保留大体结构。然后在中途开始去噪，于是生成出来的图像就既能遵循输入的引导，又显得真实。但是需要 realism-faithfulness trade-off  
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
基于扩散模型的图像编辑，使用到的技术有DDIM Inversion，CLIP微调扩散模型。  
{% include figure.html path="assets/img/Pasted image 20231010141023.png" width="100%" %}  
- 首先使用DDIM反演得到图像的latent code  
- 使用损失函数微调模型：  
{% include figure.html path="assets/img/Pasted image 20231011135020.png" width="100%" %}  
{% include figure.html path="assets/img/Pasted image 20231010141749.png" width="100%" %}  
{% include figure.html path="assets/img/Pasted image 20231010141758.png" width="100%" %}  
- direction损失中，ΔI提供了一个从参考文本到目标文本的方向，让参考图像和目标图像之间的距离对齐  
- 损失中还有可以加入id loss，l1 loss等  
# Prompt-to-Prompt  
无分类器引导的扩散模型主要通过交叉注意力将文本信息和图像进行融合，作者观察扩散模型中的cross-attention map，发现attention map和文本有着对应关系，并且整幅图的结构在去噪早期就已经确定（低频信息）  
{% include figure.html path="assets/img/Pasted image 20231010164516.png" width="100%" %}  
所以作者提出通过替换attention map的方式实现图像编辑  
{% include figure.html path="assets/img/Pasted image 20231010164614.png" width="100%" %}  
# Imagic  
提供一张参考图片和目标文本，将参考图片朝着目标文本的语义方向编辑  
{% include figure.html path="assets/img/Pasted image 20231010142606.png" width="100%" %}  
基于优化的方法，每一次编辑都需要重新优化  
{% include figure.html path="assets/img/Pasted image 20231010142704.png" width="100%" %}  
- 首先得到目标文本的编码$$e_{tgt}$$作为初始值，将这个编码设定为可学习参数  
- 类似SDS，使用参数冻结的预训练扩散模型优化文本编码到$$e_{opt}$$。不过SDS是将图像作为输入，而这里是将embedding作为输入（condition）  
- 固定文本编码，微调扩散模型（目的是使得生成的图片更具有多样性）  
- 使用$$e_{tgt}$$和$$e_{opt}$$的插值作为新的文本编码输入到微调后的扩散模型中，生成的图片具有文本语义的同时，也和参考图像的内容保持一致（需要调整插值参数，trade-off）  
# Pix2Pix-Zero  
基于扩散模型的图像编辑  
{% include figure.html path="assets/img/Pasted image 20231010143456.png" width="100%" %}  
使用的方法是正则化DDIM反演，feature map监督  
{% include figure.html path="assets/img/Pasted image 20231010143834.png" width="100%" %}  
- 首先反演得到隐变量  
- 不需要重新训练模型，只需要在推理的时候类似classifier guidance的方法对逆向后验分布的均值进行修改，对均值的修改是目标feature map和参考feature map的差对于$$x_t$$的梯度，期望是让参考feature map和目标feature map的差距减小  
{% include figure.html path="assets/img/Pasted image 20231010155302.png" width="100%" %}  
# Null-text Inversion  
在基于扩散模型的编辑中，几乎都有一个重要的步骤：DDIM反演，即将图片逆映射到噪声（隐变量）。但是传统的方法得到的隐变量在使用扩散模型进行去噪，最终得到的图片往往和输入图片有偏差。如果单纯地使用反演后的隐变量进行图像编辑，效果会有所限制。  
作者观察到两个现象：  
- DDIM Inversion每一步都会产生误差，对于无条件扩散模型，累积误差可以忽略。当应用classifier-free guidance扩散模型时，DDIM Inversion不能准确的重建原图像。  
- 优化classifier-free guidance扩散模型的Null-text Embedding能准确重建原图像，同时避免了微调模型和文本嵌入，从而完整保留了模型的先验信息和语义信息  
{% include figure.html path="assets/img/Pasted image 20231010165723.png" width="100%" %}  
- 当CFG值为1（无Null-text）时，DDIM inversion产生的轨迹为T1，最终得到隐变量$$z_T^*$$，重建效果较好，称之为pivot  
- 在CFG值>1时，从$$z_T^*$$开始进行采样，产生的轨迹为T2，会逐渐偏离T1  
- 本文提出在所有采样步骤，让T2尽可能接近T1，从而保留原图像的语义和视觉信息  
- 优化对象是null-text embedding，并且每一采样步骤的null-text embedding都不一样，优化目标是：  
{% include figure.html path="assets/img/Pasted image 20231010170257.png" width="100%" %}  
{% include figure.html path="assets/img/Pasted image 20231010170306.png" width="100%" %}  
相当于是使用$$T$$个变量储存了每一步T2到T1的偏差。最终模型输出隐变量和Null-text embedding，可供其它编辑方法使用，增强编辑效果  
# Direct Inversion  
由于传统DDIM Inversion具有偏差，所以通常是使用基于优化的方法修正偏离。  
Null-text Inversion方法通过优化Null-text embedding来修正偏离，但是有几个问题：  
- 单是Null-text embedding很难完美地储存偏差，所以偏差只是被减小而没有被消除  
- 优化后的Null-text embedding在扩散模型训练的时候并没有出现过，可能会影响生成效果  
所以本文直接记录偏离路径到原始路径之间的差值  
TODO