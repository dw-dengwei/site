---  
layout: distill  
title: Diffusion Models - 3D Generation  
date: 2023-10-06  
description: 3D generation diffusion model papers  
tags:  
  - diffusion-model  
  - 3d-generation  
giscus_comments: "true"  
authors:  
  - name: Wei Deng  
    url: "https://wdaicv.site"  
    affiliations:  
      name: nil  
toc:  
  - name: 22-10-06_Watson_Novel View Synthesis with Diffusion Models  
  - name: 22-12-06_Deng_NeRDi_Single-View NeRF Synthesis with Language-Guided Diffusion as General Image Priors  
  - name: 23-02-15_Zhou_SparseFusion_Distilling View-conditioned Diffusion for 3D Reconstruction  
  - name: 23-02-20_Gu_NerfDiff_Single-image View Synthesis with NeRF-guided Distillation from 3D-aware Diffusion  
  - name: 23-03-17_Lei_RGBD2_Generative Scene Synthesis via Incremental View Inpainting using RGBD Diffusion Models  
  - name: 23-03-20_Liu_Zero-1-to-3_Zero-shot One Image to 3D Object  
  - name: 23-03-30_Tseng_Consistent View Synthesis with Pose-Guided Diffusion Models  
  - name: 23-03-31_Xiang_3D-aware Image Generation using 2D Diffusion Models  
  - name: 23-04-05_Chan_Generative Novel View Synthesis with 3D-Aware Diffusion Models  
---  
# 22-10-06_Watson_Novel View Synthesis with Diffusion Models  
Google Research作品  
- 将输入图片旋转到指定视角  
{% include figure.html path="assets/img/Pasted image 20231029160243.png" width="100%" %}  
- pose-conditional image-to-image diffusion model  
{% include figure.html path="assets/img/Pasted image 20231029160258.png" width="100%" %}  
- 使用stochastic conditioning鼓励扩散模型生成3D一致的样本  
{% include figure.html path="assets/img/Pasted image 20231029164722.png" width="100%" %}  
自回归地生成新视角，conditioning set表示已经有的视角，最开始只有输入图片，随着不断地生成新视角图片，conditioning set逐渐扩增。其中，采样过程中，每一次denoising step都会更换image condition（从conditioning set中随机抽取）  
- 针对该任务对UNet结构重新设计  
使用concat的方法将conditioning image融合进模型，效果不好，作者的猜测：只使用self-attention难以学到视角的变换。对于结构的改进见论文，也许后续的文章不再使用这个结构。  
- 提出geometry-free新视角生成模型的metric：3D consistency scoring  
现有的评价方法无法正确地对三位一致性进行评判，于是提出3D consistency scoring，满足如下要求：  
1. 对不满足3D一致性的输出进行惩罚  
2. **不**对满足3D一致性但是和GT有出入的输出进行惩罚  
3. 对不符合输入图片的输出进行惩罚  
使用NeRF重建多个新视角图片，具体见原论文  
总结：使用pose-guided diffusion model生成新视角，在采样的时候对之前生成的图片都进行考虑，以达到3D一致性  
  
# 22-12-06_Deng_NeRDi: Single-View NeRF Synthesis with Language-Guided Diffusion as General Image Priors  
使用SDS优化NeRF有一个问题，预训练模型提供给NeRF的先验不够specific，所以生成出来的三维模型比较平滑，趋向与生成一个比较“平均”的模型  
{% include figure.html path="assets/img/Pasted image 20231030002657.png" width="100%" %}  
所以本文将预训练扩散模型提供的先验分布“缩小”。使用到的扩散模型是text-to-image扩散模型，输入的condition是caption以及使用textual inversion获得的与图片对齐的文本编码。  
NeRF除了使用SDS进行优化以外，还使用GT单视角图片进行监督。相当于NeRF看到的是输入的单视角图片+先验分布被缩小的扩散模型  
  
# 23-02-15_Zhou_SparseFusion: Distilling View-conditioned Diffusion for 3D Reconstruction  
少视角NeRF重建  
相对于DreamFusion，主要的创新是训练了一个新的扩散模型VLDM  
{% include figure.html path="assets/img/Pasted image 20231029205551.png" width="100%" %}  
输入少量视角的图片、对应的相机参数以及目标相机参数，使用EFT将所有condition编码起来  
  
# 23-02-20_Gu_NerfDiff: Single-image View Synthesis with NeRF-guided Distillation from 3D-aware Diffusion  
依然是训练一个新的view+image conditioned 扩散模型  
{% include figure.html path="assets/img/Pasted image 20231029211217.png" width="100%" %}  
使用UNet对图像进行翻译，转化为三平面，然后在目标视角下采样作为CDM的condition image。两个UNet同时训练：去噪loss+重建loss  
扩散模型中用到的Image-conditioned NeRF可以输入图片和相机视角，输出对应视角的图片，但是目前输出的新视角质量较差，于是使用训练完成的扩散模型对这个NeRF进行微调。  
有点左脚踩右脚升天的感觉，但是论文效果不错  
  
# 23-03-17_Lei_RGBD2: Generative Scene Synthesis via Incremental View Inpainting using RGBD Diffusion Models  
{% include figure.html path="assets/img/Pasted image 20231029214114.png" width="100%" %}  
输入RGB-D序列（只有少量视角），输出对整个场景重建的mesh  
核心思想是利用扩散模型对场景中的孔洞进行补全  
  
# 23-03-20_Liu_Zero-1-to-3: Zero-shot One Image to 3D Object  
输入图片和目标相机视角，输出符合图片和相机视角的图片，然后可以用这个预训练模型结合SJC生成3D模型  
这篇文章相比之前的文章来说，贡献之一是证明了大规模预训练扩散模型有潜力理解图像的三维结构，那么使用in-the-wild 图片也能很好地进行三维重建  
{% include figure.html path="assets/img/Pasted image 20231029151838.png" width="100%" %}  
  
# 23-03-30_Tseng_Consistent View Synthesis with Pose-Guided Diffusion Models  
同样是训练一个pose-guided diffusion model，输入图片和目标相机参数，输出对应图片。  
本文的创新在于提出了一种让UNet理解相机参数的机制，具体来说，提出了一种名为极线注意力机制，利用相机内外参得到的极线约束，对Cross attention产生的注意力分数矩阵进行修改，从而提供让网络能够理解相机视角  
文中还交代了一些提高三维一致性的trick  
{% include figure.html path="assets/img/Pasted image 20231030011953.png" width="100%" %}  
# 23-03-31_Xiang_3D-aware Image Generation using 2D Diffusion Models  
和RGBD2 (Lei et. al.) 的方法类似，使用扩散模型预测RGBD，然后手动旋转，将产生的孔洞用扩散模型补全，循环  
图形学设计较多  
{% include figure.html path="assets/img/Pasted image 20231030012551.png" width="100%" %}  
  
# 23-04-05_Chan_Generative Novel View Synthesis with 3D-Aware Diffusion Models  
{% include figure.html path="assets/img/Pasted image 20231118183023.png" width="100%" %}  
和其它的新视角生成模型相比，不同的就是condition的方式  
zero123是将相机参数融合到图像的CLIP embedding里  
这个方法是引入了nerf，首先将一个或多个视角的图片编码得到nerf表示，然后进行体渲染得到目标视角的feature作为condition  
为什么要引入nerf？TODO  
# 23-06-16_DreamSparse: Escaping from Plato's Cave with 2D Frozen Diffusion Model Given Sparse Views  
{% include figure.html path="assets/img/Pasted image 20231118193101.png" width="100%" %}  
又提供了一种将目标视角作为condition的方法，目前看不懂  
# 23-06-29_One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization  
{% include figure.html path="assets/img/Pasted image 20231118195042.png" width="100%" %}  
直接使用Zero123生成的多视角图片进行NeRF重建效果不好，是因为Zero123生成的图片具有多视角不一致性  
本方法使用Zero123生成的有缺陷的图片作为SparseNeuS的输入进行重建  
这个方法没有重新训练或微调一个扩散模型，而是利用了现有的Zero123进行多视角图片的生成3D模型，并且不是基于SDS那样的优化方法，生成速度更快  
