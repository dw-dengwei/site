---  
layout: distill  
title: Generative Models - Editing  
date: 2023-12-14  
description: Image editing using DPMs  
tags:  
  - generative-models  
giscus_comments: "true"  
authors:  
  - name: Wei Deng  
    url: "https://wdaicv.site"  
    affiliations:  
      name: nil  
toc:  
  - name: DPMs  
---  
# 2022-01-04_SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations  
{% include figure.html path="assets/img/Pasted image 20231214151746.png" width="100%" %}  
使用Stroke来引导生成，主要原理是扩散模型加噪过程是先破坏高频细节，最后破坏低频信息  
于是在Stroke上加噪，保留低频的信息（大致轮廓、颜色），然后切换到逆向去噪过程生成高频细节。  
