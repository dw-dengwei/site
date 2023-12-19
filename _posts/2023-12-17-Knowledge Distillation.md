---  
layout: distill  
title: Knowledge Distillation  
date: 2023-10-04 15:09:00  
description: some KD-related notes  
tags:  
  - diffusion-model  
giscus_comments: "true"  
authors:  
  - name: Wei Deng  
    url: "https://wdaicv.site"  
    affiliations:  
      name: nil  
toc:  
  - name: temp  
---  
# 21_ICLR_Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective  
知识蒸馏被视为一种正则化方式。从统计学的角度看，正则化旨在降低模型的方差。  
在模型训练的过程中，拟合数据和正则是一对矛盾体。拟合数据需要模型的bias尽量减小，靠近数据的均值，正则需要减小模型的方差防止过拟合。  
使用软标签进行训练会导致较大的bias和较小的方差。