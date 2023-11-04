---  
layout: distill  
title: Generative Models - GANs v.s. VAEs v.s. Diffusion Models  
date: 2023-10-06  
description: comparision of GANs, VAEs and Diffusion Models  
tags:  
  - generative-models  
giscus_comments: "true"  
authors:  
  - name: Wei Deng  
    url: "https://wdaicv.site"  
    affiliations:  
      name: nil  
toc:  
  - name: GANs  
  - name: VAEs  
  - name: Diffusion Models  
---  
> From: https://pub.towardsai.net/diffusion-models-vs-gans-vs-vaes-comparison-of-deep-generative-models-67ab93e0d9ae  
---  
# GANs  
- GAN = Generator + Discriminator  
- Training loss: adversarial loss. The generator aims to "fool" a discriminator.  
- High-fidelity results. The discriminator cannot distinguish between the fake and real samples.  
- Low-diversity results (mode collapse): When the discriminator has over-trained or catastrophic forgetting happens, the generator might be happy enough to produce a small part of data diversity.  
- Hard to train. It can difficult to determine when your network converged.  
  
# VAEs  
- VAE = Encoder + Decoder  
- Training by maximizing log-likelihood.  
  
{% include figure.html path="assets/img/../assets/img/Pasted image 20231029144619.png" width="100%" %}