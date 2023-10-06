---  
layout: distill  
title: Diffusion Models - Score-based Generative Models  
date: 2023-10-06  
description: introduction about score-based generative models  
tags:  
  - diffusion-model  
giscus_comments: "true"  
authors:  
  - name: Wei Deng  
    url: "https://wdaicv.site"  
    affiliations:  
      name: nil  
toc:  
  - name: Introduction  
  - name: Preliminary  
	- name: How Score-based Generative Models Work  
  - name: Reference  
---  
# Introduction  
score-based生成模型是一种新的生成模型范式，在score-based之前，主要存在两种类型的生成模型：  
1. **基于似然的生成模型**：基于最大似然估计（MLE），使得从真实数据分布中采样出的数据能够在所建模的数据分布中概率最大化，即$$\max_\theta \mathbb{E}_{x\sim p_\mathrm{data}(x)}\left[\log p(x;\theta)\right]$$。这类模型通过最大化似然函数，直接学习数据集的分布，主要方法有VAE、流模型、能量模型  
2. **隐式生成模型**：非直接学习数据集的分布，主要方法有GAN  
它们分别具有以下限制：  
1. **对于基于似然的生成模型**：对神经网络的结构要求高  
2. **对于隐式生成模型**：训练不稳定、容易导致模式坍塌  
# Preliminary  
score-based生成模型就能很好地避免这些限制，在介绍score-based生成模型之前，需要明确几个概念：  
1. **能量函数**  
对于大多数分布，可以使用概率密度函数（PDF）进行描述，也可以使用能量函数进行描述：  
$$  
p(x)=\frac{e^{-E(x)}}{Z}  
$$  
其中$$p(x)$$是PDF，$$E(x)$$是能量函数，$$Z=\int e^{-E(x)}\mathrm{d}x$$是归一化因子。以高斯分布为例，可以使用以下能量函数进行描述：  
$$  
\begin{align}  
E(x;\mu,\sigma^2)&=\frac{1}{2\sigma^2}(x-\mu)^2\\  
p(x)&=\frac{e^{-E(x)}}{\int e^{-E(x)}\mathrm dx}=\frac{e^{\frac{1}{2\sigma^2}(x-\mu)^2}}{\sqrt{2\pi\sigma^2}}  
\end{align}  
$$  
2. **能量模型**  
能量模型是基于能量函数的生成模型：  
$$  
p_\theta(x)=\frac{\exp(-E_\theta(x))}{Z_\theta}  
$$  
基于能量函数对数据分布进行建模的时候，**如何计算能量模型的归一化函数$$Z_\theta$$是一个较难的问题**。传统的基于似然的生成模型（如自回归模型、流模型、VAE）都有自己的解决方式，但是都对能量模型做了太多的约束，各自都有其限制。  
3. **蒙特卡罗采样方法：拒绝-接受法**  
目的：希望从一个复杂分布$$p(x)$$采样$$N$$个样本  
方法：使用一个简单分布$$q(x)$$为媒介（例如：高斯分布），这个分布必须满足它的$$c>0$$倍大于等于$$p(x)$$。首先从简单分布$$q(x)$$中采样得到$$x^*$$，然后以$$\frac{p(x^*)}{cq(x^*)}$$的概率保留这个样本，直到得到$$N$$个样本结束。  
{% include figure.html path="assets/img/Pasted image 20231002164638.png" width="100%" %}  
  
5. **MCMC**  
**MCMC方法可以从复杂分布中进行采样**  
**符号定义**：$$\pi_i\triangleq\pi(i)=\lim_{t\to +\infty}P(X_t=i)$$，即马尔科夫链达到平稳分布的时候，处于第$$i$$个状态的概率。  
**满足遍历定理的马尔可夫链**：从任意起始点出发，最终都会收敛到同一个平稳分布，即殊途同归。  
如果定义一个满足遍历定理的马尔可夫链，使得它的平稳分布等于目标分布$$p(x)$$，那么当经过足够长的游走时间$$m$$后，该链收敛，在之后的时间内每游走一次，就得到了服从目标分布的一个样本。该算法就被称为MCMC方法。  
现在最大的问题就是：**如何构造这么一个马尔可夫链，使得它的平稳分布等于目标分布？**  
TODO  
7. **基于MCMC的MLE方法**  
TODO  
# How Score-based Generative Model Work  
对于一个基于能量函数的概率分布：  
$$  
p_\theta(x)=\frac{e^{-E_\theta(x)}}{Z_\theta}  
$$  
对其对数似然求导：  
$$  
\begin{align}  
\nabla_\theta\log p_\theta(x) &=\nabla_\theta\log\exp(-E_\theta(x))-\nabla_\theta\log Z_\theta\\  
&=-\nabla_\theta\log\exp(E_\theta(x))-\frac{1}{Z_\theta}\nabla_\theta Z_\theta &\  ;\text{chain rule} \\  
&=-\nabla_\theta\log\exp(E_\theta(x))-\frac{1}{Z_\theta}\nabla_\theta \int\exp(-E_\theta(x))\mathrm dx &\ ;\text{definition of } Z_\theta\\  
&=-\nabla_\theta\log\exp(E_\theta(x))+\frac{1}{Z_\theta}\int\exp(-E_\theta(x))\nabla_\theta E_\theta(x)\mathrm dx &\ ;\text{chain rule}\\  
&=-\nabla_\theta\log\exp(E_\theta(x))+\int\frac{\exp(-E_\theta(x))}{Z_\theta}\nabla_\theta E_\theta(x)\mathrm dx\\  
&=-\nabla_\theta\log\exp(E_\theta(x))+\mathbb{E}_{x\sim p_\theta(x)}\left[\nabla_\theta E_\theta(x)\right] &\ ;\text{definition of } p_\theta(x)\\  
\end{align}  
$$  
  
基于score的生成模型和扩散模型非常相似，使用了score matching和Langevin dynamics技术进行生成。其中，  
1. score  matching是估计目标分布的概率密度的梯度 （即score，分数），记$$p(x)$$是数据分布的概率密度函数，则这个分布的score被定义为$$\nabla_x\log p(x)$$，score matching则是训练一个网络$$s_\theta$$去近似score：  
$$\mathcal{E}_{p(x)}\left[ \Vert\nabla_x\log p(x)-s_\theta(x)\Vert^2_2 \right]=\int p(x)\Vert\nabla_x\log p(x)-s_\theta(x)\Vert^2_2 dx$$  
3. Langevin dynamics是使用score采样生成数据，采样方式如下：  
$$  
x_t=x_{t-1}+\frac{\delta}{2}\nabla_x\log p(x_{t-1})+\sqrt{\delta}\epsilon, \text{    where } \epsilon\sim\mathcal{N}(0, I)  
$$  
# Reference  
1. [能量模型能做什么，和普通的神经网络模型有什么区别，为什么要用能量模型呢？](https://www.zhihu.com/question/499485994/answer/2552791458)  
2. [扩散模型与能量模型，Score-Matching和SDE，ODE的关系](https://zhuanlan.zhihu.com/p/576779879)  
3. [你一定从未看过如此通俗易懂的马尔科夫链蒙特卡罗方法(MCMC)解读(上)](https://zhuanlan.zhihu.com/p/250146007)  
4. [How to Train Your Energy-Based Models](https://arxiv.org/pdf/2101.03288.pdf)