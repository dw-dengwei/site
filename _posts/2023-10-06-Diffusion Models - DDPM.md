---  
layout: distill  
title: Diffusion Models - DDPM  
date: 2023-10-06  
description: introduction about DDPM  
tags:  
  - diffusion-model  
giscus_comments: "true"  
authors:  
  - name: Wei Deng  
    url: https://wdaicv.site  
    affiliations:  
      name: nil  
toc:  
  - name: Diffusion Process  
  - name: Reverse Process  
  - name: Training Object  
  - name: Summary  
  - name: Reference  
---  
# Diffusion Process  
前向扩散指的是将一个复杂分布转换成简单分布的过程$$\mathcal{T}:\mathbb{R}^d\mapsto\mathbb{R}^d$$，即：  
$$  
\mathbf{x}_0\sim p_\mathrm{complex}\Longrightarrow \mathcal{T}(\mathbf{x}_0)\sim p_\mathrm{prior}  
$$  
在DDPM中，将这个过程定义为**马尔可夫链**，通过不断地向复杂分布中的样本$$x_0\sim p_\mathrm{complex}$$添加高斯噪声。这个加噪过程可以表示为$$q(\mathbf{x}_t\vert\mathbf{x}_{t-1})$$：  
$$  
\begin{align}  
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) &= \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})\\  
\mathbf{x}_t&=\sqrt{1-\beta_t}\mathbf{x}_{t-1}+\sqrt{\beta_t}\boldsymbol\epsilon \quad \boldsymbol\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})  
\end{align}  
$$  
其中，$$\{\beta_t\in(0,1)\}^T_{t=1}$$，是超参数。  
从$$\mathbf{x}_0$$开始，不断地应用$$q(\mathbf{x}_t\vert\mathbf{x}_{t-1})$$，经过足够大的$$T$$步加噪之后，最终得到纯噪声$$\mathbf{x}_T$$：  
$$  
\mathbf{x}_0\sim p_\mathrm{complex}\rightarrow \mathbf{x}_1\rightarrow \cdots \mathbf{x}_t\rightarrow\cdots\rightarrow \mathbf{x}_T\sim p_\mathrm{prior}  
$$  
除了迭代地使用$$q(\mathbf{x}_t\vert\mathbf{x}_{t-1})$$外，还可以使用$$q(\mathbf{x}_t\vert\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$$一步到位，证明如下（两个高斯变量的线性组合仍然是高斯变量）：  
$$  
\begin{aligned}  
\mathbf{x}_t   
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} &\  ;\alpha_t=1-\alpha_t\\  
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2}  \\  
&= \dots \\  
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} &\  ;\boldsymbol{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I}),\bar{\alpha}_t=\prod_{i=1}^t \alpha_i\  
\end{aligned}  
$$  
一般来说，超参数$$\beta_t$$的设置满足$$0<\beta_1<\cdots<\beta_T<1$$，则$$\bar{\alpha}_1 > \cdots > \bar{\alpha}_T\to1$$，则$$\mathbf{x}_T$$会只保留纯噪声部分。  
# Reverse Process  
在前向扩散过程中，实现了：  
$$  
\mathbf{x}_0\sim p_\mathrm{complex}\rightarrow \mathbf{x}_1\rightarrow \cdots \mathbf{x}_t\rightarrow\cdots\rightarrow \mathbf{x}_T\sim p_\mathrm{prior}  
$$  
如果能够实现将前向扩散过程反转，也就实现了从简单分布到复杂分布的映射。逆向扩散过程则是将前向过程反转，实现从简单分布随机采样样本，迭代地使用$$q(\mathbf{x}_{t-1}\vert\mathbf{x}_t)$$，最终生成复杂分布的样本，即：  
$$  
\mathbf{x}_T\sim p_\mathrm{prior}\rightarrow \mathbf{x}_{T-1}\rightarrow \cdots \mathbf{x}_t\rightarrow\cdots\rightarrow \mathbf{x}_0\sim p_\mathrm{complex}  
$$  
为了求取$$q(\mathbf{x}_{t-1}\vert\mathbf{x}_t)$$，使用贝叶斯公式：  
$$  
\begin{align}  
q(\mathbf{x}_{t-1}\vert\mathbf{x}_t)&=\frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}  
\end{align}  
$$  
然而，公式中$$q(x_{t-1})$$和$$q(x_t)$$不好求，根据DDPM的马尔科夫假设，可以为$$q(\mathbf{x}_{t-1}\vert\mathbf{x}_t)$$添加条件（可以证明，如果向扩散过程中的$$\beta_t$$足够小，那么$$q(\mathbf{x}_{t-1}\vert\mathbf{x}_t)$$是高斯分布。）：  
$$  
\begin{align}  
q(\mathbf{x}_{t-1}\vert\mathbf{x}_t)&=q(\mathbf{x}_{t-1}\vert\mathbf{x}_t,\mathbf{x}_0)\\  
              &=\frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1},\mathbf{x}_0)q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)}{q(\mathbf{x}_t\vert\mathbf{x}_0)}\\  
              &=\frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)}{q(\mathbf{x}_t\vert\mathbf{x}_0)}\\  
              &=\mathcal{N}(\mathbf{x}_{t-1};\mu(\mathbf{x}_t;\theta),\sigma_t^2\mathbf I)  
\end{align}  
$$  
其中，$$\mu(x_t;\theta)$$是高斯分布的均值，$$\sigma_t$$可以用超参数表示：  
$$  
\begin{align}  
\mu(\mathbf{x}_t;\theta)&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t+  
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0\\  
\sigma_t&=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t  
\end{align}  
$$  
式中$$x_0$$可以反用公式$$\mathbf x_t=\sqrt{\bar{\alpha}_t}\mathbf x_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon_t$$：  
$$  
\mathbf x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon_t\right)  
$$  
则：  
$$  
\begin{align}  
\mu(\mathbf{x}_t;\theta)&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t+  
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0\\  
&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t+  
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar{\alpha}_t}\frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon_t\right)\\  
&=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_t\right)  
\end{align}  
$$  
而在推理的时候，$$\boldsymbol\epsilon_t$$是未知的，所以使用神经网络进行预测。综上，逆向扩散过程：  
$$  
\begin{align}  
q(\mathbf{x}_{t-1}\vert\mathbf{x}_t)&=\mathcal{N}(\mathbf{x}_{t-1};\mu(\mathbf{x}_t;\theta),\sigma_t^2\mathbf I)\\  
&=\mathcal{N}\left(\mathbf x_{t-1};\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_\theta(\mathbf x_t, t)\right),\left(\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t\right)^2\mathbf I\right)\\  
\mathbf x_{t-1}&=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_\theta(\mathbf x_t, t)\right)+\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t\cdot\boldsymbol\epsilon\quad\boldsymbol\epsilon\sim\mathcal N(\mathbf 0, \mathbf I)  
\end{align}  
$$  
# Training Object  
DDPM的训练目标是最小化训练数据的负对数似然：  
$$  
\begin{align}  
-\log p_\theta(\mathbf x_0) &\le -\log p_\theta(\mathbf x_0) + \mathrm{KL}\left(q(\mathbf x_{1:T}\vert\mathbf x_0)\Vert p_\theta(\mathbf x_{1:T}\vert\mathbf x_0)\right) &\ ;\mathrm{KL}(\cdot\Vert\cdot)\ge 0\\  
&=-\log p_\theta(\mathbf x_0)+\mathbb{E}_{\mathbf x_{1:T}\sim q(\mathbf x_{1:T}\vert\mathbf x_0)}\left[\log\frac{q(\mathbf x_{1:T}\vert\mathbf x_0)}{p_\theta(\mathbf x_{0:T})/p_\theta(\mathbf x_0)}\right]&\ ;p_\theta(\mathbf x_{1:T}\vert\mathbf x_0)=\frac{p_\theta(\mathbf x_{0:T})}{p_\theta(\mathbf x_0)}\\  
&=-\log p_\theta(\mathbf x_0)+\mathbb{E}_{\mathbf x_{1:T}\sim q(\mathbf x_{1:T}\vert\mathbf x_0)}\left[\log\frac{q(\mathbf x_{1:T}\vert\mathbf x_0)}{p_\theta(\mathbf x_{0:T})}+\log p_\theta(\mathbf x_0)\right]\\  
&=\mathbb{E}_{\mathbf x_{1:T}\sim q(\mathbf x_{1:T}\vert\mathbf x_0)}\left[\log\frac{q(\mathbf x_{1:T}\vert\mathbf x_0)}{p_\theta(\mathbf x_{0:T})}\right]\\  
\end{align}  
$$  
其中$$p_\theta(\mathbf x_{1:T}\vert\mathbf x_0)$$是使用网络估计分布$$q$$（变分推断），定义$$\mathcal{L}_{\mathrm{VLB}}\triangleq\mathbb{E}_q(\mathbf x_{0:T})\left[\log\frac{q(\mathbf x_{1:T}\vert\mathbf x_0)}{p_\theta(\mathbf x_{0:T})}\right]\ge-\mathbb{E}_{q(\mathbf x_0)}\log p_\theta(\mathbf x_0)$$，那么VLB是训练数据的负对数似然的上节，最小化VLB就是最小化负对数似然。继续对VLB拆分：  
$$  
\begin{align}  
\mathcal{L}_{\mathrm{VLB}}&=\mathbb{E}_{q(\mathbf x_{0:T})}\left[\log\frac{q(\mathbf x_{1:T}\vert\mathbf x_0)}{p_\theta(\mathbf x_{0:T})}\right]\\  
&=\mathbb{E}_q\left[\log\frac{\prod_{t=1}^{T}q(\mathbf x_t\vert\mathbf x_{t-1})}{p_\theta(\mathbf x_T)\prod_{t=1}^{T}p_\theta(\mathbf x_{t-1}\vert\mathbf x_t)}\right]\\  
&=\mathbb{E}_q\left[-\log p_\theta(\mathbf x_T)+\sum\limits^{T}_{t=1}\log\frac{q(\mathbf x_t\vert\mathbf x_{t-1})}{p_\theta(\mathbf x_{t-1}\vert\mathbf x_t)}\right]\\  
&=\mathbb{E}_q\left[-\log p_\theta(\mathbf x_T)+\sum\limits^{T}_{t=2}\log\frac{q(\mathbf x_t\vert\mathbf x_{t-1})}{p_\theta(\mathbf x_{t-1}\vert\mathbf x_t)}+\log\frac{q(\mathbf x_1\vert\mathbf x_0)}{p_\theta(\mathbf x_0\vert\mathbf x_1)}\right]\\  
&=\mathbb{E}_q\left[-\log p_\theta(\mathbf x_T)+\sum\limits^{T}_{t=2}\log\frac{q(\mathbf x_t\vert\mathbf x_{t-1}, \mathbf x_0)}{p_\theta(\mathbf x_{t-1}\vert\mathbf x_t)}+\log\frac{q(\mathbf x_1\vert\mathbf x_0)}{p_\theta(\mathbf x_0\vert\mathbf x_1)}\right] &\ ;q(\mathbf x_t\vert\mathbf x_{t-1})=q(\mathbf x_t\vert\mathbf x_{t-1}, \mathbf x_0)\\  
&=\mathbb{E}_q\left[-\log p_\theta(\mathbf x_T)+\sum\limits^{T}_{t=2}\log\left(\frac{q(\mathbf x_{t-1}\vert\mathbf x_{t}, \mathbf x_0)}{p_\theta(\mathbf x_{t-1}\vert\mathbf x_t)} \frac{q(\mathbf x_t\vert\mathbf x_0)}{q(\mathbf x_{t-1}\vert\mathbf x_0)}\right)+\log\frac{q(\mathbf x_1\vert\mathbf x_0)}{p_\theta(\mathbf x_0\vert\mathbf x_1)}\right] &\ ;\text{Bayes Theorem}\\  
	&=\mathbb{E}_q\left[\log\frac{q(\mathbf x_T\vert\mathbf x_0)}{p_\theta(\mathbf x_T)}+\sum_{t=2}^{T}\log\frac{q(\mathbf x_{t-1}\vert\mathbf x_t, \mathbf x_0)}{p_\theta(\mathbf x_{t-1}\vert\mathbf x_t)}-\log p_\theta(\mathbf x_0\vert\mathbf x_1)\right]\\  
&=\mathbb{E}_q\left[\underbrace{\mathrm{KL}(q(\mathbf x_T\vert\mathbf x_0) \Vert p_\theta(\mathbf x_T))}_{\mathcal{L}_T} + \sum_{t=2}^{T}\underbrace{\mathrm{KL}(q(\mathbf x_{t-1}\vert\mathbf x_t, \mathbf x_0) \Vert p_\theta(\mathbf x_{t-1}\vert\mathbf x_t))}_{\mathcal{L}_{t-1}}-\underbrace{\log p_\theta(\mathbf x_0\vert\mathbf x_1)}_{\mathcal{L}_0}\right]\\  
&=\mathbb{E}_q\left[\mathcal{L}_T+\sum_{t=2}^{T}\mathcal{L}_{t-1}-\mathcal{L}_0\right]  
\end{align}  
$$  
1. 由于$$\mathbf x_T$$是纯噪声，所以$$\mathcal{L}_T$$是常数  
2. 对于$$\mathcal{L}_0$$，DDPM专门设计了特殊的$$p_\theta(\mathbf x_0\vert\mathbf x_1)$$  
3. 对于$$\mathcal{L}_t\triangleq\mathrm{KL}(q(\mathbf x_t\vert\mathbf x_{t+1}, \mathbf x_0) \Vert p_\theta(\mathbf x_t \vert \mathbf x_{t+1})) \quad 1\le t \le T-1$$，是两个正态分布的KL散度，有解析解。在DDPM中，使用了简化之后的损失函数：  
$$  
\begin{align}  
\mathcal{L}_t^{\mathrm{simple}}&=\mathbb{E}_{t\sim[1,T],\mathbf x_0,\boldsymbol\epsilon_t}\left[\Vert\boldsymbol\epsilon_t-\boldsymbol\epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf x_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon_t,t)\Vert^2_2\right]  
\end{align}  
$$  
# Summary  
综上，DDPM的训练和采样/推理过程如下图所示：  
{% include figure.html path="assets/img/Pasted image 20231002142935.png" width="100%" %}  
# Reference  
1. [从零开始了解Diffusion Models](https://www.bilibili.com/video/BV13P411J7dm)  
2. [https://ayandas.me/blog-tut/2021/12/04/diffusion-prob-models.html](https://ayandas.me/blog-tut/2021/12/04/diffusion-prob-models.html)  
3. [What are Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)  
4. [An introduction to Diffusion Probabilistic Models](https://yang-song.net/blog/2021/score/)