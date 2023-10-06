---  
layout: distill  
title: Diffusion Models - DDIM  
date: 2023-10-06  
description: introduction about DDIM  
tags:  
  - diffusion-model  
giscus_comments: "true"  
authors:  
  - name: Wei Deng  
    url: "https://wdaicv.site"  
    affiliations:  
      name: nil  
toc:  
  - name: Review of DDPM  
  - name: From DDPM to DDIM  
  - name: Reference  
---  
# Review of DDPM  
1. Diffusion阶段  
$$  
\begin{align}  
q(x_t\vert x_0)&=\boxed{\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)}\\  
         &=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon  
         \text{ ,where } \epsilon\sim\mathcal{N}(0,I)  
\end{align}  
$$  
  
2. Reverse阶段  
使用贝叶斯公式  
$$  
\begin{align}  
q(x_{t-1}\vert x_t)&=\frac{q(x_t\vert x_{t-1})q(x_{t-1})}{q(x_t)}  
\end{align}  
$$  
发现公式中$$q(x_{t-1})$$和$$q(x_t)$$不好求，根据DDPM的马尔科夫假设：  
$$  
\begin{align}  
q(x_{t-1}\vert x_t)&=q(x_{t-1}\vert x_t,x_0)\\  
              &=\frac{q(x_t\vert x_{t-1},x_0)q(x_{t-1}\vert x_0)}{q(x_t\vert x_0)}\\  
              &=\frac{q(x_t\vert x_{t-1})q(x_{t-1}\vert x_0)}{q(x_t\vert x_0)}\\  
              &=\boxed{\mathcal{N}(x_{t-1};\mu(x_t;\theta),\sigma_t^2I)}  
\end{align}  
$$  
其中，$$\sigma_t$$可以用超参数表示，$$\mu(x_t;\theta)$$是一个神经网络，用于预测均值：  
$$  
\begin{align}  
\mu(x_t;\theta)&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+  
\frac{\sqrt{\bar{x}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0\\  
&=\boxed{\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+  
\frac{\sqrt{\bar{x}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\hat{x}_{0\vert t}}\\  
\sigma_t^2&=\boxed{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t}  
\end{align}  
$$  
# From DDPM to DDIM  
同样是对分布$$q(x_{t-1}\vert x_t,x_0)$$进行求解：  
$$  
\begin{align}  
q(x_{t-1}\vert x_t,x_0)  
&=\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0\vert t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon  
\text{ ,where }\epsilon\sim\mathcal{N}(0,I)  
\end{align}  
$$  
在上式中，$$\epsilon$$是一个噪声，虽然可以重新从高斯分布采样，但是也可以使用噪声估计网络估计出来的结果$$\epsilon_\theta(x_t,t)$$：  
$$  
\begin{align}  
q(x_{t-1}\vert x_t,x_0)  
&=\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0\vert t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon  
\text{ ,where }\epsilon\sim\mathcal{N}(0,I)\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0\vert t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t,t)\\  
\end{align}  
$$  
甚至可以同时考虑$$\epsilon$$和$$\epsilon_\theta(x_t,t)$$：  
$$  
\begin{align}  
q(x_{t-1}\vert x_t,x_0)  
&=\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0\vert t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon  
\text{ ,where }\epsilon\sim\mathcal{N}(0,I)\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0\vert t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t,t)\\  
\end{align}  
$$  
# Reference  
