---  
layout: distill  
title: Mathematics  
date: 2023-12-14  
description: Mathematic list  
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
# 得分匹配  
目标:  
  
$$  
\begin{align}  
J(\theta)  
&= \mathbb{E}_{x\sim p(x)}   
\left[  
\Vert s(x;\theta) - \nabla_x\log p(x) \Vert^2  
\right] \\  
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta)\Vert^2 + \underbrace{\Vert \nabla_x\log p(x)\Vert^2}_{\text{constant}} - 2s(x;\theta)^T\nabla_x\log p(x)  
\right ] \\  
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta) \Vert^2  
\right]   
- 2\int_{x\in\mathbb{R}^n} p(x)\sum_i s_i(x;\theta)\frac{\partial\log p(x)}{\partial x_i} \mathrm{d}x + \text{constant} \\  
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta) \Vert^2  
\right]   
- 2\sum_i\int_{x\in\mathbb{R}^n}s_i(x;\theta)\frac{\partial p(x)}{\partial x_i} \mathrm{d}x + \text{constant} \\  
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta) \Vert^2  
\right]   
-2\sum_i\underbrace{\iint\cdots\int}_{n}s_i(x;\theta)\frac{\partial p(x)}{\partial x_i}\mathrm{d}x_1\cdots\mathrm{d}x_n + \text{constant} \\   
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta) \Vert^2  
\right]   
-2\sum_i\iint\cdots\int_{x_i\in\mathbb{R}}s_i(x;\theta)\frac{\partial p(x)}{\partial x_i} \mathrm{d}x_i  
\iint\cdots\int\frac{\mathrm{d}x_1\cdots\mathrm{d}x_n}{\mathrm{d}x_i} + \text{constant} \\  
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta) \Vert^2  
\right] -\\  
&2\sum_i\iint\cdots\int_{x_i\in\mathbb{R}}  
\left(  
\underbrace{s_i(x;\theta) p(x)\bigg|_{x_i\to-\infty}^{x_i\to+\infty}}_{\lim_{x_i\to\infty} s_i(x;\theta)p(x)\to 0}  
-\frac{\partial s_i(x;\theta)}{\partial x_i}p(x)  
\right)\mathrm{d}x_i  
\iint\cdots\int\frac{\mathrm{d}x_1\cdots\mathrm{d}x_n}{\mathrm{d}x_i} + \text{constant} \\  
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta) \Vert^2  
\right]   
+2\sum_i\iint\cdots\int_{x_i\in\mathbb{R}}  
\left(  
\frac{\partial s_i(x;\theta)}{\partial x_i}p(x)  
\right)\mathrm{d}x_i  
\iint\cdots\int\frac{\mathrm{d}x_1\cdots\mathrm{d}x_n}{\mathrm{d}x_i} + \text{constant} \\  
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta) \Vert^2  
\right]  
+2\sum_i\underbrace{\iint\cdots\int}_{n}  
\frac{\partial s_i(x;\theta)}{\partial x_i}p(x)\mathrm{d}x_1\cdots\mathrm{d}x_n + \text{constant}\\  
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta) \Vert^2  
\right]  
+2\sum_i\int_{x\in\mathbb{R}^n}  
\frac{\partial s_i(x;\theta)}{\partial x_i}p(x)\mathrm{d}x + \text{constant}\\  
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta) \Vert^2  
\right]  
+2\int_{x\in\mathbb{R}^n}\sum_i  
\frac{\partial s_i(x;\theta)}{\partial x_i}p(x)\mathrm{d}x + \text{constant}\\  
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta) \Vert^2  
\right]  
+2\int_{x\in\mathbb{R}^n}  
\mathrm{tr}(\nabla_x s(x;\theta)) p(x)\mathrm{d}x + \text{constant}\\  
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta) \Vert^2  
\right]  
+2\mathbb{E}_{x\sim p(x)}  
\left[  
\mathrm{tr}(\nabla_x s(x;\theta))  
\right] + \text{constant}\\  
&= \mathbb{E}_{x\sim p(x)}  
\left[  
\Vert s(x;\theta) \Vert^2  
+2\mathrm{tr}(\nabla_x s(x;\theta))  
\right] + \text{constant}  
\end{align}  
$$  
  
# 期望符号的下标  
期望符号的下标有两种含义：  
1. 下标符号中的变量作为条件：$$\mathbb{E}_{x}\left[y\right]=\mathbb{E}\left[y\vert x\right]$$  
2. 下标符号中的变量用作计算平均：$$\mathbb{E}_{x}\left[y\right]=\int yp(x)\mathrm{d}x$$  
  
# KL散度和交叉熵相对Logits的梯度  
$$  
\frac  
{\partial \mathcal{L}_{\mathrm KD}}  
{\partial z_i}  
= \tau \left(\hat{y}^s_{i,\tau} - \hat{y}^t_{i,\tau}\right)  
$$  
  
$$  
\frac  
{\partial \mathcal{L}_{\mathrm CE}}  
{\partial z_i}  
= \hat{y}^s_{i,1} - \hat{y}_{i}  
$$