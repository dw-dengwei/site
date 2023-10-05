---  
title: 2023-10-05-Diffusion Models  
share: "true"  
---  
 %% TODO  
2023 年扩散模型还有什么可做的方向？ - 谷粒多·凯特的回答 - 知乎 https://www.zhihu.com/question/568791838/answer/3195773725 %%  
# 1 原理篇   
## 1.1 DDPM  
### 1.1.1 前向扩散  
前向扩散指的是将一个复杂分布转换成简单分布的过程$\mathcal{T}:\mathbb{R}^d\mapsto\mathbb{R}^d$，即：  
$$  
\mathbf{x}_0\sim p_\mathrm{complex}\Longrightarrow \mathcal{T}(\mathbf{x}_0)\sim p_\mathrm{prior}  
$$  
在DDPM中，将这个过程定义为**马尔科夫链**，通过不断地向复杂分布中的样本$x_0\sim p_\mathrm{complex}$添加高斯噪声。这个加噪过程可以表示为$q(\mathbf{x}_t|\mathbf{x}_{t-1})$：  
$$  
\begin{align}  
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) &= \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})\\  
\mathbf{x}_t&=\sqrt{1-\beta_t}\mathbf{x}_{t-1}+\sqrt{\beta_t}\boldsymbol\epsilon \quad \boldsymbol\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})  
\end{align}  
$$  
其中，$\{\beta_t\in(0,1)\}^T_{t=1}$，是超参数。  
从$\mathbf{x}_0$开始，不断地应用$q(\mathbf{x}_t|\mathbf{x}_{t-1})$，经过足够大的$T$步加噪之后，最终得到纯噪声$\mathbf{x}_T$：  
$$  
\mathbf{x}_0\sim p_\mathrm{complex}\rightarrow \mathbf{x}_1\rightarrow \cdots \mathbf{x}_t\rightarrow\cdots\rightarrow \mathbf{x}_T\sim p_\mathrm{prior}  
$$  
除了迭代地使用$q(\mathbf{x}_t|\mathbf{x}_{t-1})$外，还可以使用$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$一步到位，证明如下（两个高斯变量的线性组合仍然是高斯变量）：  
$$  
\begin{aligned}  
\mathbf{x}_t   
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} &\  ;\alpha_t=1-\alpha_t\\  
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2}  \\  
&= \dots \\  
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} &\  ;\boldsymbol{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I}),\bar{\alpha}_t=\prod_{i=1}^t \alpha_i\  
\end{aligned}  
$$  
一般来说，超参数$\beta_t$的设置满足$0<\beta_1<\cdots<\beta_T<1$，则$\bar{\alpha}_1 > \cdots > \bar{\alpha}_T\to1$，则$\mathbf{x}_T$会只保留纯噪声部分。  
### 1.1.2 逆向扩散  
在前向扩散过程中，实现了：  
$$  
\mathbf{x}_0\sim p_\mathrm{complex}\rightarrow \mathbf{x}_1\rightarrow \cdots \mathbf{x}_t\rightarrow\cdots\rightarrow \mathbf{x}_T\sim p_\mathrm{prior}  
$$  
如果能够实现将前向扩散过程反转，也就实现了从简单分布到复杂分布的映射。逆向扩散过程则是将前向过程反转，实现从简单分布随机采样样本，迭代地使用$q(\mathbf{x}_{t-1}|\mathbf{x}_t)$，最终生成复杂分布的样本，即：  
$$  
\mathbf{x}_T\sim p_\mathrm{prior}\rightarrow \mathbf{x}_{T-1}\rightarrow \cdots \mathbf{x}_t\rightarrow\cdots\rightarrow \mathbf{x}_0\sim p_\mathrm{complex}  
$$  
为了求取$q(\mathbf{x}_{t-1}|\mathbf{x}_t)$，使用贝叶斯公式：  
$$  
\begin{align}  
q(\mathbf{x}_{t-1}|\mathbf{x}_t)&=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}  
\end{align}  
$$  
然而，公式中$q(x_{t-1})$和$q(x_t)$不好求，根据DDPM的马尔科夫假设，可以为$q(\mathbf{x}_{t-1}|\mathbf{x}_t)$添加条件（可以证明，如果向扩散过程中的$\beta_t$足够小，那么$q(\mathbf{x}_{t-1}|\mathbf{x}_t)$是高斯分布。）：  
$$  
\begin{align}  
q(\mathbf{x}_{t-1}|\mathbf{x}_t)&=q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)\\  
              &=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}\\  
              &=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}\\  
              &=\mathcal{N}(\mathbf{x}_{t-1};\mu(\mathbf{x}_t;\theta),\sigma_t^2\mathbf I)  
\end{align}  
$$  
其中，$\mu(x_t;\theta)$是高斯分布的均值，$\sigma_t$可以用超参数表示：  
$$  
\begin{align}  
\mu(\mathbf{x}_t;\theta)&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t+  
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0\\  
\sigma_t&=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t  
\end{align}  
$$  
式中$x_0$可以反用公式$\mathbf x_t=\sqrt{\bar{\alpha}_t}\mathbf x_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon_t$：  
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
而在推理的时候，$\boldsymbol\epsilon_t$是未知的，所以使用神经网络进行预测。综上，逆向扩散过程：  
$$  
\begin{align}  
q(\mathbf{x}_{t-1}|\mathbf{x}_t)&=\mathcal{N}(\mathbf{x}_{t-1};\mu(\mathbf{x}_t;\theta),\sigma_t^2\mathbf I)\\  
&=\mathcal{N}\left(\mathbf x_{t-1};\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_\theta(\mathbf x_t, t)\right),\left(\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t\right)^2\mathbf I\right)\\  
\mathbf x_{t-1}&=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_\theta(\mathbf x_t, t)\right)+\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t\cdot\boldsymbol\epsilon\quad\boldsymbol\epsilon\sim\mathcal N(\mathbf 0, \mathbf I)  
\end{align}  
$$  
### 1.1.3 模型训练  
DDPM的训练目标是最小化训练数据的负对数似然：  
$$  
\begin{align}  
-\log p_\theta(\mathbf x_0) &\le -\log p_\theta(\mathbf x_0) + \mathrm{KL}\left(q(\mathbf x_{1:T}|\mathbf x_0)\|p_\theta(\mathbf x_{1:T}|\mathbf x_0)\right) &\ ;\mathrm{KL}(\cdot\|\cdot)\ge 0\\  
&=-\log p_\theta(\mathbf x_0)+\mathbb{E}_{\mathbf x_{1:T}\sim q(\mathbf x_{1:T}|\mathbf x_0)}\left[\log\frac{q(\mathbf x_{1:T}|\mathbf x_0)}{p_\theta(\mathbf x_{0:T})/p_\theta(\mathbf x_0)}\right]&\ ;p_\theta(\mathbf x_{1:T}|\mathbf x_0)=\frac{p_\theta(\mathbf x_{0:T})}{p_\theta(\mathbf x_0)}\\  
&=-\log p_\theta(\mathbf x_0)+\mathbb{E}_{\mathbf x_{1:T}\sim q(\mathbf x_{1:T}|\mathbf x_0)}\left[\log\frac{q(\mathbf x_{1:T}|\mathbf x_0)}{p_\theta(\mathbf x_{0:T})}+\log p_\theta(\mathbf x_0)\right]\\  
&=\mathbb{E}_{\mathbf x_{1:T}\sim q(\mathbf x_{1:T}|\mathbf x_0)}\left[\log\frac{q(\mathbf x_{1:T}|\mathbf x_0)}{p_\theta(\mathbf x_{0:T})}\right]\\  
\end{align}  
$$  
其中$p_\theta(\mathbf x_{1:T}|\mathbf x_0)$是使用网络估计分布$q$（变分推断），定义$\mathcal{L}_{\mathrm{VLB}}\triangleq\mathbb{E}_q(\mathbf x_{0:T})\left[\log\frac{q(\mathbf x_{1:T}|\mathbf x_0)}{p_\theta(\mathbf x_{0:T})}\right]\ge-\mathbb{E}_{q(\mathbf x_0)}\log p_\theta(\mathbf x_0)$，那么VLB是训练数据的负对数似然的上节，最小化VLB就是最小化负对数似然。继续对VLB拆分：  
$$  
\begin{align}  
\mathcal{L}_{\mathrm{VLB}}&=\mathbb{E}_{q(\mathbf x_{0:T})}\left[\log\frac{q(\mathbf x_{1:T}|\mathbf x_0)}{p_\theta(\mathbf x_{0:T})}\right]\\  
&=\mathbb{E}_q\left[\log\frac{\prod_{t=1}^{T}q(\mathbf x_t|\mathbf x_{t-1})}{p_\theta(\mathbf x_T)\prod_{t=1}^{T}p_\theta(\mathbf x_{t-1}|\mathbf x_t)}\right]\\  
&=\mathbb{E}_q\left[-\log p_\theta(\mathbf x_T)+\sum\limits^{T}_{t=1}\log\frac{q(\mathbf x_t|\mathbf x_{t-1})}{p_\theta(\mathbf x_{t-1}|\mathbf x_t)}\right]\\  
&=\mathbb{E}_q\left[-\log p_\theta(\mathbf x_T)+\sum\limits^{T}_{t=2}\log\frac{q(\mathbf x_t|\mathbf x_{t-1})}{p_\theta(\mathbf x_{t-1}|\mathbf x_t)}+\log\frac{q(\mathbf x_1|\mathbf x_0)}{p_\theta(\mathbf x_0|\mathbf x_1)}\right]\\  
&=\mathbb{E}_q\left[-\log p_\theta(\mathbf x_T)+\sum\limits^{T}_{t=2}\log\frac{q(\mathbf x_t|\mathbf x_{t-1}, \mathbf x_0)}{p_\theta(\mathbf x_{t-1}|\mathbf x_t)}+\log\frac{q(\mathbf x_1|\mathbf x_0)}{p_\theta(\mathbf x_0|\mathbf x_1)}\right] &\ ;q(\mathbf x_t|\mathbf x_{t-1})=q(\mathbf x_t|\mathbf x_{t-1}, \mathbf x_0)\\  
&=\mathbb{E}_q\left[-\log p_\theta(\mathbf x_T)+\sum\limits^{T}_{t=2}\log\left(\frac{q(\mathbf x_{t-1}|\mathbf x_{t}, \mathbf x_0)}{p_\theta(\mathbf x_{t-1}|\mathbf x_t)} \frac{q(\mathbf x_t|\mathbf x_0)}{q(\mathbf x_{t-1}|\mathbf x_0)}\right)+\log\frac{q(\mathbf x_1|\mathbf x_0)}{p_\theta(\mathbf x_0|\mathbf x_1)}\right] &\ ;\text{Bayes Theorem}\\  
	&=\mathbb{E}_q\left[\log\frac{q(\mathbf x_T|\mathbf x_0)}{p_\theta(\mathbf x_T)}+\sum_{t=2}^{T}\log\frac{q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)}{p_\theta(\mathbf x_{t-1}|\mathbf x_t)}-\log p_\theta(\mathbf x_0|\mathbf x_1)\right]\\  
&=\mathbb{E}_q\left[\underbrace{\mathrm{KL}(q(\mathbf x_T|\mathbf x_0) \| p_\theta(\mathbf x_T))}_{\mathcal{L}_T} + \sum_{t=2}^{T}\underbrace{\mathrm{KL}(q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0) \| p_\theta(\mathbf x_{t-1}|\mathbf x_t))}_{\mathcal{L}_{t-1}}-\underbrace{\log p_\theta(\mathbf x_0|\mathbf x_1)}_{\mathcal{L}_0}\right]\\  
&=\mathbb{E}_q\left[\mathcal{L}_T+\sum_{t=2}^{T}\mathcal{L}_{t-1}-\mathcal{L}_0\right]  
\end{align}  
$$  
1. 由于$\mathbf x_T$是纯噪声，所以$\mathcal{L}_T$是常数  
2. 对于$\mathcal{L}_0$，DDPM专门设计了特殊的$p_\theta(\mathbf x_0|\mathbf x_1)$  
3. 对于$\mathcal{L}_t\triangleq\mathrm{KL}(q(\mathbf x_t|\mathbf x_{t+1}, \mathbf x_0) \| p_\theta(\mathbf x_t | \mathbf x_{t+1})) \quad 1\le t \le T-1$，是两个正态分布的KL散度，有解析解。在DDPM中，使用了简化之后的损失函数：  
$$  
\begin{align}  
\mathcal{L}_t^{\mathrm{simple}}&=\mathbb{E}_{t\sim[1,T],\mathbf x_0,\boldsymbol\epsilon_t}\left[\|\boldsymbol\epsilon_t-\boldsymbol\epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf x_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon_t,t)\|^2_2\right]  
\end{align}  
$$  
### 1.1.4 总结  
综上，DDPM的训练和采样/推理过程如下图所示：  
![Pasted image 20231002142935.png](../assets/img/Pasted%20image%2020231002142935.png)  
## 1.2 基于score的生成模型  
### 1.2.1 引言  
score-based生成模型是一种新的生成模型范式，在score-based之前，主要存在两种类型的生成模型：  
1. **基于似然的生成模型**：基于最大似然估计（MLE），使得从真实数据分布中采样出的数据能够在所建模的数据分布中概率最大化，即$\max_\theta \mathbb{E}_{x\sim p_\mathrm{data}(x)}\left[\log p(x;\theta)\right]$。这类模型通过最大化似然函数，直接学习数据集的分布，主要方法有VAE、流模型、能量模型  
2. **隐式生成模型**：非直接学习数据集的分布，主要方法有GAN  
它们分别具有以下限制：  
1. **对于基于似然的生成模型**：对神经网络的结构要求高  
2. **对于隐式生成模型**：训练不稳定、容易导致模式坍塌  
### 1.2.2 先验知识  
score-based生成模型就能很好地避免这些限制，在介绍score-based生成模型之前，需要明确几个概念：  
1. **能量函数**  
对于大多数分布，可以使用概率密度函数（PDF）进行描述，也可以使用能量函数进行描述：  
$$  
p(x)=\frac{e^{-E(x)}}{Z}  
$$  
其中$p(x)$是PDF，$E(x)$是能量函数，$Z=\int e^{-E(x)}\mathrm{d}x$是归一化因子。以高斯分布为例，可以使用以下能量函数进行描述：  
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
基于能量函数对数据分布进行建模的时候，**如何计算能量模型的归一化函数$Z_\theta$是一个较难的问题**。传统的基于似然的生成模型（如自回归模型、流模型、VAE）都有自己的解决方式，但是都对能量模型做了太多的约束，各自都有其限制。  
3. **蒙特卡罗采样方法：拒绝-接受法**  
目的：希望从一个复杂分布$p(x)$采样$N$个样本  
方法：使用一个简单分布$q(x)$为媒介（例如：高斯分布），这个分布必须满足它的$c>0$倍大于等于$p(x)$。首先从简单分布$q(x)$中采样得到$x^*$，然后以$\frac{p(x^*)}{cq(x^*)}$的概率保留这个样本，直到得到$N$个样本结束。  
![Pasted image 20231002164638.png](../assets/img/Pasted%20image%2020231002164638.png)  
  
5. **MCMC**  
**MCMC方法可以从复杂分布中进行采样**  
**符号定义**：$\pi_i\triangleq\pi(i)=\lim_{t\to +\infty}P(X_t=i)$，即马尔科夫链达到平稳分布的时候，处于第$i$个状态的概率。  
**满足遍历定理的马尔可夫链**：从任意起始点出发，最终都会收敛到同一个平稳分布，即殊途同归。  
如果定义一个满足遍历定理的马尔可夫链，使得它的平稳分布等于目标分布$p(x)$，那么当经过足够长的游走时间$m$后，该链收敛，在之后的时间内每游走一次，就得到了服从目标分布的一个样本。该算法就被称为MCMC方法。  
现在最大的问题就是：**如何构造这么一个马尔可夫链，使得它的平稳分布等于目标分布？**  
TODO  
7. **基于MCMC的MLE方法**  
TODO  
### 1.2.3 score-based模型的解决方案  
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
1. score  matching是估计目标分布的概率密度的梯度 （即score，分数），记$p(x)$是数据分布的概率密度函数，则这个分布的score被定义为$\nabla_x\log p(x)$，score matching则是训练一个网络$s_\theta$去近似score：  
$$\mathcal{E}_{p(x)}\left[ \|\nabla_x\log p(x)-s_\theta(x)\|^2_2 \right]=\int p(x)\|\nabla_x\log p(x)-s_\theta(x)\|^2_2 dx$$  
3. Langevin dynamics是使用score采样生成数据，采样方式如下：  
$$  
x_t=x_{t-1}+\frac{\delta}{2}\nabla_x\log p(x_{t-1})+\sqrt{\delta}\epsilon, \text{    where } \epsilon\sim\mathcal{N}(0, I)  
$$  
## 1.3 DDIM  
#### 1.3.1.1 Review of DDPM  
1. Diffusion阶段  
$$  
\begin{align}  
q(x_t|x_0)&=\boxed{\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)}\\  
         &=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon  
         \text{ ,where } \epsilon\sim\mathcal{N}(0,I)  
\end{align}  
$$  
  
2. Reverse阶段  
使用贝叶斯公式  
$$  
\begin{align}  
q(x_{t-1}|x_t)&=\frac{q(x_t|x_{t-1})q(x_{t-1})}{q(x_t)}  
\end{align}  
$$  
发现公式中$q(x_{t-1})$和$q(x_t)$不好求，根据DDPM的马尔科夫假设：  
$$  
\begin{align}  
q(x_{t-1}|x_t)&=q(x_{t-1}|x_t,x_0)\\  
              &=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}\\  
              &=\frac{q(x_t|x_{t-1})q(x_{t-1}|x_0)}{q(x_t|x_0)}\\  
              &=\boxed{\mathcal{N}(x_{t-1};\mu(x_t;\theta),\sigma_t^2I)}  
\end{align}  
$$  
其中，$\sigma_t$可以用超参数表示，$\mu(x_t;\theta)$是一个神经网络，用于预测均值：  
$$  
\begin{align}  
\mu(x_t;\theta)&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+  
\frac{\sqrt{\bar{x}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0\\  
&=\boxed{\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+  
\frac{\sqrt{\bar{x}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\hat{x}_{0|t}}\\  
\sigma_t^2&=\boxed{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t}  
\end{align}  
$$  
### 1.3.2 From DDPM to DDIM  
  
同样是对分布$q(x_{t-1}|x_t,x_0)$进行求解：  
$$  
\begin{align}  
q(x_{t-1}|x_t,x_0)  
&=\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0|t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon  
\text{ ,where }\epsilon\sim\mathcal{N}(0,I)  
\end{align}  
$$  
在上式中，$\epsilon$是一个噪声，虽然可以重新从高斯分布采样，但是也可以使用噪声估计网络估计出来的结果$\epsilon_\theta(x_t,t)$：  
$$  
\begin{align}  
q(x_{t-1}|x_t,x_0)  
&=\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0|t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon  
\text{ ,where }\epsilon\sim\mathcal{N}(0,I)\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0|t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t,t)\\  
\end{align}  
$$  
甚至可以同时考虑$\epsilon$和$\epsilon_\theta(x_t,t)$：  
$$  
\begin{align}  
q(x_{t-1}|x_t,x_0)  
&=\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0|t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon  
\text{ ,where }\epsilon\sim\mathcal{N}(0,I)\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0|t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t,t)\\  
\end{align}  
$$  
# 2 应用篇  
## 2.1 SR3  
超分，训练数据是LR和SR配对的图片，以LR图片作为condition，生成SR图片  
## 2.2 CDM  
超分，级联的方式对小图进行超分，采用的方法就是SR3  
![725](../assets/img/Pasted%20image%2020230927200225.png)  
## 2.3 SDEdit  
![900](../assets/img/Pasted%20image%2020230927200246.png)  
由于加噪过程是首先破坏高频信息，然后才破坏低频信息，所以加噪到一定程度之后，就就可以去掉不想要的细节纹理，但仍保留大体结构，于是生成出来的图像就既能遵循输入的引导，又显得真实。但是需要 realism-faithfulness trade-off  
## 2.4 ILVR  
给定一个参考图像$y$，通过调整DDPM去噪过程，希望让模型生成的图像接近参考图像，作者定义的接近是让模型能够满足  
$$  
\phi_N(x_t)=\phi_N(y_t)  
$$  
$\phi_N(\cdot)$是一个低通滤波器（下采样之后再插值回来）。使用如下的算法：  
![450](../assets/img/Pasted%20image%2020230927201110.png)  
即，对DDPM预测的$x'_{t-1}$加上bias：$\phi_N(y_{t-1})-\phi_N(x'_{t-1})$，可以证明，如果上/下采样采用的是最近邻插值，使用这种方法可以使得$\phi_N(x_t)=\phi_N(y_t)$.  
这种方法和classifier guidance很相似，甚至不需要训练一个外部模型，对算力友好。  
## 2.5 DiffusionCLIP  
基于扩散模型的图像编辑，使用到的技术有DDIM Inversion，CLIP  
TODO  
# 3 参考  
1. [从零开始了解Diffusion Models](https://www.bilibili.com/video/BV13P411J7dm)  
2. [https://ayandas.me/blog-tut/2021/12/04/diffusion-prob-models.html](https://ayandas.me/blog-tut/2021/12/04/diffusion-prob-models.html)  
3. [What are Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)  
4. [An introduction to Diffusion Probabilistic Models](https://yang-song.net/blog/2021/score/)  
5. [能量模型能做什么，和普通的神经网络模型有什么区别，为什么要用能量模型呢？](https://www.zhihu.com/question/499485994/answer/2552791458)  
6. [扩散模型与能量模型，Score-Matching和SDE，ODE的关系](https://zhuanlan.zhihu.com/p/576779879)  
7. [你一定从未看过如此通俗易懂的马尔科夫链蒙特卡罗方法(MCMC)解读(上)](https://zhuanlan.zhihu.com/p/250146007)  
8. [How to Train Your Energy-Based Models](https://arxiv.org/pdf/2101.03288.pdf)