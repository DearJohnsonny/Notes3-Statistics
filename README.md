
<a href="https://dearjohnsonny.github.io/Notes1-Biotech/">Notes1-Biotech</a>

<a href="https://dearjohnsonny.github.io/Notes2-Biotech/">Notes2-Biotech</a>

<a href="https://dearjohnsonny.github.io/Notes4-Linear-Algebra/">Notes4-Linear-Algebra</a>

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/195986587-43417de2-eb07-491d-96c6-1f1fb1170b21.png" width="900">
</div>

# About the thought of statistics

## 概率和统计是一个东西吗？
概率（probabilty）和统计（statistics）看似两个相近的概念，其实研究的问题刚好相反。

概率研究的问题是，已知一个模型和参数，怎么去预测这个模型产生的结果的特性（例如均值，方差，协方差等等）。 举个例子，我想研究怎么养猪（模型是猪），我选好了想养的品种、喂养方式、猪棚的设计等等（选择参数），我想知道我养出来的猪大概能有多肥，肉质怎么样（预测结果）。

统计研究的问题则相反。统计是，有一堆数据，要利用这堆数据去预测模型和参数。仍以猪为例。现在我买到了一堆肉，通过观察和判断，我确定这是猪肉（这就确定了模型。在实际研究中，也是通过观察数据推测模型是／像高斯分布的、指数分布的、拉普拉斯分布的等等），然后，可以进一步研究，判定这猪的品种、这是圈养猪还是跑山猪还是网易猪，等等（推测模型参数）。

一句话总结：概率是已知模型和参数，推数据。统计是已知数据，推模型和参数。

## 大数定律和中心极限定理
### 大数定律
定理  设 ${X_n}$ 是一列独立同分布 (i.i.d.) 的随机变量序列, 具有公共的数学期望 $\mu$ 和方差 $\sigma^2$. 则

$$
\bar{X}=\frac{1}{n} \sum_{k=1}^n X_k \stackrel{p}{\rightarrow} \mu,
$$

即 ${X_n}$ 服从 (弱) 大数定律。

理解：当样本容量足够大时，样本均值趋向于总体均值（说白了就是期望）

或者说实验次数足够多时，事件发生的频率就趋向于事件发生的概率

应用：大数定理说明了，足够大的样本可以反映出总体的真实组成。这让我们放心地去抽样调查，去monte carlo模拟等等

### 中心极限定理
定理  设 ${X_n}$ 为 i.i.d 的随机变量序列, 具有公共的数学期望 $\mu$ 和方差 $\sigma^2$. 则 $X_1+\cdots+X_n$ 的标准化形式 $\frac{1}{\sqrt{n} \sigma}\left(X_1+\cdots+X_n-n \mu\right)$ 满足中心极限定理. 即对任意 $x \in \mathbb{R}$, 有

$$
\lim _{n \rightarrow \infty} F_n(x)=\Phi(x),
$$

其中 $F_n(x)$ 为 $\frac{1}{\sqrt{n} \sigma}\left(X_1+\cdots+X_n-n \mu\right)$ 的分布函数, 而 $\Phi(x)$ 为标 准正态分布 $N(0,1)$ 的分布函数. 记为

$$
\frac{1}{\sqrt{n} \sigma}\left(X_1+\cdots+X_n-n \mu\right) \stackrel{d}{\rightarrow} N(0,1)
$$

证明：

这个定理是容易理解、记忆的。首先记住{ $X_n$ }的均值 $X_n$ 近似服从正态分布，接下来只需要解 出这个正态分布的期望和方差。期望有
$\mathbb{E} \bar{X}_n=\frac{1}{n} \sum_{i=1}^n \mathbb{E} X_i=\frac{n \mu}{n}=\mu$
方差有
$\mathbb{D} \bar{X}_n=\frac{1}{n^2} \sum_{i=1}^n \mathbb{D} X_i=\frac{n \sigma^2}{n^2}=\frac{\sigma^2}{n}$
那么 $\bar{X}_n$ 近似服从的正态分布就是 $N\left(\mu, \frac{\sigma^2}{n}\right)$ ，归一化后的随机变量 $\frac{\bar{X}_n-\mu}{\sigma / \sqrt{n}}$ 近似服从标准正态分布 $N(0,1)$.

理解：当样本容量足够大时，样本均值服从正态分布（样本均值的分布会慢慢变成正态分布）。其均值为总体均值，方差为总体方差的1/n。其实也是在刻画样本均值趋向于总体均值，只是刻画得更细（**不管总体是什么分布，任意一个总体的样本平均值都会围绕在总体的整体平均值周围，并且呈正态分布**）

![image](https://user-images.githubusercontent.com/111955215/196581592-fd44d86c-140d-4696-a099-2cadff3cbf23.png)

应用：样本量较大时的区间估计和**假设检验理论基础**

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/196572904-1a0fe79e-1c5c-4698-abe5-b7daf453980e.png" width="900">
</div>

## 频率学派与贝叶斯派
在说极大似然估计（Maximum Likelihood Estimate,MLE）与最大后验概率估计（Maximum A Posteriori estimation,MAP）之前，不得不说对于概率看法不同的两大派别频率学派与贝叶斯派。他们看待世界的视角不同，导致他们对于产生数据的模型参数的理解也不同。

### 频率学派

他们认为世界是确定的。他们直接为事件本身建模，也就是说事件在多次重复实验中趋于一个稳定的值p，那么这个值就是该事件的概率。

他们认为模型参数是个定值，希望通过类似解方程组的方式从数据中求得该未知数。这就是频率学派使用的参数估计方法-极大似然估计（MLE），这种方法往往在大数据量的情况下可以很好的还原模型的真实情况。

### 贝叶斯派

他们认为世界是不确定的，因获取的信息不同而异。假设对世界先有一个预先的估计，然后通过获取的信息来不断调整之前的预估计。 他们不试图对事件本身进行建模，而是从旁观者的角度来说。因此对于同一个事件，不同的人掌握的先验不同的话，那么他们所认为的事件状态也会不同。

他们认为模型参数源自某种潜在分布，希望从数据中推知该分布。对于数据的观测方式不同或者假设不同，那么推知的该参数也会因此而存在差异。这就是贝叶斯派视角下用来估计参数的常用方法-最大后验概率估计（MAP），这种方法在先验假设比较靠谱的情况下效果显著，**随着数据量的增加，先验假设对于模型参数的主导作用会逐渐削弱**，相反真实的数据样例会大大占据有利地位。极端情况下，比如把先验假设去掉，或者假设先验满足均匀分布的话，那她和极大似然估计就如出一辙了。

### 极大似然估计与最大后验概率估计

显然，下文会提到的MLE和MAP都是**统计**领域（而不是概率）的问题。它们都是用来**推测参数**的方法。

**MAP与MLE最大区别是MAP中加入了模型参数本身的概率分布，或者说。MLE中认为模型参数本身的概率的是均匀的，即该概率为一个固定值。**

#### 极大似然估计（Maximum Likelihood Estimate,MLE）
MLE认为，θ是非随机变量或者分布未知的随机变量，这两种情况都可以认为P(θ)均匀分布的，即该概率是一个固定值，P(θ)=C，所以其目标为：

$$
\hat{\theta}_{\mathrm{MLE}}=\arg \max _\theta P(x \mid \theta) C=\arg \max _\theta P(x \mid \theta)
$$

举例：假设有一个造币厂生产某种硬币，现在我们拿到了一枚这种硬币，想试试这硬币是不是均匀的。即想知道抛这枚硬币，正面出现的概率记为θ是多少？
得到的数据是：反正正正正反正正正反

其似然函数为：

$$
\mathrm{f}\left(\mathrm{x}_0, \theta\right)=(1-\theta) \times \theta \times \theta \times \theta \times \theta \times(1-\theta) \times \theta \times \theta \times \theta \times(1-\theta)=\theta^7(1-\theta)^3=\mathrm{f}(\theta)
$$

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/196455706-f82dc502-b1f3-4a4b-99c2-123ad3553e3b.png" width="900">
</div>

可以看出，在θ = 0.7时，似然函数取得最大值。则取值为0.7

#### 最大后验概率估计（Maximum A Posteriori estimation,MAP）
MAP认为，θ是一个随机变量，其先验概率密度函数是已知的，为P(θ)，所以其目标为最大化 $\mathrm{P}\left(\theta \mid \mathrm{x}_0\right)=\frac{\mathrm{P}\left(\mathrm{x}_0 \mid \theta\right) \mathrm{P}(\theta)}{\mathrm{P}\left(\mathrm{x}_0\right)}$， 不过因为 $\mathrm{x}_0$ 是确定的（即投出的“反正正正正反正正正反”)， $\mathrm{P}\left(\mathrm{x}_0\right)$ 是一个已知值，

因此需要优化的函数为：

$$
\hat{\theta}_{M A P}=\arg \max _\theta P(x \mid \theta) P(\theta)
$$

以上面的扔硬币的例子来看，对于一个硬币，大概率认为他是正常的，正面的概率为0.5的可能性最大，因此我们可以认为θ的分布为 $P(\theta)=\theta(1-\theta)$. 甚至可以是 $P(\theta)=\theta^n(1-\theta)^n$.（只需要保证 θ和1-θ的指数相同即可）

那么最后的似然函数为： $y=x^8(1-x)^4$ 得到的函数图与之前的相比不同：

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/196459724-d88c5b25-78fa-4079-baa0-59664a4eab79.png" width="1500">
</div>

除了 $P(\theta)=\theta^n(1-\theta)^n$ 这种函数以外， $\theta$ 的分布还可以是任何贝叶斯学派认为合理的分布，如：将θ的概率分布假设为均值为0.5，方差为1的正态分布，即θ的密度函数可表示为： $f(\theta)=\frac{1}{\sqrt{2 \pi}} e^{-\frac{(\theta-0.5)^2}{2}}$

举极端的例子，如果抛10、100、1000次硬币，得到的都是正面，则MLE认为θ为1，但是极大似然估计不是，得到的分布函数最大时，横轴的θ不为1

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/196460488-b7d8ac26-6fd5-4bbe-9ddc-cebe18895f52.png" width="1500">
</div>


##### 总结一下MAP
* 最大后验的实质就是对参数的每一个可能的取值，都进行极大似然估计，并根据这个取值可能性的大小，设置极大似然估计的权重，然后选择其中最大的一个，作为最大后验估计的结果。
* 随着数据量的增加，MAP的参数分布会更倾向于向数据靠拢，先验假设的影响会越来越小。
* 随着数据的增加，先验的作用越来越弱，数据的作用越来越强，MAP参数的分布会向着最大似然估计靠拢。
* 而且可以证明，最大后验估计的结果是先验和最大似然估计的凸组合。

# Basic  statistics knowledge

## 一些术语和指标
### 方差相关
#### 样本方差和总体方差
关于样本方差用n-1

$$
S^2=\frac{1}{n-1} \sum_{i=1}^n\left(X_i-\bar{X}\right)^2
$$

  而不是

$$
S^2=\frac{1}{n} \sum_{i=1}^n\left(X_i-\mu\right)^2
$$

  总体方差除以n的原因是什么？实际上是每个出现的概率为n分之一，因为每一项都是自由变换的。而在实际计算中，我们总是先求出了样本平均数，这样导致样本取值就无法像总体数值一样自由。样本的n项其实如果确定了（n-1）项，则第n项就百分百确定，所以每一项出现的概率只有（n-1）分之一，即自由度为n-1
#### 协方差
协方差用来刻画两个随机变量 X，Y之间的相关性

如果两个变量的变化趋势一致，也就是说如果其中一个大于自身的期望值，另外一个也大于自身的期望值，那么两个变量之间的协方差就是正值。

如果两个变量的变化趋势相反，即其中一个大于自身的期望值，另外一个却小于自身的期望值，那么两个变量之间的协方差就是负值。

协方差的公式如下：

$$
\sigma(x, y)=\frac{1}{n-1} \sum_a^b\left(x_i-\bar{x}\right)\left(y_i-\bar{y}\right)
$$

**方差就是协方差的一种特殊形式**，当两个变量相同时，协方差就是方差了

另一种用期望来表示的方式：

$$
\begin{aligned}
\operatorname{Cov}(X, Y)=& \mathrm{E}[(X-\mathrm{E}[X])(Y-\mathrm{E}[Y])] \\
&=\mathrm{E}[X Y]-2 \mathrm{E}[Y] \mathrm{E}[X]+\mathrm{E}[X] \mathrm{E}[Y] \\
&=\mathrm{E}[X Y]-\mathrm{E}[X] \mathrm{E}[Y]=\mathrm{E}[X Y]-\mathrm{E}[X] \mathrm{E}[Y]
\end{aligned}
$$

如果X与Y是统计独立的，那么二者之间的协方差就是0，因为两个独立的随机变量满足

$$
\mathrm{E}[X Y]=\mathrm{E}[X] \mathrm{E}[Y]
$$

但是反过来并不成立。即如果X与Y的协方差为0，二者并不一定是统计独立的。
#### 协方差矩阵
协方差矩阵就是很多个变量两两之间的协方差，构成的矩阵：给定d个随机变量，根据协方差的定义，求出两两之间的协方差（每个随机变量都有n个观测值，故都有一个观测样本的均值）

$$
\sigma\left(x_m, x_k\right)=\frac{1}{n-1} \sum_{i=1}^n\left(x_{m i}-\bar{x}_m\right)\left(x_{k i}-\bar{x}_k\right)
$$

故协方差矩阵为：

$$
\Sigma=\left[\begin{array}{ccc}
\sigma\left(x_1, x_1\right) & \cdots & \sigma\left(x_1, x_d\right) \\
\vdots & \ddots & \vdots \\
\sigma\left(x_d, x_1\right) & \cdots & \sigma\left(x_d, x_d\right)
\end{array}\right] \in R^{d \times d}
$$

其中，对角线上的元素为各个随机变量的方差，非对角线上的元素为两两随机变量之间的协方差，根据协方差的定义，我们可以认定：该协方差矩阵矩阵为对称矩阵(symmetric matrix)，其大小为d×d。

### 预测评价指标
#### MSE（Mean Square Error）均方误差
真实值与预测值的插值的平方然后求和平均

$$
M S E=\frac{1}{m} \sum_{i=1}^m\left(y i-f\left(x_i\right)\right)^2
$$

范围[0,+∞)，当预测值与真实值完全吻合时等于0，即完美模型；误差越大，该值越大。

#### RMSE（Root Mean Square Error）均方根误差
均方根误差是预测值与真实值偏差的平方与观测次数n比值的平方根。其实就是MSE加了个根号，这样数量级上比较直观，比如RMSE=10，可以认为回归效果相比真实值平均相差10。

衡量的是预测值与真实值之间的偏差，并且对数据中的异常值较为敏感。

$$
R M S E=\sqrt{\frac{1}{N} \sum_{i=1}^n\left(Y_i-f\left(x_i\right)\right)^2}
$$

RMSE与标准差对比：标准差是用来衡量一组数自身的离散程度，而均方根误差是用来衡量观测值同真值之间的偏差，它们的研究对象和研究目的不同，但是计算过程类似。

#### MAE（Mean Absolute Error）平均绝对误差
范围[0,+∞)，当预测值与真实值完全吻合时等于0，即完美模型；误差越大，该值越大。

$$
\mathrm{MAE}=\frac{1}{\mathrm{n}} \sum_{\mathrm{i}=1}^{\mathrm{n}}\left|\hat{y}_{\mathrm{i}}-\mathrm{y}_{\mathrm{i}}\right|
$$

#### MAPE（Mean Absolute Percentage Error）平均绝对百分比误差
范围[0,+∞)，MAPE 为0%表示完美模型，MAPE 大于 100 %则表示劣质模型。

可以看到，MAPE跟MAE很像，就是多了个分母。

注意点：当真实值有数据等于0时，存在分母0除问题，该公式不可用！

$$
\operatorname{MAPE}=\frac{100 \%}{\mathrm{n}} \sum_{\mathrm{i}=1}^{\mathrm{n}}\left|\frac{\hat{\mathrm{y}}_{\mathrm{i}}-\mathrm{y}_{\mathrm{i}}}{\mathrm{y}_{\mathrm{i}}}\right|
$$

#### SMAPE（Symmetric Mean Absolute Percentage Error）对称平均绝对百分比误差
注意点：当真实值有数据等于0，而预测值也等于0时，存在分母0除问题，该公式不可用！

$$
\text { SMAPE }=\frac{100 \%}{\mathrm{n}} \sum_{\mathrm{i}=1}^{\mathrm{n}} \frac{\left|\hat{\mathrm{y}}_{\mathrm{i}}-\mathrm{y}_{\mathrm{i}}\right|}{\left(\left|\hat{\mathrm{y}}_{\mathrm{i}}\right|+\left|\mathrm{y}_{\mathrm{i}}\right|\right) / 2}
$$

## 线性回归 Linear Regression
线性回归分析（Linear Regression Analysis)是确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法。本质上说，这种变量间依赖关系就是一种线性相关性，线性相关性是线性回归模型的理论基础

线性回归要做的是就是找到一个数学公式能相对较完美地把所有自变量组合（加减乘除）起来，得到的结果和目标接近。线性模型试图学习到一个通过属性的线性组合来进行预测的函数，一般用向量形式写成：

$$
f(X)=W^T X+b
$$

目标：模型预测出来的值和真实值无限接近

当此为一元线性回归时，向量W为标量m，该式子表示为：

$$
\hat{Y}=m X+b
$$

而线性回归实际上就是求优化的过程：

$$
m^*, b^*=\underset{m, b}{\arg \min } L(m, b)=\underset{m, b}{\arg \min } \frac{1}{N} \sum_{i=1}^N\left[\left(m x_i+b\right)-y_i\right]^2
$$

argmin(L)就是对L中的参数进行优化

### 一元线性回归
只用一个x来预测y的模型

$$
\hat{y}=\beta_0+\beta_1 x
$$

然而可以看到此为预测值，而真正的y值应该和y hat有一个误差，因此真实值为（μ代表误差）：

$$
y=\beta_0+\beta_1 x+\mu
$$

我们假定模型已经被求出，则用ei表示每一个预测值与真实值之间的差距：

$$
e_i=y_i-\hat{y}_i
$$

再将所有的e_i^2相加，就能量化出拟合的直线和实际之间的误差。公式如下：

$$
Q=\sum_1^n\left(y_i-\hat{y}_i\right)^2=\sum_1^n\left(y_i-\left(\hat{\beta}_0+\hat{\beta}_1 x_i\right)\right)^2
$$

这个公式是残差平方和，即SSE（Sum of Squares for Error），在机器学习中它是回归问题中最常用的损失函数（loss function）。这个公式是一个二次方程，上面公式中β0和β1未知，有两个未知参数的二次方程，画出来是一个三维空间中的图像，因此我们分别对β0和β1求偏导并令其为0：

$$
\begin{aligned}
&\frac{\partial Q}{\partial \beta_0}=2 \sum_1^n\left(y_i-\hat{\beta}_0-\hat{\beta}_1 x_i\right)=0 \\
&\frac{\partial Q}{\partial \beta_1}=2 \sum_1^n\left(y_i-\hat{\beta}_0-\hat{\beta}_1 x_i\right) x_i=0
\end{aligned}
$$

从而利用xi和yi将β0和β1反解出来，与中学背的那一串是一样的。这就是最小二乘法，“二乘”是平方的意思。最小二乘法：所选择的回归模型应该使所有观察值的残差平方和达到最小，即采用平方损失函数

### 多元线性回归
比一元线性回归复杂的是，多元线性回归组成的不是直线，是一个多维空间中的超平面，数据点散落在超平面两侧。比如二元线性回归预测的是一个平面。

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/195474650-1b26c285-29f0-4374-8505-2bba2ba5f6b4.png" width="900">
</div>

当有D种维度的影响因素时，机器学习领域将这D种影响因素成为特征(feature)，每个样本有一个需要预测的Y和一组D维向量X hat，原来的参数m变成了D维的向量W

表示为如下：

$$
\begin{gathered}
y_i=b+\sum_{d=1}^D w_d x_{i, d} \quad b=w_{D+1} \\
y_i=\sum_{d=1}^{D+1} w_d x_{i, d}
\end{gathered}
$$

为了表示方便，将独立出来的偏置项归纳到向量W中，b= WD+1，将D维特征扩展成为D+1维

$$
\mathbf{w}=\left[\begin{array}{c}
w_1 \\
w_2 \\
\vdots \\
w_D \\
w_{D+1}
\end{array}\right] \quad \mathbf{X}=\left[\begin{array}{c}
x_1 \\
x_2 \\
\vdots \\
x_N
\end{array}\right]=\left(\begin{array}{ccccc}
x_{1,1} & x_{1,2} & \cdots & x_{1, D} & 1 \\
x_{2,1} & x_{2,2} & \cdots & x_{2, D} & 1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
x_{N, 1} & x_{N, 2} & \cdots & x_{N, D} & 1
\end{array}\right) \quad \mathbf{y}=\left[\begin{array}{c}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{array}\right]
$$

从而：

$$
y_i=\sum_{d=1}^{D+1} w_d x_{i, d}=\mathbf{w}^{\top} \mathbf{x}_{\mathbf{i}} \Rightarrow \mathbf{y}=\mathbf{X} \mathbf{w}
$$

因此对多元线性回归的损失函数做最小二乘法：

$$
L(w)=(\mathbf{X} \mathbf{w}-\mathbf{y})^{\top}(\mathbf{X} \mathbf{w}-\mathbf{y})=\|\mathbf{X} \mathbf{w}-\mathbf{y}\|_2^2
$$

上图的右边读作L2范数的平方，下图展示了一个向量x的L2范数的平方及其导数

$$
\|\mathbf{x}\|_2^2=\left(\left(\sum_{i=1}^N x_i^2\right)^1 / 2\right)^2 \quad \nabla\|\mathbf{x}\|_2^2=2 \mathbf{x}
$$

因此对L(w)求导，得到下式：

$$
\frac{\partial}{\partial \mathbf{w}} L(\mathbf{w})=2 \mathbf{X}^{\top}(\mathbf{X} \mathbf{w}-\mathbf{y})=0
$$

从而推出：

$$
\begin{gathered}
\mathbf{X}^{\top} \mathbf{X} \mathbf{w}=\mathbf{X}^{\top} \mathbf{y} \Rightarrow\left(\mathbf{X}^{\top} \mathbf{X}\right)^{-1} \mathbf{X}^{\top} \mathbf{X} \mathbf{w}=\left(\mathbf{X}^{\top} \mathbf{X}\right)^{-1} \mathbf{X}^{\top} \mathbf{y} \\
\mathbf{w}^*=\left(\mathbf{X}^{\top} \mathbf{X}\right)^{-1} \mathbf{X}^{\top} \mathbf{y}
\end{gathered}
$$

最优解其实是在根据自变量向量X和因变量标量y求解。

**但是**： $X^T X$ 在现实任务中往往不是满秩矩阵（未知数大于方程个数。如：3个变量，但是只有2个方程，故无法求得唯一的解），所以无法求解矩阵的逆，故无法求得唯一的解。遇到这种情况，需要进行：1）降维处理（LASSO和PLS偏最小二乘法）；2）引入正则化(regularization)：将矩阵补成满秩

