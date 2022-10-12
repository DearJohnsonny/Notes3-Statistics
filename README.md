
<a href="https://dearjohnsonny.github.io/Notes-from-DearJohn/">Notes from DearJohn</a>

<a href="https://dearjohnsonny.github.io/Notes-from-DearJohn-2/">Notes from DearJohn-2</a>

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/195361749-01b1343d-0dc6-497b-bbc1-1b0ae149ca5e.png" width="900">
</div>

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

### 一元线性回归
只用一个x来预测y

$$
\hat{y}=\beta_0+\beta_1 x
$$
