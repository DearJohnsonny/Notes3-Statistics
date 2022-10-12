
<a href="https://dearjohnsonny.github.io/Notes-from-DearJohn/">Notes from DearJohn</a>

<a href="https://dearjohnsonny.github.io/Notes-from-DearJohn-2/">Notes from DearJohn-2</a>

以下为测试公式

This sentence uses $ delimiters to show math inline: $\sqrt{3x-1}+(1+x)^2$

$$
S^2=\frac{1}{n} \sum_{i=1}^n\left(X_i-\mu\right)^2
$$

# Basic  statistics knowledge

## 一些术语和指标

### MSE（Mean Square Error）均方误差
真实值与预测值的插值的平方然后求和平均

$$
M S E=\frac{1}{m} \sum_{i=1}^m\left(y i-f\left(x_i\right)\right)^2
$$

范围[0,+∞)，当预测值与真实值完全吻合时等于0，即完美模型；误差越大，该值越大。

### RMSE（Root Mean Square Error）均方根误差
均方根误差是预测值与真实值偏差的平方与观测次数n比值的平方根。

衡量的是预测值与真实值之间的偏差，并且对数据中的异常值较为敏感。

$$
R M S E=\sqrt{\frac{1}{N} \sum_{i=1}^n\left(Y_i-f\left(x_i\right)\right)^2}
$$

RMSE与标准差对比：标准差是用来衡量一组数自身的离散程度，而均方根误差是用来衡量观测值同真值之间的偏差，它们的研究对象和研究目的不同，但是计算过程类似。

## 线性回归 Linear Regression

## 方差分析
##### 关于样本方差用n-1

$$
S^2=\frac{1}{n-1} \sum_{i=1}^n\left(X_i-\bar{X}\right)^2
$$

  而不是

$$
S^2=\frac{1}{n} \sum_{i=1}^n\left(X_i-\mu\right)^2
$$

  总体方差除以n的原因是什么？实际上是每个出现的概率为n分之一，因为每一项都是自由变换的。而在实际计算中，我们总是先求出了样本平均数，这样导致样本取值就无法像总体数值一样自由。样本的n项其实如果确定了（n-1）项，则第n项就百分百确定，所以每一项出现的概率只有（n-1）分之一，即自由度为n-1
