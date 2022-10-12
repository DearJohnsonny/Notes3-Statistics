
<a href="https://dearjohnsonny.github.io/Notes-from-DearJohn/">Notes from DearJohn</a>

<a href="https://dearjohnsonny.github.io/Notes-from-DearJohn-2/">Notes from DearJohn-2</a>

以下为测试公式

This sentence uses $ delimiters to show math inline: $\sqrt{3x-1}+(1+x)^2$

$$
S^2=\frac{1}{n} \sum_{i=1}^n\left(X_i-\mu\right)^2
$$

# Basic  statistics knowledge

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
