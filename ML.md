机器学习包括有监督学习和无监督学习。其中有监督学习相比无监督学习最大的区别就是有无标签数据(label)，有标签的是监督学习，无标签的是无监督学习。常见的监督学习有分类和回归，无监督学习有聚类。我们会在这里简单介绍一下相关学习算法。

机器学习问题可以视为优化问题。我们以监督学习为例，给定数据集{$(x_1,y_1),\cdots,(x_n,y_n)$},学习算法就是要求出一个学习器$f^*$,使得
\begin{equation}
f^*=argmin_{f\in\mathcal{F}}F(f)
\end{equation}

其中$F$为性能度量函数，也可以称为损失函数。例如对于回归任务，常用均方误差MSE作为损失函数。其他常见的性能指标有错误率
## 线性模型和logistics模型
给定由d个属性$\mathbf{x}=(x_1,\cdots,x_d)$,线性模型就是试图学得一个通过属性的线性组合来预测的函数，即
\begin{equation}
f(\mathbf{x})=w_1x_1+\cdots+w_dx_d+b \quad \text{(1)}
\end{equation}
写成向量形式为
\begin{equation}
f(\mathbf{x})=\mathbf{w}^{T}\mathbf{x}+b\quad \text{(2)}
\end{equation}
其中向量$\mathbf{w}=(w_1,\cdots,w_d)^T$代表的各个属性的权重，b为偏置。$\mathbf{w}$和b学得之后，模型就确定了。

线性模型形式简单，模型可解释性强，易于建模。许多功能更加强大的非线性模型可以在线性模型的基础上通过引入层级结构或高位映射而得。

线性模型常用于回归任务，我们试图最小化均值方差来获得最优参数，即
\begin{equation}
(\mathbf{w}^*,b^*)=arg min_{(\mathbf{w},b)}\sum_{i=1}^{n}(f(\mathbf{x}_i)-y_i)^2
\end{equation}
表示成向量形式为：
\begin{equation}
\mathbf{w}^*=arg min_{\mathbf{w}}(\mathbf{y}-\mathbf{X}\mathbf{w})^T(\mathbf{y}-\mathbf{X}\mathbf{w})
\end{equation}
其中，

$$
\mathbf{X}=\left(\begin{array}{cccc}
x_{11} & \cdots & x_{1 d} & 1 \\
\vdots & \ddots & \vdots & \vdots\\
x_{n 1} & \cdots & x_{n d} & 1
\end{array}\right)
$$
$\mathbf{y}=(y_1,\cdot,y_n)^T$。

上述问题常用最小二乘法解决。当$\mathbf{X}^T\mathbf{X}$满秩矩阵时，可以求得最优解为
\begin{equation}
\mathbf{w}^*=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}\quad \text{(3)}
\end{equation}

通常在损失函数中引入正则化项，包括L1正则和L2正则。我们后面会介绍。
### logistics模型

logistics模型是一种分类模型。对于二分类模型，我们设y=1和y=0的后验概率估计分别为$p(y=1|x)$和$p(y=0|x)$，他们的对数几率表示为：
$$\ln\frac{p(y=1|x)}{p(y=0|x)}=\mathbf{w}^Tx+b$$
显然有
$$p(y=1|x)=\frac{e^{w^Tx+b}}{1+e^{w^T+b}}$$
$$p(y=0|x)=\frac{1}{1+e^{w^T+b}}$$
我们通过极大似然法来估计参数。对给定数据集$\{(x_i,y_i)\}_{i=1}^n$，对数似然为
\begin{equation}
l(w,b)=\sum_{i=1}^{n}\ln p(y_i|x_i;w,b) \quad \text{(4)}
\end{equation}
我们将$w^Tx+b$简写为$\beta^T\tilde{x}$，则上式可以重写为
\begin{equation}
l(\beta)=\sum_{i=1}^{n}(-y_i\beta_i^T\tilde{x}_i+\ln(1+e^{\beta_i^T\tilde{x}_i}) ) \quad \text{(5)}
\end{equation}

上式是关于参数的高阶可导连续凸函数，由凸优化理论，梯度下降法、牛顿法都可求得其最优解。
## SVM(支持向量机)

## 决策树

## 集成学习
### bagging和