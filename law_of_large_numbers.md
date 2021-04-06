# NOTE
    本文为模拟大数定律和中心极限定理的py代码

## 一、大数定理
```py
import random
import matplotlib.pyplot as plt


def flip_plot(minExp, maxExp):
    """
    Assumes minExp and maxExp positive integers; minExp < maxExp
    Plots results of 2**minExp to 2**maxExp coin flips
    """
    # 两个参数的含义，抛硬币的次数为2的minExp次方到2的maxExp次方，也就是一共做了(2**maxExp - 2**minExp)批次实验，每批次重复抛硬币2**n次

    ratios = []
    xAxis = []
    for exp in range(minExp, maxExp + 1):
        xAxis.append(2**exp)
    # 循环批次实验
    for numFlips in xAxis:
        numHeads = 0 # 初始化，硬币正面朝上的计数为0
        # 对每批次进行重复抛硬币
        for n in range(numFlips):
            if random.random() < 0.5:  # random.random()从[0, 1)随机的取出一个数
                numHeads += 1  # 当随机取出的数小于0.5时，正面朝上的计数加1
        numTails = numFlips - numHeads  # 得到本次试验中反面朝上的次数
        ratios.append(numHeads/float(numTails))  #正反面计数的比值
    plt.title('Heads/Tails Ratios')
    plt.xlabel('Number of Flips')
    plt.ylabel('Heads/Tails')
    plt.plot(xAxis, ratios)
    plt.hlines(1, 0, xAxis[-1], linestyles='dashed', colors='r')
    plt.show()

#flip_plot(4, 16)



if __name__ == "__main__":
    flip_plot(4, 20)

```

## 二、中心极限定理

### 1、模拟服从伯努利分布的随机变量之和
```py
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def sampling2pmf(n, dist, t=10000):
    """
    n: sample size for each experiment
    t: how many times do you do experiment, fix in 10000
    dist: frozen distribution
    """
    ber_dist = dist
    sum_of_samples = []
    for i in range(t):
        samples = ber_dist.rvs(size=n)  # 与每次取一个值，取n次效果相同
        sum_of_samples.append(np.sum(samples))
    val, cnt = np.unique(sum_of_samples, return_counts=True)
    pmf = cnt / len(sum_of_samples)
    return val, pmf


def plot(n, dist, subplot, plt_handle):
    """
    :param n: sample size
    :param dist: distribution of each single sample
    :param subplot: location of sub-graph, such as 221, 222, 223, 224
    :param plt_handle: plt object
    :return: plt object
    """
    bins = 10000
    plt = plt_handle
    plt.subplot(subplot)
    mu = n * dist.mean()
    sigma = np.sqrt(n * dist.var())
    samples = sampling2pmf(n=n, dist=dist)
    plt.vlines(samples[0], 0, samples[1],
               colors='g', linestyles='-', lw=3)
    plt.ylabel('Probability')
    plt.title('Sum of bernoulli dist. (n={})'.format(n))
    # normal distribution
    norm_dis = stats.norm(mu, sigma)
    norm_x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, bins)
    pdf1 = norm_dis.pdf(norm_x)
    plt.plot(norm_x, pdf1, 'r--')
    return plt

size = [1, 4, 20, 80, 200, 1000]

# sum of bernoulli distribution
dist_type = 'bern'
bern_para = [0.4]
single_sample_dist = stats.bernoulli(p=bern_para[0])  # 定义一个伯努利分布

# 下面是利用matplotlib画图
plt.figure(1)
plt = plot(n=size[0], dist=single_sample_dist, subplot=321, plt_handle=plt)
plt = plot(n=size[1], dist=single_sample_dist, subplot=322, plt_handle=plt)
plt = plot(n=size[2], dist=single_sample_dist, subplot=323, plt_handle=plt)
plt = plot(n=size[3], dist=single_sample_dist, subplot=324, plt_handle=plt)
plt = plot(n=size[4], dist=single_sample_dist, subplot=325, plt_handle=plt)
plt = plot(n=size[5], dist=single_sample_dist, subplot=326, plt_handle=plt)
plt.tight_layout()
plt.savefig('sum_of_{}_dist.png'.format(dist_type), dpi=200)
```

### 2、模拟服从二项分布的随机变量之和

```py
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def sampling2pmf(n, dist, t=1000000):
    """
    n: sample size for each experiment
    t: how many times do you do experiment, fix in 10000
    dist: frozen distribution
    """
    ber_dist = dist
    sum_of_samples = []
    for i in range(t):
        samples = ber_dist.rvs(size=n)  # 与每次取一个值，取n次效果相同
        sum_of_samples.append(np.sum(samples))
    val, cnt = np.unique(sum_of_samples, return_counts=True)
    pmf = cnt / len(sum_of_samples)
    return val, pmf


def plot(n, dist, subplot, plt_handle):
    """
    :param n: sample size
    :param dist: distribution of each single sample
    :param subplot: location of sub-graph, such as 221, 222, 223, 224
    :param plt_handle: plt object
    :return: plt object
    """
    bins = 1000000
    plt = plt_handle
    plt.subplot(subplot)
    mu = n * dist.mean()
    sigma = np.sqrt(n * dist.var())
    samples = sampling2pmf(n=n, dist=dist)
    plt.vlines(samples[0], 0, samples[1],
               colors='g', linestyles='-', lw=3)
    plt.ylabel('Probability')
    plt.title('Sum of bernoulli dist. (n={})'.format(n))
    # normal distribution
    norm_dis = stats.norm(mu, sigma)
    norm_x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, bins)
    pdf1 = norm_dis.pdf(norm_x)
    plt.plot(norm_x, pdf1, 'r--')
    return plt

size = [1, 4, 20, 80, 200, 1000]

# sum of bernoulli distribution
dist_type = 'bino'
bino_para = [20, 0.4]
single_sample_dist = stats.binom(n=bino_para[0], p=bino_para[1])  # 定义一个二项分布

# 下面是利用matplotlib画图
plt.figure(1)
plt = plot(n=size[0], dist=single_sample_dist, subplot=321, plt_handle=plt)
plt = plot(n=size[1], dist=single_sample_dist, subplot=322, plt_handle=plt)
plt = plot(n=size[2], dist=single_sample_dist, subplot=323, plt_handle=plt)
plt = plot(n=size[3], dist=single_sample_dist, subplot=324, plt_handle=plt)
plt = plot(n=size[4], dist=single_sample_dist, subplot=325, plt_handle=plt)
plt = plot(n=size[5], dist=single_sample_dist, subplot=326, plt_handle=plt)
plt.tight_layout()
plt.show()
```

### 3、模拟服从均匀分布的随机变量之和

```py
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def sampling2pmf(n, dist, t=1000000):
    """
    n: sample size for each experiment
    t: how many times do you do experiment, fix in 10000
    dist: frozen distribution
    """
    ber_dist = dist
    sum_of_samples = []
    for i in range(t):
        samples = ber_dist.rvs(size=n)  # 与每次取一个值，取n次效果相同
        sum_of_samples.append(np.sum(samples))
    val, cnt = np.unique(sum_of_samples, return_counts=True)
    pmf = cnt / len(sum_of_samples)
    return val, pmf


def plot(n, dist, subplot, plt_handle):
    """
    :param n: sample size
    :param dist: distribution of each single sample
    :param subplot: location of sub-graph, such as 221, 222, 223, 224
    :param plt_handle: plt object
    :return: plt object
    """
    bins = 1000000
    plt = plt_handle
    plt.subplot(subplot)
    mu = n * dist.mean()
    sigma = np.sqrt(n * dist.var())
    samples = sampling2pmf(n=n, dist=dist)
    plt.vlines(samples[0], 0, samples[1],
               colors='g', linestyles='-', lw=3)
    plt.ylabel('Probability')
    plt.title('Sum of bernoulli dist. (n={})'.format(n))
    # normal distribution
    norm_dis = stats.norm(mu, sigma)
    norm_x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, bins)
    pdf1 = norm_dis.pdf(norm_x)
    plt.plot(norm_x, pdf1, 'r--')
    return plt

size = [1, 4, 20, 80, 200, 1000]

# sum of bernoulli distribution
dist_type = 'bino'
bino_para = [20, 0.4]
single_sample_dist = stats.binom(n=bino_para[0], p=bino_para[1])  # 定义一个二项分布

# 下面是利用matplotlib画图
plt.figure(1)
plt = plot(n=size[0], dist=single_sample_dist, subplot=321, plt_handle=plt)
plt = plot(n=size[1], dist=single_sample_dist, subplot=322, plt_handle=plt)
plt = plot(n=size[2], dist=single_sample_dist, subplot=323, plt_handle=plt)
plt = plot(n=size[3], dist=single_sample_dist, subplot=324, plt_handle=plt)
plt = plot(n=size[4], dist=single_sample_dist, subplot=325, plt_handle=plt)
plt = plot(n=size[5], dist=single_sample_dist, subplot=326, plt_handle=plt)
plt.tight_layout()
plt.show()
```

