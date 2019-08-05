---
layout: post
title: Pytorch Note1
date: 2017-11-21 09:32:24.000000000 +08:00
---

[TOC]

## hook函数
分为三个函数，分别是register_hook, register_backward_hook, register_forward_hook，第一个是针对Variable，后面两个是针对modules的

### 1. register_hook函数
针对中间层的Variable的梯度进行处理，比如修改和打印

```python

# 打印中间层Variable的梯度
import torch
from torch.autograd import Variable

grad_list = []

def hook(grad): # 该函数必须是function(grad)这种形式，grad的参数默认给出
    grad_list.append(grad)

x = Variable(torch.randn(2, 1), requires_grad=True)
y = x+2
z = torch.mean(torch.pow(y, 2))
lr = 1e-3
y.register_hook(hook)
z.backward()
x.data -= lr*x.grad.data

print grad_list


# 修改中间层Variable的梯度
x = Variable(torch.randn(5, 5), requires_grad=True)
y = x + 2
y.register_hook(lambda grad: grad * 2)
y.sum().backward()
x.grad # is now filled with 2

```

### 2. register_backward_hook函数
该函数是注册在module上的，而不是在Variable上，同时该module必须是一个function，而不是有container的函数，里面不能包含多个module。

具体的形式是function(module, grad_in, grad_out)，该函数可以返回一个新的grad_in用于替代原始的grad_in。而不是直接修改grad_in

```python
def bh(m,gi,go):
    print("Grad Input")
    print(gi)
    print("Grad Output")
    print(go)
    return gi[0]*0,gi[1]*0
# 注意当grad_in是多个值时，里面保存的形式是一个tuple(就是小括号的形式，优点是无法修改很安全)。
mod=Linear(3, 1, bias=False)
mod.register_backward_hook(bh) # 在这里给module注册了backward hook
```

### 3. register_forward_hook 函数
该函数是先进行正常的Forward方法， 然后对于Forward以后的结果，进行自定义的处理。
注意该函数与register_backward_hook不同，他不能改变output。而register_backward_hook是可以用一个新的grad_input来替代grad。


##### PS: 注意在dataParalle中的load函数的话，一定要在model parallel之后，进行load，因为通过parallel以后，函数的keys会在原来的key的基础上加module

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
