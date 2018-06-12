# -*- coding:utf-8 -*-
"""
Sample script of Sequence to Sequence model.
You can also use Batch and GPU.
This model is based on below paper.

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le.
Sequence to sequence learning with neural networks.
In Advances in Neural Information Processing Systems (NIPS 2014).
"""
import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import cupy as cp
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

x=[[[1,2,3,4],[2,2,3,5],[2,2,3,6]],[[5,2,3,2],[1,0,9,10],[2,2,3,4]]]
x= Variable(cp.array(x, dtype=cp.float32))
print(x.shape)
y=F.concat(x,axis=1)
print(y)