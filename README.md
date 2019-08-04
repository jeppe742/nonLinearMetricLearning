# Non-Linear Metric Learning

This project is the result of the "02460 ADVANCED MACHINE LEARNING" course at DTU.

The code implements the NLMNN algorithm by [Kedem et. al.](https://papers.nips.cc/paper/4840-non-linear-metric-learning), and runs experiements on two different datasets.

## Simplex projection
During experiments it was found, that the softmax transformation, proposed by Kedem et. al, had several issues, like slow convergence and local minimas.

Inspired by [Yang et. al](http://downloads.hindawi.com/journals/mpe/2015/352849.pdf), we implemet an optimization scheme based on taking the full gradient step, and afterwards do a projection back into the simplex.

See the [report](report.pdf) for more details.


