# Deep Learning For Efficient Frontier in Portfoliomanagement
## Deep Learning method from Xavier Warin: Deep learning for efficient frontier calculation in finance

This repository contains code corresponding to a thesis developed at University of Munich (LMU).

Cosgun, Ozan:
Optimal Portfolio Selection with Conditional Value at Risk Criteria for Continuous Time Diffusion Processes with Applications in Deep Learning

### Deep Learning for efficient frontier
The code and methodology is based on the work "Deep Learning for efficient frontier calculation in finance" https://arxiv.org/abs/2101.02044 of [Warin, Xaver] https://scholar.google.com/citations?user=qxurzx4AAAAJ&hl=en.

### Methodology
For fix time horizon $T$ and discretization $0=t_0 \lt ...\lt t_N=T$ this code implements the Markowitz-type optimization problem of a underlying 
$X_T$ at end time horizon $T$ given by a accumulating of increments with starting value $X_0=x_0$ 
$$X_T= X_0 + \sum_{i=1}^{n-1} u_i X_{t_i} \frac{S_{t_{i+1}} - S_{t_i}}{S_{t_i}}$$
and optimizing the objective
$$\sup_u:\ \mathrm{E}\left(X_T^u \right)  - \beta CVaR(X_T^u)  \qquad \beta > 0$$
The approach, in its implementation being model independent, aims to train portfolio weights approximated in every time point by means of neural network. 
The weights $u = (u^1,...,u^n)$ are characterized by an porfolio agents constraints $$u_t^i \geq 0,\sum_{i=1}^{n} u_{t}^{i} = 1$$ in any time point $t$.


