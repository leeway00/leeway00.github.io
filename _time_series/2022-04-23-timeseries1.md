---
title: "Time Series 1"

category: Time Series
tags:
  - [statistics]

# layout: single_v2

permalink: /time_series/ts1/
excerpt: "Chapter 3. Special Distributions"
last_modified_at: Now

toc: true
toc_sticky: true
katex: true
---

multiple h1 is availalbe?
## White noise
We assume it is uncorrelated
1. $Cov(a_t, a_{t+1}) = 0$ Covariance measures linear relationship between X, Y. e.g.) $Y=X^2+1$ have $Cov=0$ 
    - e.g.2) What if $Y=X^2$? $Cov(X,Y) = E[X \cdot X^2] - E[x]E[X^2]$. 
    - Since X and Y are (0,0) symmetric, $Cov(X,Y) = 0$
2. $E(a_t) = 0$
3. $Var(a_t) = \sigma^2$ constatnt w.r.t. time

=> Denoted as $a_t \sim^{iid} witenoise(0, \sigma^2)$\

Usually assume $i.i.d.$ wn. Independent is sufficient condition of $Cov(a_t, a_{t+1}) = 0$ but not for the other way.

## Stationary time series
$X_t$ is statoary  = $X_t$'s distribution does not change
### Strictly Stationary
$Let\:F_{t_1,t_2,\cdots, t_n}(x_1, \cdots, x_n) = P(t\leq x_1, \cdots, t_n\leq x_n )$. If this equals with $F_{t_{1+h},\cdots, t_{n+h}}(x_{1+h}, \cdots, x_{n+h})$, then the time series is stationary.\
If then, $Cov(x_t, X_s) = Cov(X_{t+h}, X_{s+h})$
Why? looking at cdf of random variable is impossible. At most joint pobability of 2 variable are possible to find.
### Weakly stationary
In most case it is weakly stationary if we call some process is **stationary**.
$X_t$ has $Cov(X_t, X_{t+h})$ only dependent on h and not dependent on t.
As such, $\gamma (h) = Cov(X_t, X_{t+h})$ is a **Auto Covariance Function**
1. $E[x_t]$ is constant
2. $Cov(X_t, X_s) = Cov(X_{t+},X_{s+h})$, for all $s, t$
    1. for $t=s$, Var(x_t) = Var(X_{t+h})$
> White Noise is stationary?\
> Weekly 1) Yes. 2) Yes. 2-1) Yes as $sigma^2$

전구 하나에 50개, 퓨즈 한번에 50개. 관계가 있는지를 정확히 할거면, 50*50의 관계를 모두 관찰해야한다. 이게 어렵기 때문에 stationary를 가정한다.

## Auto Correlation Function
Signal scale에 무관하게 볼 수 있다. lag와 시계열에 얼마나 유사성이 있는가. Auto corr, auto var가지고, stationary func인지 알수 있다. 연구하는 루틴은,
1. trend 없애고
2. residual가지고 sequence가지고 만든다.

### Autocorrelation function
From auto-covariacnce function $\gamma_x(h) = E[X_t, X_{t+h}]$,  auto-correlation function $\rho_x(h) = \frac{\gamma_x(h)}{\gamma_x(0)}]$ is defined. $\rho$의 범위는 cauchy property로 증명 가능하다.

#### Properties
1) $abs(\rho_x(h)) \leq \gamma_x(0)$
2) $\gamma (h) = \gamma (-h)$\
    > Proof\
    > $\gamma (h) = \gamma(t+h-t) = Cov(X_{t+h}+X_t)$\
    $= \gamma (t-(t-h))=Cov(X_t, X_{t-h} = \gamma (-h)$\
    > Thus, $\gamma (h) = \gamma (-h)$

### Linear Models
#### Moving Average of order q $MA(q)$
$X_t = a_t - \theta_1 a_{t-1} - \cdots - \theta_q a_{t-q}$ 
1. 화이트 노이즈가 들어가있다고 가정.
2. 직전의 노이들이 $\theta$ 만큼 영향

So that $MA(1) = a_t - \theta_1 a_{t-1}$.
##### Is it statrionary?
1) $E[X_t] = E[a_t] - \theta E[a_{t-1}]+\mu = \mu$ is constnat
2) $Cov(x_t, X_{t+h})=\gamma(h)$
   1) $\begin{aligned}[t] \gamma(0) = Cov(a_t - \theta_1 a_{t-1}, a_t - \theta_1 a_{t-1}) = \sigma_a^2 + \theta^2 \cdot \sigma_q^2 \end{aligned}$
   1) This is because white noises are uncorrelated with each other in other time steps.


Something needed more here


## Smoothing
### Moving Average: $X_t = m_t+Y_t$

$$Let\: W_t = \frac{1}{2q+1}\sum_{j=-q}^{q} X_{t-j} \\
= \frac{1}{2q+1}\sum_{j=-q}^{q} ,_{t-j} + \frac{1}{2q+1}\sum_{j=-q}^{q} Y_t-j \\
= m_t$$

Residual을 smoothing해버리면 0에가까워질 것이라는 인사이트로 moving average filter를 사용하는 것이다

### Exponential Smoothing
$\hat{m_t} = \alpha X_t + (1-\alpha)\hat{X_{t-1}}$

### Smoothing Splines
What is Spline? Connecting method!
3-point interpolation is cubic spline.
If there's too many points, 점들을 다 연결하기에, trend를 놓칠 수 있다. 이걸 부드럽게 만들어주는 regularized cubic spline을 사용한다. 어느정도 smooth할지 parameter를 본다.\
Given by 
$$argmin\:\sum_{i=1}^n [X_t-f_t]^2 + \lambda \int (f''_t)^2dt$$ 
where $f_t$ is 구간별로 3차식을 사용하여 fit 시킨 것이다. 뒷 항은 smoothing level이다.\
How to find lambda? Cross validation을 사용하면 된다. 
1. lambda is fixed as initial point. 
2. Randomly find 30% of the data as test data. 
3. Train the model with the rest of the data.
4. Randomly delete 30% data, calcualte SSR and keep repeat. 
5. Change lambda and repeat.
=> we can find lambda that minimize SSR

### Kernel Smoothing
all observed point $n$ affect t. i-th observation's constant. W is the weight that t has on i-th observation.
$$\hat{m_t} = \sum_{i=1}^{n} W_i(t)X_i\\ where\ W_i(t) = K(\frac{t-i}{b})/\sum_{j=1}^{n} K(\frac{t-j}{b})\\ and\ K(z) =\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{z^2}{2}\right)$$
As a result, 영향이 거리에 따라서 감소한다.\
b is **bandwith**

### Trend elimination by differencing
$X_t = m_t + Y_t$
> e.g.) By OLS, we got\
> $\hat{m_t} = -11.2 + 0.5t$\
> $\hat{Y_t} = X_t -\hat{m_t} -X_t+11.2 - 0.5t$
> Look at this residual is stationary or not.

$Z_t = X_t - X_{t-1}= (m_t - m_{t-1}) + \dots$ and first term is constant
if $Y_t$ is stationary, $Z_t$ is also stationary.

For the sake of notation,
> $\nabla X_t = X_t - X_{t-1} = (1-B)X_t$\
> $\nabla\nabla(X_t) = \nabla(X_t-X_{t-1}) = (X_t - X_{t-1}) - (X_{t-1} - X_{t-2}) = X_t - 2X_{t-1} + X_{t-2} = (1-B)^2X_t$\
> where Backshift Operator $BX_t = X_{t-1}$
> So that, $\nabla^d = (1-B)^d$
Seasonality?
FFT, periodogram!

## Autoregressive model $AR(1)$
$$X_t = \phi X_{t-1}+a_t+\mu$$ 
where $\mu$ is offset, $a_t$ is white noise at time t.\
> $E[X_t] = \phi E[X_{t-1}] + \mu$\
> $Var[X_t] = \phi^2 Var[X_{t-1}] + 2\phi Var[X_{t-1}, a_t] \sigma_a^2 = \phi^2 Var[X_{t-1}]+\sigma_a^2$\
> Since stationary or Causal time series, 지금의 관측치는 미래의 노이즈와는 무관하다. so, middle part =0

