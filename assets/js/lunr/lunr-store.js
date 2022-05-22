var store = [{
        "title": "Chapter3 Special Distributions",
        "excerpt":"Bernoulli and Binomial Bernoulli Distribution Bernoulli experiment a random experiment that outcome are classified with two mutually exclusive and exhaustive ways Bernoulli process a sequence of Bernoulli trials. Let X be a random variable associated with a Bernoulli trial The pmf of X is p(x)=px(1−p)1−x,x=0,1p(x) = p^x(1-p)^{1-x}, x=0,1p(x)=px(1−p)1−x,x=0,1 The expected...","categories": ["Mathematical Statistics"],
        "tags": ["Distribution"],
        "url": "/math-stat/ch3/",
        "teaser": null
      },{
        "title": "Questions in Python",
        "excerpt":"General Python          How class and get works https://stackoverflow.com/a/44741455            Adjusting tqdm message with sys.stdout and sys.stderr modification https://stackoverflow.com/questions/36986929/redirect-print-command-in-python-script-through-tqdm-write/37243211#37243211       Pandas          Make DataFrame iterable based on index (row-wise iteration) https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas            How to use MultiIndex? Advanced pandas https://towardsdatascience.com/how-to-use-multiindex-in-pandas-to-level-up-your-analysis-aeac7f451fce       Numpy     Hadamard product https://stackoverflow.com/questions/40034993/how-to-get-element-wise-matrix-multiplication-hadamard-product-in-numpy  ","categories": ["Python"],
        "tags": ["Python","Pandas","Numpy"],
        "url": "/python/ask-python/",
        "teaser": null
      },{
        "title": "Time Series 1",
        "excerpt":"multiple h1 is availalbe? White noise We assume it is uncorrelated Cov(at,at+1)=0Cov(a_t, a_{t+1}) = 0Cov(at​,at+1​)=0 Covariance measures linear relationship between X, Y. e.g.) Y=X2+1Y=X^2+1Y=X2+1 have Cov=0Cov=0Cov=0 e.g.2) What if Y=X2Y=X^2Y=X2? Cov(X,Y)=E[X⋅X2]−E[x]E[X2]Cov(X,Y) = E[X \\cdot X^2] - E[x]E[X^2]Cov(X,Y)=E[X⋅X2]−E[x]E[X2]. Since X and Y are (0,0) symmetric, Cov(X,Y)=0Cov(X,Y) = 0Cov(X,Y)=0 E(at)=0E(a_t) = 0E(at​)=0...","categories": ["Time Series"],
        "tags": ["White Noise","Stationary"],
        "url": "/time-series/ch1/",
        "teaser": null
      },{
        "title": "Data Cleaning",
        "excerpt":"Data Cleaning Why important? Key aspects of ML depend on clened data e.g.) Observation Labels: predicted Algorithms: estimation Features Model: assume this is acutal data representent Messy data generate garbage-in, garbage-out Reason: Lack of data, too much data, bad data How to deal with it? Duplicate or unnecessary data filter...","categories": ["IBM Machine Learning"],
        "tags": ["EDA","Week2"],
        "url": "/ibm/eda2/",
        "teaser": null
      },{
        "title": "Feature Engineering",
        "excerpt":"Variable Transformation We often assume normally distributed data. But often skewed -&gt; Data transformation solve this Log transformation Log transformation is a transformation that takes the natural log of the data. Useful for linear regression. Solve right skewed Polynomial Features higher order relationship from sklearn.preprocessing import PolynomialFeatures poly = PolynomialFeatures(degree=2)...","categories": ["IBM Machine Learning"],
        "tags": ["EDA","Week3"],
        "url": "/ibm/eda3/",
        "teaser": null
      },{
        "title": "Inferential Statistics and Hypothesis Testing",
        "excerpt":"Estimation and Inference Estimation is the application of an algorithm, to estimate parameter, e.g. mean, variance, etc. Inference involves putting an accuracy on the estimated value ? Statistical significancy Machine Learning and Statistical inference are similar. ML uses data to learn/infer qualities of a distirbution that generated the data, which...","categories": ["IBM Machine Learning"],
        "tags": ["EDA","Week4"],
        "url": "/ibm/eda4/",
        "teaser": null
      },{
        "title": "Logistic Regression",
        "excerpt":"Logistic Regression Sigmoid function y=11+e−xy = \\frac{1}{1 + e^{-x}}y=1+e−x1​ Apply Sigmoid function to the regression. Yβ(x)=11+e−βTx+ϵPr⁡(Yi=y∣Xi)=piy(1−pi)1−y=(eβ⋅Xi1+eβ⋅Xi)y(1−eβ⋅Xi1+eβ⋅Xi)1−y=eβ⋅Xi⋅y1+eβ⋅Xi\\begin{aligned} Y_{\\beta}(x) &amp; = \\frac{1}{1 + e^{-\\beta^T x+\\epsilon}}\\\\ \\\\ \\Pr(Y_{i}=y\\mid \\mathbf{X} _{i}) &amp; = {p_{i}}^{y}(1-p_{i})^{1-y}\\\\ &amp; = \\left({\\frac{e^{\\mathbf{\\beta}\\cdot \\mathbf{X} _{i}}}{1+e^{\\mathbf{\\beta} \\cdot \\mathbf{X} _{i}}}}\\right)^{y}\\left(1-{\\frac {e^{\\mathbf{\\beta}\\cdot \\mathbf{X} _{i}}}{1+e^{\\mathbf{\\beta}\\cdot \\mathbf{X} _{i}}}}\\right)^{1-y}\\\\ &amp; = {\\frac {e^{\\mathbf{\\beta}\\cdot \\mathbf{X} _{i}\\cdot y}}{1+e^{\\mathbf{\\beta}\\cdot...","categories": ["IBM Machine Learning"],
        "tags": ["Supervised","Classification","Week2","Logistic"],
        "url": "/ibm/cls1/",
        "teaser": null
      },{
        "title": "KNN and SVM",
        "excerpt":"K Nearest Neighbours KNN is predicting the unknown value of the point based on the values nearby. Decision Boundary KNN does not provide a correct K such that the right value of K depends on which error metric is most importnat. Elbow method is a cmmon way to find the...","categories": ["IBM Machine Learning"],
        "tags": ["Supervised","Classification","Week3","KNN","SVM"],
        "url": "/ibm/cls2/",
        "teaser": null
      },{
        "title": "Decision Tree",
        "excerpt":"Decision Tree Building a decision tree select a feature split the data into two groups. Split until the leaf node are pure (only one class remains) The maximum depth of the tree is reached A performance metric is achieved The decision tree uses a greedy search to find the best...","categories": ["IBM Machine Learning"],
        "tags": ["Supervised","Classification","Week3","Decision Tree"],
        "url": "/ibm/cls3/",
        "teaser": null
      },{
        "title": "Unbalanced Classes",
        "excerpt":"Unbalanced classes Classiferes are built to optimize accuracy and hence will often perform poorly on unbalanced classes/unrepresented classes. Downsampling deleted data from the training set to balance the classes. Upsampling duplicate data Resampling limit the value of larger class by sampling and increase smaller class by upsampling. cf) Weighting used...","categories": ["IBM Machine Learning"],
        "tags": ["Supervised","Classification","Week4","IBM"],
        "url": "/ibm/cls4/",
        "teaser": null
      },{
        "title": "Additional ML sources from IBM",
        "excerpt":"From IBM  Random Forest  https://www.ibm.com/cloud/learn/random-forest  ","categories": ["IBM Machine Learning"],
        "tags": ["Additional"],
        "url": "/ibm/add/",
        "teaser": null
      },{
        "title": "K-Means",
        "excerpt":"K-Means algorithm taking K random points as centroids. For each point, decide which centroid is closer, which forms clusters Move centroids to the mean of the clusters repeat 2-3 until centroids are not moving anymore K-Means++ It is smart initialization method. When adding one more point, no optimal is often...","categories": ["IBM Machine Learning"],
        "tags": ["Unsupervised","Week1","K-Means"],
        "url": "/ibm/uml1/",
        "teaser": null
      },{
        "title": "Clustering Algorithms",
        "excerpt":"Distant Metrics Manhattan Distance, L1 distance Another distance metric is the L1 distance or the Manhattan distance, and instead of squaring each term we are adding up the absolute value of each term. It will always be larger than the L2 distance, unless they lie on the same axis. We...","categories": ["IBM Machine Learning"],
        "tags": ["Unsupervised","Week2","Clustering"],
        "url": "/ibm/uml2/",
        "teaser": null
      },{
        "title": "Dimensionality Reduction",
        "excerpt":"Dimensionality Reduction Too many features leads to worse performance. Distance measures perform poorly and the indicent of outliers increases. Data can be represented in a lower dimensional space. Reduce dimensionality by selecting subset (feature elimination). Combine with linear and non-linear transformation. PCA Principal Component Analysis (PCA) is a dimensionality reduction...","categories": ["IBM Machine Learning"],
        "tags": ["Unsupervised","Week3","PCA"],
        "url": "/ibm/uml3/",
        "teaser": null
      },{
        "title": "Regression",
        "excerpt":"Box Cox Transformation The box cox transformation is a parametrized transformation that tries to get distributions “as close to a normal distribution as possible” It is defined as: boxcox(yi)=yiλ−1λ\\text{boxcox}(y_i) = \\frac{y_i^{\\lambda} - 1}{\\lambda}boxcox(yi​)=λyiλ​−1​ You can think of it as a generalization of the square root function: the square root function...","categories": ["IBM Machine Learning"],
        "tags": ["Supervised","Regression","IBM"],
        "url": "/ibm/reg/",
        "teaser": null
      },{
        "title": "Deep Learning",
        "excerpt":"Learning and Regularization Tecniques Dropout - This is a mechanism in which at each training iteration (batch) we randomly remove a subset of neurons. This prevents a neural network from relying too much on individual pathways, making it more robust. At test time the weight of the neuron is rescaled...","categories": ["IBM Machine Learning"],
        "tags": ["Deep Learning","IBM"],
        "url": "/ibm/dl/",
        "teaser": null
      },{
        "title": "RNN",
        "excerpt":"RNN Vanila RNN Context? use the notion of recurrence. Two outputs: Prediction and State (state of recurrent neural network) Learning goals Let r = input vector dimension, s = hidden state dimension, and t = output dimension. Us×r,Ws×s,Vt×ssi=f(UiTx+WiTsi+bi)U_{s\\times r}, W_{s\\times s}, V_{t\\times s}\\\\ s_i = f(U_i^T x + W_i^T s_i...","categories": ["IBM Machine Learning"],
        "tags": ["Deep Learning","IBM","Week4"],
        "url": "/ibm/dl4/",
        "teaser": null
      },{
        "title": "Using Machine Learning in Trading and Finance",
        "excerpt":"Endogenous and Exogenous trading rule Endo: long/short decision based on price data Exgo: uses other factors to make decision Are stock price and voluem data sufficient to enter a trade Can patterns of past predict? Extrapolate data pattern? E.g. Monetary policy Geopolitical Derivative market Excit rules Profit-exit Bid-ask spread, Brokerage...","categories": ["Quant"],
        "tags": ["Deep Learning","Finance","Trading","Coursera"],
        "url": "/quant/mltf/",
        "teaser": null
      },{
        "title": "Autoencoder and GAN",
        "excerpt":"Autoencoder Several applications: Dimensional reduction Preprocessing for classification Identifying essential elements of the input data and filtering out noise. Deal with some of these PCA limitations: PCA has learned features that are linear combinations of original features. VAE: Variational Autoencoder Data are assumed to be represented by a set of...","categories": ["IBM Machine Learning"],
        "tags": ["Deep Learning","IBM","Week5"],
        "url": "/ibm/dl5/",
        "teaser": null
      },{
        "title": "Reinforcement Learning",
        "excerpt":"Reinforcement Learning Agents interact with an Environment Choose from a set of available actions Actions impact the environment, which impacts agents via rewards Rewards are unknown and must be estimated by the agent Solutions represent a Policy by which Agents choose Actions in response to the State Agents typically maximize...","categories": ["IBM Machine Learning"],
        "tags": ["Deep Learning","IBM","Week6"],
        "url": "/ibm/dl6/",
        "teaser": null
      },{
        "title": "Machine Learning on Apple stock daily return",
        "excerpt":"import numpy as np import pandas as pd import os import matplotlib.pyplot as plt %matplotlib inline import seaborn as sns 1. Data aapl = pd.read_csv('./data/aapl.csv') X = aapl.drop(['Date','daily_ret'], axis=1) y = aapl['daily_ret'] aapl.columns Index(['Date', 'Close', 'Volume', 'Lo', 'HO', 'CO', 'support_low', 'support_open', 'support_high', 'std20', 'std120', 'std_open20', 'std_high20', 'std_intra', 'ma20', 'ma120', 'daily_ret',...","categories": ["IBM Machine Learning"],
        "tags": ["Machine Learning","IBM","Project"],
        "url": "/ibm/proj1/",
        "teaser": null
      }]
