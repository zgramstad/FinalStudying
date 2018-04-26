# Study Guide

# Lecture Topics (for the second half of the semester)
1. Trees
2. Ensembles
	3. Gradient Boosting
4. Graphical models (HMMs, )
5. K-means clustering
6. Gaussian Mixture Models
7. Principal Component Analysis and Kernel PCA
8. The Reenforcement Learning Problem and relationship to Markov Decision Processes
9. Temporal Difference Learning

# P1 Topics

1. forward backward pass of HMMs
2. EM
3. simple neural network stuff
4. SVMs
5. k-means
6. Simple PCA: compute small covariance matrix and find the eigens

# P2 Topics 
1. graphical models
2. reinforcement learning



## What is EM (expectation maximization)?
Lets review EM. In EM, you randomly initialize your model parameters, then you alternate between (E) assigning values to hidden variables, based on parameters and (M) computing parameters based on fully observed data.

**E-Step:** Coming up with values to hidden variables, based on parameters. If you work out the math of chosing the best values for the class variable based on the features of a given piece of data in your data set, it comes out to "for each data-point, chose the centroid that it is closest to, by euclidean distance, and assign that centroid's label." The proof of this is within your grasp! See lecture.

**M-Step:** Coming up with parameters, based on full assignments. If you work out the math of chosing the best parameter values based on the features of a given piece of data in your data set, it comes out to "take the mean of all the data-points that were labeled as c."

So what? Well this gives you an idea of the qualities of k-means. Like EM, it is provably going to find a local optimum. Like EM, it is not necessarily going to find a global optimum. It turns out those random initial values do matter.

from [Stanford CS221](http://stanford.edu/~cpiech/cs221/handouts/kmeans.html)

## HMMs

For when you have a sequence of observations and a sequence of hidden states that generated those observaions, and you want determine the most likely set of underlying hidden states.

* $e_t$: an obsevation
* $x_t$: a hidden state
* $\pi$: the initial state probability matrix (P(X_0 = X))
* $a$: hidden state transition probabilities (probability from going from one hidden state to another)
* $b$: emission probability (probability of emitting observation given hidden state)


HMM defined by set S, O and probabilites params $[\pi, a, b] = \lambda$

1. Sample $ X\_0 $ from $ P(X\_0) $ which is $\pi$
2. Repeat for t = 1...T
	1. Sample x\_t from P(X\_t | X\_{t-1}) which is a
	2. Sample e\_t from P(e\_t | X\_t) which b


You observe a sequence of emissions. What is the probability of the hidden state of the last state given the previous emissions ($P(X_5 | h h h t t)$)?

You know $S,O, \lambda(\pi,a,b)$

### HMM Forward Pass (filtering)
To find the probabillity of the most T-th hidden state given 1,...,T emission states.

$\alpha_t(i) := P(e_1,...,e_t, X_t = s_i)$ (the probability of observing sequence of emissions and $X_t$ being $s_i \in S$

1. $\alpha_0(i) = \pi_i$
This is basically saying that the 0-th state is the initial state probability.

2. $\alpha_{t+1}(j) = b_j(e_{t+1}) \sum^n_{i=1}\alpha_t(i)a_{i,j}$
This is basically saying that after the 0-th state, to find the probability that X is in the hidden state $j$ at time $t+1$, you:
	1. look at the joint probability of emmitting state $e_{t+1}$ given the hidden state $j$
	2. Multiply the emission probability $b_j(e_{t+1})$ by the sum of all probabilities of the possible paths to get to this hidden state given all the possible previous states. 

### HMM backward Pass (smoothing)
This is about finding the probability of a hidden state $X_k$ $(k < t)$ given all the emissions from $1$ to $t$.

This probability is proportional to $P(e_{k+1},...,e_t | X_k)P(X_k|e_1,...,e_t)$

$\beta_k(i) := P(e_{k+1},...,e_t | X_k = s_i)$ the probability of the next t - k + 1 emissions given the current hidden state

1. $\beta_T(i) = 1$ the probability of no more emissions given the end state
2. $\beta_k(i) = \sum^n_{j=1}a_{ij}b_j(e_{t+1})\beta_{k+1}(j)$ the probability of the transition from the next state to the current state multiplied by the emission of the next state multiplied by the $\beta$ value of the next state, summed over all $j$ next states.

### Learning the parameters of an HMM

Case 1: When you are given the observation and the hidden state sequences:

* $\pi$: Count $x_0$s
* $a_{ij}$: $\frac{\text{count of } si \rightarrow sj}{\text{count of } si}$
* $b_j(O_k)$: $\frac{\text{count of } O_k \text{ associated with }s_j}{\text{count of } s_j}$ where $O_k$ is an emmission pair ($x_t, e_t$)

Case 2: When you are given only the observation state sequence:

Find a $\lambda$ that maximizes $P(e_1,...e_t | \lambda)$. I.e. find the set of $(\pi,a,b)$ that maximizes the probability of the provided sequence. Use the Baum-Welch EM (expected maximization) algorithm.

$\zeta_t(i,j) = P(X_t = s_i, X_{t+1} = s_j | e_1,...,e_T, \lambda) = 
\frac{\alpha_t(i)a_{ij}b_j(e_{t+1})\beta_{t+1}(j)}{\sum^n_{i=1}\sum^n_{j=1}\alpha_t(i)a_{ij}b_j(e_{t+1})\beta_{t+1}(j)}$

$\nu_t(i) := P(X_t = s_i | e_1,...,e_T, \lambda) = \sum^n_{j=1}\zeta_t(i,j)$ The sum over all possible $X_{t_1}$ states.

The expected number of times $s_i$ is visited in total then is:
$\sum_{t=1}^{T-1}\nu_t(i)$

The expected number of transitions from $s_i$ to $s_j$ is:
$\sum^{T-1}_{t=1}\zeta_t(i,j)$

### Baum-Welch Algorithm
1. Guess $\lambda_0 = [\pi_0, a_0, b_0]$
2. Calculate $\alpha$, $\beta$ from $\lambda$ 
3. Reestimate $\lambda$ from $\alpha$, $\beta$ 
4. Repeat 2 and 3 until convergence

## PCA

### [PCA explanation online that Devika referenced](http://sebastianraschka.com/Articles/2014_kernel_pca.html)

**Principal Components:** new axes of the dataset that maximize the variance along those axes (i.e. the eigenvectors of the covariance matrix). 

Its main purpose is to reduce the dimensions of the dataset with minimal loss of information. The dataset is project onto a new subspace of lower dimension.

There are 6 steps to PCA:

1. Compute the covariance matrix of the original d-dimensional datset $X$.
2. Compute the eigenvectors and eigenvalues of the dataset.
3. Sort the eigenvalues by decreasing order.
4. Choose the $k$ eigenvectors that correspond to the $k$ largest eigenvalues where $k is the number of dimensions for the new feature subspace.
5. Construct the projection matrix $W$ of the $k$ selected eigenvectors.
6. Transform the original dataset $X$ to obtain the k-dimensional feature subspace $Y$: $$Y= W^T \cdot X$$


### [Computing the covariance matrix of the original d-dimensional datset $X$](https://math.stackexchange.com/questions/710214/how-to-construct-a-covariance-matrix-from-a-2x2-data-set?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)

$\bar x$ = means along the row dimension of $X$ (squash the columns).

$\bar y$ = means along the column dimension of $X$ (squash the rows).

For a 2x2 $X$, the variance-covariance matrix has the following structure:

$$\begin{bmatrix}
var(x) & cov(x,y)\\
cov(x,y)& var(y)
\end{bmatrix}
$$

where $var(x)=\frac{1}{n−1}∑(x_i−\bar x)^2$ and $cov(x,y)=\frac{1}{n−1}∑(x_i−\bar x)(y_i−\bar y)$.

For:

$$ X|y = \begin{bmatrix}
3 & |7\\
2& |2
\end{bmatrix}
$$

* $\bar x=\frac{(3+2)}{2}=\frac{5}{2}$
* $\bar y=\frac{(7+4)}{2}=\frac{11}{2}$
* $var(x)=(3−\frac{5}{2})^2+(2−\frac{5}{2})^2$
* $var(y)=(7−\frac{11}{2})^2+(4−\frac{11}{2})^2$
* $cov(x,y)=(3−\frac{5}{2})(7−\frac{11}{2})+(2−\frac{5}{2})(4−\frac{11}{2})$

### [Computing the eigenvalues and eigenvectors of a 2x2 matrix](http://lpsa.swarthmore.edu/MtrxVibe/EigMat/MatrixEigen.html)

[Of a 3x3](http://wwwf.imperial.ac.uk/metric/metric_public/matrices/eigenvalues_and_eigenvectors/eigenvalues2.html)

### [The Kernel Trick and PCA](http://sebastianraschka.com/Articles/2014_kernel_pca.html)
It is possible to use a kernel trick (and preferable if the data is not easily linearly separable).

Commonly used is the [RBF (gaussian radial basis function) Kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel): $$K(x,x') = exp(\frac{-||x-x'||^2}{2\sigma^2})$$ where $\sigma$ is a free parameter.

## K-means Clustering
K-means clustering is a form of unsupervised learning (used to explore data structure, outliers and better understand features). K-Means is really just the EM (Expectation Maximization) algorithm applied to a particular naive bayes model.

The goal is to partition the data into K clusters such that the inter-point distances within each cluster are minimized.

### Objective Cost Function

$$ J = \sum^m_{i=1}\sum^K_{k=1}z_k^{(i)}||x^{(i)} - \mu_k||^2$$

* $\mu_k$: the "centroid" or D-dimensional center of cluster $k$
* $z_k^{(i)}$: $1$ if $x^{(i)}$ in cluster $k$, $0$ otherwise

$J$ then is the sum of all of the euclidean distances between the centroids and the datapoints that are assigned to their respective centroids.

### High level K-means clustering algorithm (from [this](https://home.deib.polimi.it/matteucc/Clustering/tutorial_html/kmeans.html) great, simple explanation):

1. Place K points (randomly) into the space represented by the objects that are being clustered. These points represent initial group centroids.
2. **E-Step:** Assign each object to the group that has the closest centroid. 
3. **M-step:** When all objects have been assigned, recalculate the positions of the K centroids.
4. Repeat Steps 2 and 3 until the centroids no longer move. This produces a separation of the objects into groups from which the metric to be minimized can be calculated.

### More in-depth description of K-means algorithm

**E-Step: assigning points to clusters**

* $J$ is linear in $z$ **(why does this matter?)**
* each $x^{(i)}$ is independent **(is this an assumption or a fact?)**
* Calculate the addional cost for the value of $k$ for each point $x^{(i)}$
* Select the value of k that has the smallest value for J
$$z_k^{(i)} = 1 \text{ if } k = \text{argmin}_j ||x^{(i)} - \mu_j||^2; 0 \text{ otherwise}$$

**M-Step: relocate the centroids**

* $J$ is quadratic in $\mu$ **(again, why does this matter?)**
* each $x^{(i)}$ is independent **(again, is this an assumption or a fact?)**
* 

### Important discussion about K-means clustering

## Gaussian Mixture Models
### [In-depth explanation of GMMs Devika referenced](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html)

## Additional Suggestions
* Run through Value iteration (Slide 57 for class 4/18) (mdp for RL.pptx)


## Questions
Is this a typo? Shouldn't it be $P(X_k | e_1...e_k)$ for the second part?
![yolo](images/Screen Shot 2018-04-25 at 10.46.37 AM.png)

"The expected number of transitions from $s_i$ to $s_j$ is:
$\sum^{T-1}_{t=1}\zeta_t(i,j)$" Why is it the expected number?



