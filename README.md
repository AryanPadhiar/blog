# An extension to normal bias-variance tradeoff and Double Descent Curve.

##  Overview

Avoiding overfitting is a well-known prescription in Data Science and is taught in all statistics courses and in classic machine learning classes. Traditionally, the bias-variance trade-off is depicted as a U-shaped risk curve (Figure 1). Both the train and test errors are high when the model complexity is low, by increasing the number of parameters the two errors decrease until the number of parameters reaches the coveted sweet point.

However,entering into the field of deep learning with experience in traditional machine learning, you may often ponder over this question: Since a typical deep neural network has so many parameters and training error can easily be perfect, it should surely suffer from substantial overfitting. How could it be ever generalized to out-of-sample data points?For example models containing millions of parameters (AlexNet in 2012 had 60M parameters to classify 1.2M examples, VGG-16 and VGG-19 both exceeded 100M parameters).In other words, how is it possible that models with millions of parameters are capable of reaching zero training error while performing extremely well on unseen data,and this leads us to double descent curve(Fig 2)
![i1](https://drive.google.com/uc?id=1dLJcy_bRRZk3OZGxeXIw3y6IvBmNBkv7)
![i2](https://drive.google.com/uc?id=1Ajs6Z9Q6WLmKOsUh7MT85KQgVMfFFOed)

One of the reasons this can be attributed to is The universal approximation theorem,we can always find a neural network to represent the target function with error under any desired threshold, but we need to pay the price — the network might grow super large and with larger function class we may be able to interpolate a simpler function that fits the data nicely.
>The Universal Approximation Theorem states that a feedforward network with: 1) a linear output layer, 2) at least one hidden layer containing a finite number of neurons and 3) some activation function can approximate any continuous functions on a compact subset of R^n to arbitrary accuracy. The theorem was first proved for sigmoid activation function (Cybenko, 1989). Later it was shown that the universal approximation property is not specific to the choice of activation (Hornik, 1991) but the multilayer feedforward architecture.

Interestingly, Belkin et al. (2019a)’s empirical study of test error provides some evidence that our bias-variance finding might not be unique to neural networks and might be found in other models such as decision trees. In subsequent work,1 Belkin et al. [2019](https://arxiv.org/abs/1812.11118);perform a theoretical analysis of student-teacher linear models (with random features), showing the double descent curve theoretically.

So,not only in deep learning but even in Machine learning it has been found that with sufficiently large no of features,the double descent curve can be obtained(though not as smooth as in fig 3)Now,we will further explore why is this happening? Let's break down the causes of variance,what things affect it? The upcoming slides would discuss about Bias-variance decomposition and Various experiments and their results.
![i3](https://drive.google.com/uc?id=1ClwUbRACao8cCJdzT6r5HIJmYhfmnT8p)





To understand this,I followed a research [paper](https://arxiv.org/abs/1810.08591) which tries breaking down Error in 3 parts:

1. Noise Error: It represents the inherent unpredictability or randomness in the data. It's the part of the error that arises from the variability in the data that cannot be captured by the model, no matter how complex it is. So,In the double descent curve, noise is a fixed component and does not change with the complexity of the model. It sets a lower bound on the error that cannot be reduced by improving the model.

2. Bias Error: It is the error that arises when the model is too simple to capture the underlying patterns in the data. It reflects the model's assumptions and the discrepancy between the true data distribution and the model's predictions. So,with respect to double descent curve, bias error is usually higher when the model has low complexity (e.g., few parameters), as simpler models tend to underfit. As model complexity increases, bias error generally decreases
3. Variance Error:Variance error refers to the error due to the model's sensitivity to small fluctuations in the training data. It reflects how much the model's predictions would change if it were trained on a different dataset,in context of double descent curve,Variance error initially increases as the model becomes more complex because the model starts to fit noise in the training data. However, as the model complexity continues to increase past a certain point (often beyond the point of interpolation), variance error may decrease. Now we need to see when and why this variance decreases beyond the second descent.

The maths behind:
$$
\text{Error} = E_{\text{noise}} + E_{\text{bias}} + E_{\text{variance}}
$$

Where:
Enoise = E(||y - (ybar)||^2)
Ebias=E(||h(θ,x)-y||^2)
Evariance=E(E(||h(θ,x)-E(h(θ,x))||^2))

The variance can also be broken into two parts:
1. Variance dur to sampling:We consider the variance of an ensemble of infinitely many predictors with different optimization randomness
2. Variance due to Optimisation: We consider the average (over training sets) variance over optimization randomness for a fixed training set

The total Variance can be expressed as:
Var(hθ(x)) =ES [VarO (hθ(x)|S)] + VarS (EO [hθ(x)|S])

Now,lets see a few results.
We find that variance due to sampling increases and then levels off, once sufficient width of NN is achieved. Also , we find that variance due to optimization decreases with width, causing the total variance to decrease with width. A body of recent work has provided evidence that using more number of nodes in a layer of ANN helps optimizers  optimize to global minima in  neural networks.Our observation of falling  variance on data set we used, shows that highly dense  (in terms of parameters)  ANN gives low variance, no matters what optimisation path we use.
You can check out various plots for results on different datasets like MNIST or CIFAR-10 [below](https://arxiv.org/abs/1810.08591).

Lets end then with the conclusion:
So,what so far we discussed leads us to the conclusion that in fact the larger networks can turn out to perform better and to support this we can understand it like the more the layers,the more the functional space and more the choices for an function to fit the relation.Another concept that can support this is of intrinsic dimension*
One intuition behind the measurement of intrinsic dimension is that, since the parameter space has such high dimensionality, it is probably not necessary to exploit all the dimensions to learn efficiently. If we only travel through a slice of objective landscape and still can learn a good solution, the complexity of the resulting model is likely lower than what it appears to be by parameter-counting. This is essentially what intrinsic dimension tries to assess.
It turns out many problems have much smaller intrinsic dimensions than the number of parameters. For example, on CIFAR 10 image classification, a fully-connected network with 650k+ parameters has only 9k intrinsic dimension and a convolutional network containing 62k parameters has an even lower intrinsic dimension of 2.9k.
And this is why you may even see pruning techniques and Sparse-models,even the Lottery Ticket hypothesis says that the model can perform with same accuracy or even better when it is pruned,that is some of the weights (connections) are redundant,but to eliminate them first we have to create the full network,then following a certain method of pruning we can reach the lottery ticket.
>Intrinsic dimension is a concept in machine learning and data analysis that refers to the minimum number of parameters or variables required to effectively capture the underlying structure of a dataset.

NOTE: The article's aim was not to deep dive into the mathematical details but give you a breif about the topic,what new explorations have been done on it,and all that you need to know is covered! Further references are mentioned below:


References for further reading:
[Are Deep Neural Networks Dramatically Overfitted?](https://lilianweng.github.io/posts/2019-03-14-overfit/)
[A Modern Take on the Bias-Variance Tradeoff in Neural Networks](https://arxiv.org/abs/1810.08591)
[Understanding-the-lottery-ticket-hypothesis](https://medium.com/@sayan112207/understanding-the-lottery-ticket-hypothesis-7d303f60616c)
[A Survey on Universal Approximation Theorems](https://arxiv.org/abs/2407.12895)








