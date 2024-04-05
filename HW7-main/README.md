#### HW7, Due 4/28/2023

## Preamble

In this homework, we will implement a feed-forward neural network model for predicting the value of a
drawn digit. We are using a subset of the MNIST dataset commonly used in machine learning research papers. A few
example of these handwritten-then-digitized digits from the dataset are shown below:

![image](https://user-images.githubusercontent.com/911695/230458301-d8cf7056-8a71-472a-abcf-24748fea706d.png)

Each digit is a 28 × 28 greyscale image with values ranging from 0 to 256. We represent an image as a row vector
x ∈ R^(1×784) where the image has been serialized into one long vector. Each digit has an associated class label from
0,1,2,...,9 corresponding to its value. We provide three dataset splits for this homework – a training set containing
XXX examples, a validation set containing XXXX, and our test set containing XXXX (no labels).

##### Cross-Entropy Loss for Multiclass Classification

Unlike the previous classification tasks we’ve examined, we have 10 different possible class labels here. How do
we measure error of our model? Let’s formalize this a little and say we have a dataset D = {xi, yi} for i=1..N with
yi ∈ {0, 1, 2, ..., 9}. Assume we have a model f(x; θ) parameterized by a set of parameters θ that predicts P(Y |X = x)
(a distribution over our labels given an input). Let’s refer to P(Y = c|X = x) predicted from this model as pc|x for
compactness. We can write this output as a categorical distribution:

![image](https://user-images.githubusercontent.com/911695/230458811-26daafe5-837a-4c9c-8519-ef54506f7960.png)


where I[condition] is the indicator function that is 1 if the condition is true and 0 otherwise. Using this, we can write
our negative log-likelihood of a single example as as:

![image](https://user-images.githubusercontent.com/911695/230465687-619cafbe-1a7f-4ea4-9533-f3ae92739395.png)

This loss function is also often referred to as a Cross-Entropy loss. In this homework, we will minimize this negative
log-likelihood by stochastic gradient descent. In the following, we will refer to this negative log-likelihood as

![image](https://user-images.githubusercontent.com/911695/230465745-66140eb8-8c44-4293-b720-8af7d0d34404.png)

Note that we write li as a function of θ because each pyi|xiis produced by our model f(xi; θ). Summing this over an
entire dataset (or a batch of examples as we will do later) of size N would yield the overall cross-entropy loss:

![image](https://user-images.githubusercontent.com/911695/230465803-b0bc351e-50c2-47fa-bf51-82bfac905e0a.png)

##### Implementing Backpropagation for Feed-forward Neural Network

In this homework, we’ll consider feed-forward neural networks composed of a sequence of linear layers xW1 + b1 and
non-linear activation functions g1(·). As such, a network with 3 of these layers stacked together can be written

![image](https://user-images.githubusercontent.com/911695/230465892-13ddab0b-feaa-4523-9da7-dc81fc43fdb4.png)

Note how that this is a series of nested functions, reflecting the sequential feed-forward nature of the computation.
To make our notation easier in the future, I want to give a name to the intermediate outputs at each stage so will
expand this to write:

![image](https://user-images.githubusercontent.com/911695/230465978-3830841e-a239-4d48-922e-4a80b51191cb.png)

where z’s are intermediate outputs from the linear layers and a’s are post-activation function outputs. In the case
of our MNIST experiments, z3 will have 10 dimensions – a score for each of the possible labels. Finally, the output
vector z3 is not yet a probability distribution so we apply the softmax function:

![image](https://user-images.githubusercontent.com/911695/230466053-4561a537-73d2-4716-854c-2b5a4752ec76.png)

and let p·|x be the vector of these predicted probability values.

##### Gradient Descent for Neural Networks

Considering this simple 3-layer neural network, there are quite a few
parameters spread out through the function – weight matrices W3, W2, W1 and biases vectors b3, b2, b1. Suppose we
would like to find parameters that minimize our loss L that measures our error in the network’s prediction.

How can we update the weights to reduce this error? Let’s use gradient descent and start by writing out the chain
rule for the gradient of each of these. I’ll work backwards from W3 to W1 to expose some structure here.

![image](https://user-images.githubusercontent.com/911695/230466246-962bbdc5-08f4-4619-9893-ede8d8e45450.png)

As I’ve highlighted in color above, we end up reusing the same intermediate terms over and over as we compute
derivatives for weights further and further from the output in our network.1 As discussed in class, this suggests
the straight-forward backpropagation algorithm for computing these efficiently. Specifically, we will compute these
intermediate colored terms starting from the output and working backwards.

##### Forward-Backward Pass in Backpropagation

One convenient way to implement backpropagation is to consider
each layer (or operation) f as having a forward pass that computes the function output normally as

![image](https://user-images.githubusercontent.com/911695/230466610-911d57f2-fbae-4d22-8832-cd41f232226a.png)

and a backward pass that takes in the gradient up to this point in our backward pass and then outputs the gradient
of the loss with respect to its input:

![image](https://user-images.githubusercontent.com/911695/230466703-4a811460-a495-46ab-9040-23b095bbd9e9.png)

The backward operator will also compute any gradients with respect to parameters of f and store them to be used in
a gradient descent update step after the backwards pass. The starter code implements this sort of framework.
See the snippet on the following page that defines a neural network like the one we’ve described here, except it
allows for a configurable number of linear layers. Please read the comments and code below before continuing reading
this document. To give concrete examples of the forward-backward steps for an operator, consider the Sigmoid (aka
the logistic) activation function below:

![image](https://user-images.githubusercontent.com/911695/230466803-0e1e025a-5f11-41bc-bf7e-a50f9155dfa3.png)

The implementation for forward and backward for the Sigmoid is below – in forward it computes above, in backward
it computes and returns

![image](https://user-images.githubusercontent.com/911695/230466919-886717e8-62ea-4375-9db8-f0457bce6977.png)

It has no parameters so does nothing during the "step" function.

![image](https://user-images.githubusercontent.com/911695/230466997-570ea8b2-9f6d-4188-a823-dafd06f6d529.png)

##### Operating on Batches. 

The network described in the equations earlier in this section is operating on a single input
at a time. In practice, we will want to operate on sets of n examples at once such that the layer actually computes
Z = XW + b for X ∈ R n×input_dim and Z ∈ R n×output_dim – call this a batched operation. It is straightforward to
change the forward pass to operate on these all at once. For example, a linear layer can be rewritten as Z = XW + b
where the +b is a broadcasted addition – this is already done in the code above. On the backward pass, we simply
need to aggregate the gradient of the loss of each data point with respect to our parameters. For example,

![image](https://user-images.githubusercontent.com/911695/231237693-c5d3240a-eafb-4aa8-aaaf-42a7050dd1b8.png)

where Li is the loss of the i’th datapoint and L is the overall loss.

##### Deriving the Backward Pass for a Linear Layer. 

In this homework, we’ll implement the backward pass of a linear layer. To do so, we’ll need to be able to compute dZ/db, dZ/dW, and dZ/dX. For each, we’ll start by considering the problem for a single training example x (i.e. a single row of X) and then generalize to the batch setting. In this
single-example setting, z = xW + b such that z, b ∈ R^1×c, x ∈ R^1×d, and W ∈ R^d×c. Once we solve this case, extending to the batch setting just requires summing over the gradient terms for each example.

**dZ/db**. Considering just the i’th element of z, we can write zi = x·wi +bi where wi is the i’th column of W. From
this equation, it is straightforward to observe that element bi only effects the corresponding output zi such that

![image](https://user-images.githubusercontent.com/911695/231238546-c46379c3-3544-4f17-ac5f-a52272618c48.png)

This suggests that the matrix dz/db is an identity matrix I of dimension c × c. Applying chain rule and summing over all our datapoints, we see dL/db can be computed as a sum of the rows of dL/dZ:

![image](https://user-images.githubusercontent.com/911695/231238741-fc15b8a7-495f-41de-a0df-b55011b59905.png) *Equation 1.1*

**dZ/dW.** Following the same process of reasoning from the single-example case, we can again write the i’th element
of z as zi = xw·,i + bi where w·,i is the i’th column of W. When considering the derivative of zi with respect to the
columns of W, we see that it is just x for w·,i and 0 for other columns as they don’t contribute to zi – that is to say:

![image](https://user-images.githubusercontent.com/911695/231239436-d3b6c2d8-d26f-4c5c-9ce7-b1cc32ee52f3.png)

Considering the loss gradient δL/δw·,i for a single example, we can write:

![image](https://user-images.githubusercontent.com/911695/231239523-c5c98537-16f4-4634-96d7-7367e958f952.png)

That is to say, each column i of δL/δW is the input x scaled by the loss gradient of zi. As such, we can compute the gradient for the entire W as the product:

![image](https://user-images.githubusercontent.com/911695/231239634-7f8ac0a0-2032-4309-9299-7232b87aad05.png)

Notice that x^T is d × 1 and δL/δz is 1 × c – resulting in a d × c gradient that matches the dimension of W.

Now let’s consider if we have multiple datapoints x1, ...xn as the matrix X and likewise multiple activation vectors
z1, ..., zn as the matrix Z. As our loss simply sums each datapoint’s loss, the gradient also decomposes into a sum of δL/δzi terms.

![image](https://user-images.githubusercontent.com/911695/231240012-1325c82a-9975-4920-99dd-56063debcfed.png)

We can write this even more compactly as:

![image](https://user-images.githubusercontent.com/911695/231240090-d4a58f5e-30cb-4e2f-b4f5-02b3feefb418.png)*Equation 1.2*

**dZ/dX.** This follows a very similar path as dZ/dW. We again consider the i’th element of z as zi = xw·,i + bi where
w·,i is the i’th column of W. Taking the derivative with respect to x it is clear that for zi the result will be w·,i.

![image](https://user-images.githubusercontent.com/911695/231240210-71dd4709-b681-4929-9e6d-8d013b10bfe3.png)

This suggests that the rows of dZ/dx are simply the columns of W such that dZ/dx = W^T and we can write

![image](https://user-images.githubusercontent.com/911695/231240322-20a0f5f2-1e04-4e61-b90b-ca442ad17c94.png)

Moving to the multiple example setting, the above expression gives each row of dL/dX and the entire matrix can be
computed efficiently as

![image](https://user-images.githubusercontent.com/911695/231240517-9d0af105-fe82-46be-b2d6-38d3b21de2c4.png)*Equation 1.3*



## Questions

**Q0 [2 points] Complete this before changing any files.**

1. Download a zip of the HW7 repository. (https://github.com/andrewboes/HW7)
2. Upload it to a folder called HW7 in your Artificial-Inteligence-Spring-2023 repository (don't branch first).
3. Create a branch off of your Artificial-Inteligence-Spring-2023 repository on github called “HW7”, do all of your work for this homework in the HW6 branch.
4. For all write ups and plots, add these to either a .doc file or a .pdf file in your repository called (your name).pdf.

---

**Q1 Implementing the Backward Pass for a Linear Layer [20pt]**

Implement the backward pass function of the linear layer in the skeleton code. The function takes in the matrix dL/dZ as the variable
grad and you must compute dL/dW, dL/db, and dL/dX. The first two are stored as self.grad_weights and self.grad_bias and the third is returned. The expressions for these can be found above in Equation 1.1 (dL/db), Equation 1.2 (dL/dW), and Equation 1.3 (dL/dX).

---

Once you’ve completed the above task, running the skeleton code should load the digit data and train a 2-layer
neural network with hidden dimension of 16 and Sigmoid activations. This model is trained on the training set and
evaluated once per epoch on the validation data. After training, it will produce a plot of your results that should look
like the one below. This curve plots training and validation loss (cross-entropy in this case) over training iterations (in
red and measured on the left vertical axis). It also plots training and validation accuracy (in blue and measures on the
right vertical axis). As you can see, this model achieves between 80% and 90% accuracy on the validation set.

![image](https://user-images.githubusercontent.com/911695/231242967-c8d04ab5-dc13-4d10-a084-406aa4d8372b.png)

##### Analyzing Hyperparmeter Choices

Neural networks have many hyperparameters. These range from architectural choices (How many layers? How wide
should each layer be? What activation function should be used? ) to optimization parameters (What batch size for
stochastic gradient descent? What step size (aka learning rate)? How many epochs should I train? ). This section has
you modify many of these to examine their effect. The default parameters are below for easy reference.

![image](https://user-images.githubusercontent.com/911695/231243118-84d76d97-1075-4732-8dd5-3a9d89550081.png)

##### Optimization Parameters

Optimization parameters in Stochastic Gradient Descent are very inter-related. Large
batch sizes mean less noisy estimates of the gradient, so larger step sizes could be used. But larger batch sizes
also mean fewer gradient updates per epoch, so we might need to increase the max epochs. Getting a good set of
parameters that work well can be tricky and requires checking the validation set performance. Further, these “good
parameters” will vary model-to-model.

---

**Q2 Learning Rate [5pts]**

The learning rate (or step size) in stochastic gradient descent controls how
large of a step in the direction of the loss gradient we take our parameters at each iteration. The batch size
determines how many data points we use to estimate the gradient. Modify the hyperparameters to run the
following experiments:

1. Step size of 0.0001 (leave default values for other hyperparameters)
2. Step size of 5 (leave default values for other hyperparameters)
3. Step size of 10 (leave default values for other hyperparameters)

Include these plot in your report and answer the following questions:

a.) Compare and contrast the learning curves with your curve using the default parameters. What do you
observe in terms of smoothness, shape, and what performance they reach?

b.) For (a), what would you expect to happen if the max epochs were increased?

---

**Q3 Randomness in training [5pts]**

Using the default hyperparameters, set the random seed to 5 different values and report the validation accuracies you observe after training. What impact does this randomness have on the certainty of your conclusions in the previous questions?

---

**Q4 Activation functions [5pts]**

Modify the hyperparameters to run the following experiments:
1. 5-layer with Sigmoid Activation (leave default values for other hyperparameters)
2. 5-layer with Sigmoid Activation with 0.1 step size (leave default values for other hyperparameters)
3. 5-layer with ReLU Activation (leave default values for other hyperparameters)

Include these plot in your report and answer the following questions:
a.) Compare and contrast the learning curves you observe and the curve for the default parameters in terms
of smoothness, shape, and what performance they reach. Do you notice any differences in the
relationship between the train and validation curves in each plot?

b.) If you observed increasing the learning rate in (2) improves over (1), why might that be?

c.) If (3) outperformed (1), why might that be? Consider the derivative of the sigmoid and ReLU functions.

---

**Q5. [2pts] Do not complete until you are mostly finsihed with the assignment.**

Answer the following in (your name).pdf:

1. Approximately how many hours did you spend on this assignment?
2. Would you rate it as easy, moderate, or difficult?
3. Did you work on it mostly alone or did you discuss the problems with others?
4. How deeply do you feel you understand the material it covers (0%–100%)?
5. Any other comments?

Create a pull request from the HW7 branch to main and assign me as the reviewer.

---

**B1 (Bonus question 1) [10pts]**

Add a method to display results in the validation set that are classified incorrectly. The method should take an integer to only show the first n incorrect entries (0 would display all). You may want to add a global variable (verbose, logging, etc) to turn it on or off.

---

**B2 Ensemble [10pts]**

From Q3 and B1 you might notice that some runs correctly classify items in the validation set and some don't. We can use this to our advantage to increase our validation accuracy. After you've found the hyperparameters you like for your neural network, rerun your code with 10 (or more) different random seeds and save the weights from each run. Then have each run "vote" on the validation set and report what your new validation accuracy is. 

---

**B3 Augment dataset [15pts]**

Add three or more image transformations on the training set to augment the included data. Transformations include vertical and horizontal shifting, rotations, reducing and increasing sharpness, and adding gaussian noise. Include in your report any changes in validation accuracy and training time you notice.

---

**B4 Kaggle Submission [5-20pts]**

Beat 80% for 5 pts
Beat 93% for 15 pts
Beat 95% for 20 pts

See the kaggle competition here: https://www.kaggle.com/t/f3c51bffeffe45518f55faf76b6c0bfc

---

**B5 Best Kaggle Submission [10pts]**
This is for only one student. Whoever gets the highest score in the Kaggle competition from B4 gets 10 bonus points.
