## HW6 Due March 25th, 2023

#### Part 0. (2 points) Complete this before changing any files.

1. Download a zip of the HW6 repository. (https://github.com/andrewboes/HW6)
2. Upload it to a folder called HW6 in your Artificial-Inteligence-Spring-2023 repository (don't branch first).
3. Create a branch off of your Artificial-Inteligence-Spring-2023 repository on github called “HW6”, do all of your work for this homework in the HW6 branch.

#### Part 1. kNN

In this homework, we’ll implement our first machine learning algorithm of the course – k Nearest Neighbors. We are considering a binary classification problem where the goal is to classify whether a person has an annual income more or less than $50,000 given census information. As no validation split is provided, you’ll need to perform cross-validation to fit good hyperparameters.

We’ve done the data processing for you for this assignment; however, getting familiar with the data before applying any algorithm is a very important part of applying machine learning in practice. Quirks of the dataset could significantly impact your model’s performance or how you interpret the outcome of your experiments. For instance, if I have an algorithm that achieved 70% accuracy on this task – how good or bad is that? We’ll see! We’ve split the data into two subsets – a training set with 7,000 labelled rows, and a test set with 1,000 unlabelled rows. These are “train.csv” and “test_pub.csv”.

Both files will come with a header row, so you can see which column belongs to which feature. Below you will find a table listing the attributes available in the dataset. We note that categorizing some of these attributes into two or a few categories is reductive (e.g. only 14 occupations) or might reinforce a particular set of social norms (e.g. categorizing sex or race in particular ways). For this homework, we reproduced this dataset from its source without modifying these attributes; however, it is useful to consider these issues as machine learning practitioners.

- **attribute name:** *type.* list of values
- **id:** *numerical.* Unique for each point. Don’t use this as a feature (it will hurt, badly).
- **age:** *numerical.*
- **workclass:** *categorical.* Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay
- **education-num:** *ordinal.* 1:Preschool, 2:1st-4th, 3:5th-6th, 4:7th-8th, 5:9th, 6:10th, 7:11th, 8:12th, 9:HS-grad, 10:Some-college, 11:Assoc-voc, 12:Assoc-acdm, 13:Bachelors, 14:Masters, 15:Prof-school, 16:Doctorate
- **marital-status:** *categorical.* Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
- **occupation:** *categorical.* Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlerscleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, ArmedForces
- **relationship:** *categorical.* Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
- **race:** *categorical.* White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
- **sex:** *categorical.* 0:Male, 1:Female
- **capital-gain:** *numerical.*
- **capital-loss:** *numerical.*
- **hours-per-week:** *numerical.*
- **native-country:** *categorical.* United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(GuamUSVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua,
Scotland, Thailand, Yugoslavia, El-Salvador, Trinada&Tobago, Peru, Hong, Holand-Netherlands
- **income:** *ordinal.* 0: <=50K, 1: >50K This is the class label. Don’t use this as a feature in training.

Our dataset has three types of attributes – numerical, ordinal, and nominal. Numerical attributes represent continuous numbers (e.g. hours-per-week worked). Ordinal attributes are a discrete set with a natural ordering, for instance different levels of education. Nominal attributes are also discrete sets of possible values; however, there is no clear ordering between them (e.g. native-country). These different attribute types require different preprocessing. As discussed in class, numerical fields have been normalized.
For nominal variables like workclass, marital-status, occupation, relationship, race, and native-country, we’ve transformed these into one column for each possible value with either a 0 or a 1. For example, the first instance in the training set reads: [0, 0.136, 0.533, 0.0, 0.659, 0.397, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,0] where all the zeros and ones correspond to these binarized variables. The following questions guide you through exploring the dataset and help you understand some of the steps we took when preprocessing the data.

#### Question 1 (5 points): 

Encodings and Distance To represent nominal attributes, we apply a one-hot encoding technique – transforming each possible value into its own binary attribute. For example, if we have an attribute workclass with three possible values Private, State-gov, Never-worked – we would binarize the workclass attribute as shown below (each row is a single example data point):

![image](https://user-images.githubusercontent.com/911695/225389780-10a34311-051e-425e-aa50-4708f1c15773.png)

![image](https://user-images.githubusercontent.com/911695/225389838-98484429-b03f-4e4e-a238-c79d79d6e031.png)

A common naive preprocessing is to treat all categoric variables as ordinal – assigning increasing integers to each possible value. For example, such an encoding would say 1=Private, 2=State-gov, and 3=Neverworked. Contrast these two encodings. Focus on how each choice affects Euclidean distance in kNN.

### Answer 
when using knn it is important to change the values to their own booleans because when we are calculating euclidean distance if we looked at the distanse between 1 and 3 we would get Sqrt((1-3)^2) = 2  and in the other method we would get Sqrt((0-1)^2+(0-0)^2+(0-1)^2) = Sqrt( 1+1) = 1.4142 this shows us that the true distance between the variables is inccorect in the first example because that establishes a difference between if the attributes are 1,2 or 1,3 but when you split them up the distance between all of the values will be 1.4142. This gets much more drastic the more attributes you have also

#### Question 2 (5 points)

What percent of the training data has an income >50k? Explain how this might affect your model and how you interpret the results. For instance, would you say a model that achieved 70% accuracy is a good or poor model? How many dimensions does each data point have (ignoring the id attribute and class label)? [Hint: check the data, one-hot encodings increased dimensionality]

### Answer

24.528% of the data in the dataset has an income of over 50,000 this means that you have a pretty good set of datapoints so you should be able to have pretty good accuracy so 70 would not be very good (famouse last words).

#### Question 3 (5 points)

Distances and vector norms are closely related concepts. For instance, an L2 norm of a vector x (defined below) can be intepretted as the Euclidean distance between x and the zero vector.

![image](https://user-images.githubusercontent.com/911695/225389533-01b8e367-3631-4952-a2fa-5c80ad58ef13.png)


Given a new vector z, show that the Euclidean distance between x and z can be written as an L2 norm. [kNN implementation note for later, you can compute norms efficiently with numpy using np.linalg.norm]

#### Computing efficiently

In kNN, we need to compute distance between every training example xi and a new point z in order to make a
prediction. As you showed in the previous question, computing this for one xi can be done by applying an arithmetic
operation between xi and z, then taking a norm. In numpy, arithmetic operations between matrices and vectors are
sometimes defined by “broadcasting”, even if standard linear algebra doesn’t allow for them. For example, given a
matrix X of size n × d and a vector z of size d, numpy will happily compute Y = X − z such that Y is the result of
the vector z being subtracted from each row of X. Combining this with your answer from the previous question
can make computing distances to every training point quite efficient and easy to code.


#### Question 4 (20 points)

Okay, it is time to get our hands dirty. Let’s write some code! Implement k-Nearest Neighbors using Euclidean distance by completing the skeleton code provided in knn.py. Specifically, you’ll need to finish:

get_nearest_neighbors(example_set, query, k): Given a n × d example_set matrix where each row represents one example point, return the indices of the k nearest examples to the query point. This is where the bulk of the computation will happen in your algorithm so you are strongly encouraged to review the paragraph above to get hints
for how to do it efficiently in numpy.

knn_classify_point(examples_X, examples_y, query, k): Given a n × d example_set matrix where each row represents one example point and a n × 1 column-vector with these points’ corresponding labels, return the prediction of a kNN classifier for the query point. Should use the previous function.

The code to load the data, predict for each point in a matrix of queries, and compute accuracy is already
provided towards the end of the file. You’ll want to read over these carefully.

#### Question 5 (20 points)

Next we’ll be implementing k-fold cross validation by finishing the following function in knn.py:

cross_validation(train_X, train_y, num_folds, k): Given a n × d matrix of training examples and a n × 1 column-vector of their corresponding labels, perform K-fold cross validation with num_folds folds for a k-NN classifier. To do so, split the data in num_folds equally sized chunks then evaluate performance for each chunk while considering the other chunks as the training set. Return the average and variance of the accuracies you observe.

For simplicity, you can assume num_folds evenly divides the number of training examples. You may find the numpy split and vstack functions useful.

#### Question 6 (20 points)

To search for the best hyperpameters, run 4-fold cross-validation to estimate our accuracy. For each k in 1,3,5,7,9,99,999,and 8000, report:

- accuracy on the training set when using the entire training set for kNN (call this training accuracy),

- the mean and variance of the 4-fold cross validation accuracies (call this validation accuracy).

Skeleton code for this is present in main() and labeled as Q9 Hyperparmeter Search. Finish this code to generate these values – should likely make use of predict, accuracy, and cross_validation.

Questions: 
- What is the best number of neighbors (k) you observe? 
- When k = 1, is training error 0%? Why or why not? 
- What trends (train and cross-valdiation accuracy rate) do you observe with increasing k? 
- How do they relate to underfitting and overfitting?

#### Bonus Question (10 points)

Code at the end of main() outputs predictions for the test set to test_predicted.csv. Decide on hyperparameters and add your submission set of predictions to your repository. Predictions are formatted as a two-column CSV as below:

id,income

0,0

1,0

2,1

.

.

You must beat 77% to get 5 points and 83% to get 10 points.

#### Part 3. (2 point) Do not complete until you are mostly finsihed with the assignment.

Add a text file to the HW6 folder called “feedback.txt” and answer the following:

1. Approximately how many hours did you spend on this assignment?
2. Would you rate it as easy, moderate, or difficult?
3. Did you work on it mostly alone or did you discuss the problems with others?
4. How deeply do you feel you understand the material it covers (0%–100%)?
5. Any other comments?

Create a pull request from the HW6 branch to main and assign me as the reviewer.
