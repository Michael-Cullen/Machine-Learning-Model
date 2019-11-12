# Machine-Learning-Model
Using Scikit-learn, Pandas and Numpy to create a model based from Nuclear plant data


Algorithms for Data Mining
Section 1: Data import, summary, preprocessing and visualisation

Importing Data

The nuclear plants dataset has been provided in CSV format, meaning each value is separated by a comma, with the feature headings defined on the first line then followed by the data. This is a well supported format that can easily be read into a python IDE using the pandas data analysis library. 
Pandas uses Data Frames as a two-dimensional data structure that are mutable in size and support columns of different data types. By storing the dataset within a dataframe it is simple to perform mathematical operations such as calculating the mean, standard deviation, min and max values. 

Data summary

Reading the number of columns and rows within the data frame shows how many features, alongside the number of data entries (records), the dataset contains. There are 13 total features: 4 sensors for each type of reading, including power range, pressure and temperature. The last feature contains a categorical variable representing the status of the reactor which is either ‘Normal’ or ‘Abnormal’. There are no missing or null values within the dataset.  

Displayed here is the mean value of each numerical feature within the dataset. Pressure _sensor_1 is immediately interesting as it is much larger than the other Pressure mean values, Temperature_sensor_3 is also larger than other temperature values. This could be an indication of potential outliers within the data. However,  this disparity could be legitimate as each sensor is reading at different locations within the reactor. 

Here are the standard deviations for each numerical feature in the dataset. Pressure sensor 1 and temperature sensor 3 have much larger standard deviations than the other respective sensors. This could suggest there are outliers within this data but can potentially be a result of differences is sensor location. These extreme values only appear in individual features making these univariate outliers (Santoyo, 2017). The maximum values of each feature show the highest data point read by each sensor. Pressure sensor 1 has a much higher maximum value than the other pressure sensors, again suggesting potential outliers within the data. By creating a box plot it can be seen by the circles show outliers. 

Visualisation

This boxplot visualises the differences in both the normal and abnormal status classes. It can be seen here that when running normally, the reactor power is slightly higher on average, with a higher maximum value and interquartile range, too. Boxplots will identify outliers with circles above or below the maximum indicator, as seen here there are no clear outliers within this feature. The abnormal set is grouped tighter together than the normal, as seen by the shorter box.  

This graph is a density plot of the feature Pressure_sensor_1. The density plot visualises the distribution of data through the continuous sensor values. This graph shows the normal status data is more dense around the same pressure as the abnormal set. This may not be expected behaviour and therefore by looking at the other pressure sensors in comparison this might become clearer. 

These graphs show the density plot for pressure sensors 2 and 3 respectively. As suggested, there may be some underlying error or unexplained novelties within the data (Santoyo, 2017). As seen in the two graphs above, the abnormal class has a higher peak density. 

By plotting a box plot of pressure sensor 1 we can confirm that there are outliers within both classes of data. These extreme values are likely measurement errors or are natural outliers, meaning they are not an error and are instead novelties in the data (Santoyo, 2017). The outliers in the normal class are grouped tightly above the upper extreme which explains the higher density on the density plot. 

Preprocessing data

The data provided contains data from 3 different scales, power, pressure and temperature are all measured with different metrics. The differences in scale can cause discrepancies within the model through bias within the network for one feature over another (Priddy and Keller, 2005). Therefore, data normalization or standardization are necessary to bring all features to the same scale before using the data in the ANN. Furthermore, if the scale between features varies greatly, the larger values will have a larger contribution to the output error, meaning that error reduction will be focused on these larger scaled features (Sola and Sevilla, 1997). 

This implementation uses two methods of standardization, first using the StandardScaler model (std) and the second using the scale function (std2) from the preprocessing sublibrary of Sklearn. Due to standardization and normalization only functioning on numerical features, the Status feature is removed and is stored separately during pre-processing.. 

Data normalisation refers to the process of re-scaling numerical attributes to the range of 0 and 1. This is easy to do using the Sklearn preprocessing sublibrary using the MinMaxScaler function model as seen below:

Section 2: Selecting an algorithm

When selecting a model for a machine learning algorithm, there are two primary factors to consider: the model fit and complexity (Anderson and Burnham, 2004). Model fit refers to the inefficiencies found within models that are either overfitting or underfitting to their data set. 

As seen in the first graph above, a model of low complexity; a linear regression, has underfit the data and therefore that model would not function to accurately predict unknown values due to its simplicity. On the other hand, the last image above shows a complex function attempting to represent the data. This ‘dot to dot’ model is overfit to the data and therefore will also not be able to accurately predict unknown values because of its excessive complexity attempting to represent all data within the training set (Khaled,2017) . 

Validation is the concept of partitioning the dataset into 3 datasets; training, testing and validation (Khaled, 2017). The validation dataset is not used for training the model but instead is used to “give an estimate of model skill while tuning the model hyperparameters” (Brownlee, 2017). This provides an unbiased estimate of the final turned model to allow for comparison between final models. An example of this would be using the validation dataset to tune the optimal number of nodes in the hidden layer. 

However, assigning a set amount of data to a dedicated validation set can be costly for the training set if the total sample size is small. There is an alternative method of validation called k-fold cross validation. In this method, the training set is partitioned into k subsets. On each iteration, one of these random subsets acts as the validation set while the remaining k-1 folds are used to train the model and calculate and error score (Schönleber,2018). This is repeated with each fold against the rest of the set, for each set of hyperparameters a mean validation score is computed. 

In the scenario described in the brief, I would favor using a k-fold cross validation method as opposed to a set validation set. This is because the dataset is not large ( n = 1000) and therefore would be sacrificing data that could be used to further train the model. 

Section 3: Algorithm Design

Splitting data

Data is split into a training set and a test set (and potentially a validation set). The training set contains all features including the classification feature, in this case this is the reactor status. This data is used to train the network by comparing the processed outputs to the desired outputs. Errors are propagated back through the network, adjusting the weighting of the neurons. This process is repeated with the weighting being tweaked, this ‘trains’ the network by refining the connection weight (Reingold, 1999). 

The test dataset provided to the final model that has been trained by the training set for classification. The network is provided the data (minus the status) to then predict the status based on the data. 
The train_test_split function is imported from the sklearn.model_selection sublibrary. test_size = 0.1 defines the size of the test set as 10% of the total dataset. The preprocessed data along with the isolated status feature are passed as parameters which then returns randomly split training and test sets for both the sensor data and the status classifier. A random state seed is used for testing. 

Model training

A multi-layer perceptron is a machine learning algorithm consisting of 3 layers: An input layer, a hidden layer and an output layer as shown in Figure 1. 

Each data feature has a corresponding input node in the input layer, these neurons feed forward values through to the hidden layer. Each connection between node layers has a weight associated with it. Each node, or perceptron, calculates a weighted sum function of its inputs, plus bias. This is then fed into an activation function acting as a threshold, this is usually a sigmoid function:
 
To implement a multilayer perceptron classifier, the MLPClassifier function from the sklearn.neural_network sub library. This function creates an MLP algorithm model using backpropagation to reduce the error and produce a model representing the input data. The function takes many parameters including hidden_layer_sizes defining the number of hidden layers and nodes within each layer. Max_iter defines the maximum number of iterations to be performed by the network before stopping, this can happen if the learning rate of the network is too low and therefore convergence does not occur. solver=’adam’ refers to a stochastic gradient-based optimizer (Scikit-learn.org, 2019). This is an iterative optimisation technique that finds the partial derivatives of the loss function for each example in the data. That partial derivatives produce gradients that are then used to update the weights and biases of nodes within the network (Gandhi, 2018). 

Once the model is defined, it can then be fit with the training data. 

Random forest classifier

Random forest is a supervised learning algorithm that uses a collection of decision trees to predict results. Each decision tree is made of nodes that act as a query on an attribute, each branch represents the outcome and leaf nodes represent class labels. Therefore a decision tree is a structure made of feature tests that sequentially lead to a classifier label. The ‘forest’ refers to creating many of these decision trees from randomly selected samples from the training dataset (Donges, 2018). Once the forest model has been created, testing is performed by polling the trees with the test set and the answer is the classifier with the most votes. 

To implement a random forest classifier, the RandomForestClassifier model function is imported from the sklearn.ensemble sub library. n_estimators defines the number of trees in the forest and min_samples_leaf defines the minimum number of samples required to be at a leaf node.  

Once the model is defined, it is fit to the same training data as the MLPClassifier and a model is created.  
Results

This table shows the results of the test set accuracy results tested on three different preprocessing methods. Using the standardized preprocessing methods (StandardScaler and .scale) resulted in very high accuracy results of 0.99 for the MLP and 0.9 for the random forest. The accuracy of the random forest classifier was not significantly affected by differences in standardized or normalized datasets. 
Section 4: Model Selection
10 Fold Cross-Validation
K-fold cross-validation process begins by the training data set is split into k approximately equal subsets, as this is 10 fold, this k value is 10. In the code below, k value is the cv parameter of the GridSearchCV function of the sklearn.model_selection sub library: 

Once split, one of these folds is selected to be kept separate for testing, while the remaining k-1 folds are used to train the model. This is repeated k times so that each fold is used as the test set. The k-fold cross validation error is calculated by taking the average of the mean squared error (MSE) across all folds. 
The hyperparameters that are being tested are input as a dictionary called parameters. This model is then fit to the training dataset. Gridsearch.cv_results_ shows the mean testing score for each of the defined hyperparameters. Best_estimator_ returns the optimal parameters for the selected model. Here are the results: 

It was unexpected that a hidden layer size of 50 would have a higher accuracy than 100. This could be the result of an overly complex model of more nodes, causing the overfitting of the data and therefore a weaker accuracy score across all 10 folds. 

Shown here is the full model of parameters passed to the multi-layer perceptron model, with hidden_layer_sizes =  25. 
Next, the random forest classifier was validated using 10-fold cross validation:

The number of trees within this model did not appear to have a significant impact upon the accuracy of the algorithm with a difference of 0.024 between the best and worst with the optimal number of trees being 100. Here is the best_estimator_ for the RFC model:

Conclusion

After training supervised learning algorithms in the form of a multi-layered perceptron and a random forest classifier, much testing for each was conducted to find a model that best represented the data and also had potential to predict the classification of unknown data. 

Testing pre-processing to see how standardization and normalization impacted the accuracy of the resulting models showed that of the three pre-processing methods selected, none of them had a significant impact on the random forest classifier. However, the testing did show that the multi-layer perceptron accuracy was better using a standardized dataset over normalization. This leads me to conclude that the best model would use standardization as preprocessing -   there was no meaningful difference between standardization methods (StandardScale and .scale). 

MLP with a single hidden layer of 50 nodes performed better than any alternatives in the cross-validation. Testing also found they could reach an accuracy of up to 0.99 on the test dataset. This is a higher accuracy than RFC was able to achieve, therefore I would select the MLP model with 50 hidden layer nodes. 100 hidden layer nodes appeared to produce diminishing results potentially due to overfitting with an overly complex model. 

References

Santoyo, S. (2017) A Brief Overview of Outlier Detection Techniques. Towards Data Science. Available from: https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561

Priddy, K.L. and Keller, P.E. (2005) Artificial neural networks: an introduction (Vol. 68). SPIE press.

Sola, J. and Sevilla, J. (1997) Importance of input data normalization for the application of neural networks to complex industrial problems. IEEE Transactions on nuclear science, 44(3), pp.1464-1468.

Brownlee, J. (2014) Rescaling Data for Machine Learning in Python with Scikit-Learn. Machine Learning Mastery. Available from: https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/

Brownlee, J, (2017) What is the Difference Between Test and Validation Datasets? Machine Learning Mastery. Available from: 
https://machinelearningmastery.com/difference-test-validation-datasets/

Anderson, D.R. and Burnham, K. (2004) Model selection and multi-model inference. Second. NY: Springer-Verlag, p.63.

Schönleber, D. (2018) A “short” introduction to model selection. Towards Data Science. Available from: https://towardsdatascience.com/a-short-introduction-to-model-selection-bb1bb9c73376

Reingold, E (1999) Training an Artificial Neural Network. Department of Psychology University of Toronto. Available from: http://www.psych.utoronto.ca/users/reingold/courses/ai/cache/neural3.html

Scikit-learn.org (2019) sklearn.neural_network.MLPClassifier documentation. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

Gandhi, R. (2018) A Look at Gradient Descent and RMSprop Optimizers. Towards Data Science. Available from: https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b

Donges, N (2018) The Random Forest Algorithm. Towards Data Science. Available from: https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd

Khaled, A (2017) Machine Learning - What you need to know about “Model Selection and Evaluation”. Medium.com. Available from: https://medium.com/@lotass/machine-learning-what-you-need-to-know-about-model-selection-and-evaluation-8b641fd37fd5



