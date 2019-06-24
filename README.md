# Polynomial-Estimators

Created various degrees of polynomial estimators

![](https://github.com/shashwatghatiwala/Polynomial-Estimators/blob/master/1_.png)

For the training data, as the degree of the polynomial increases, the MSE drastically reduces. This reduction is due to the rise in flexibility of the polynomial estimators. The first polynomial estimator (of degree 1) has very low flexibility and estimates the data in a linear manner. On the other hand, the degree 50 polynomial has substantially high flexibility and due to the small training data size, it fits on almost all the training data thereby, making the MSE go almost to zero. 

However, on the testing data (which is even smaller in size), the MSE is magnified as the degree of polynomial rises. The higher degree polynomials witness large MSE because they overfit the data. Their large flexibility leads them to fit each data point of the testing set and also, crossing through few data points multiple times. By tracking the noise in the testing set, the higher degree polynomials have very large variance.
