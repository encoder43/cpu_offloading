import graphlab
import numpy as np
from ridgeregression import ridge_regression_gradient_descent

data=graphlab.SFrame('E:\IOMT\Task 3\modified.csv')
'''
SFrame means scalable data frame. A tabular, column-mutable dataframe object that can scale to big data.
The data in SFrame is stored column-wise, 
and is stored on persistent storage (e.g. disk) to avoid being constrained by memory size.
'''
def get_numpy_data(data_sframe, features, output):
    
    data_sframe['constant'] = 1 # new constant column in the sframe signifying intercept
    
    features = ['constant'] + features # prepend constant to features list
    
    features_sframe = data_sframe[features] # new sframe selecting columns from data_sframe mentioned in features list

    feature_matrix = features_sframe.to_numpy() # convert sframe to numpy matrix

    output_sarray = data_sframe[output] # an sarray consisting of the output column

    output_array = output_sarray.to_numpy() # converts sarray to a numpy array

    return(feature_matrix, output_array)

'''
Function to predict output given feature matrix and weight vector
'''
def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return(predictions)

'''
'''

simple_features = ['S.no']
my_output = ['process_cpu_percent']

train_data,test_data = data.random_split(.8,seed=0)


initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000

(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
'''
Learned weights with no regulariztion
'''
model_features = ['S.no'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output =['process_cpu_percent']
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)


l2_penalty = 0.0
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 
                                                               l2_penalty, max_iterations)

def simple_weight_0_penalty():
    l2_penalty = 0.0
    simple_weight_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size,
                                                            l2_penalty, max_iterations)

'''
Learned weights with regulariztion
'''
model_features = ['S.no'] 
my_output = 'process_cpu_percent'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)


def simple_weight_high_penalty():
  l2_penalty = 1e11
  simple_weight_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size,
                                                            l2_penalty, max_iterations)
 

'''
RSS Function
'''
def RSS (predicted_output, true_output):
    residuals = predicted_output - true_output
    residuals_squared = residuals * residuals
    residuals_sum_of_squares = residuals_squared.sum()
    return residuals_sum_of_squares
'''
RSS on the TEST data: (Simple model)
The initial weights (all zeros)
The weights learned with no regularization
The weights learned with high regularization
'''
predictions = predict_output(simple_test_feature_matrix, initial_weights)
print("For simple model and initial weights:") 
print ("Weight (Coefficients): ") + str(initial_weights)
print( "RSS: ") + str(RSS(predictions, test_output))

predictions = predict_output(simple_test_feature_matrix, simple_weight_0_penalty)
print ("For simple model and weights with no regularization:")
print ("Weight (Coefficient): ") + str(simple_weight_0_penalty)
print ("RSS: ") + str(RSS(predictions, test_output))

predictions = predict_output(simple_test_feature_matrix, simple_weight_high_penalty)
print ("For simple model and weights with regularization:")
print ("Weight (Coefficient): ") + str(simple_weight_high_penalty)
print( "RSS: ") + str(RSS(predictions, test_output))

#Predicting the output using weights with no regularization
predicted_output = predict_output(test_feature_matrix[0:1], multiple_weights_0_penalty)
print( "Predicted output(weights with no regularization): ")
predicted_output[0]

