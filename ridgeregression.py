import numpy as np
from dataanalysis import predict_output
import dataanalysis

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    derivative = 0
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    if feature_is_constant:
        total_error = errors.sum()
        derivative = 2 * total_error
    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    else:
        dot_product = np.dot(errors, feature)
        rss_part = 2 * dot_product
        regularized_part = 2 * l2_penalty * weight
        derivative = rss_part + regularized_part
    return derivative


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    print ('Starting gradient descent with l2_penalty = ' + str(l2_penalty))
    
    weights = np.array(initial_weights) # a numpy array
    iteration = 0 # iteration counter
    print_frequency = 1  # for adjusting frequency of debugging output
    
    # while not reached maximum number of iterations        
    while iteration < max_iterations: 
        iteration += 1  # increment iteration counter
        
        ### === code section for adjusting frequency of debugging output. ===
        if iteration == 10:
            print_frequency = 10
        if iteration == 100:
            print_frequency = 100
        if iteration%print_frequency==0:
            print('Iteration = ' + str(iteration))
        ### === end code section ===
        
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)

        # compute the errors in prediction
        errors = predictions - output

        # from time to time, print the value of the cost function
        if iteration%print_frequency==0:
            print( 'Cost function = '), str(np.dot(errors,errors) + l2_penalty*(np.dot(weights,weights) - weights[0]**2))
        
        for i in range(len(weights)): # loop over each weight
             # feature column associated with weights[i]
            feature_column = feature_matrix[:,i]
            
            # computing derivative of weight[i]
            if i == 0:  # feature is constant
                derivative = feature_derivative_ridge(errors, feature_column, weights[i], l2_penalty, True)
            else:       # feature is not constant
                derivative = feature_derivative_ridge(errors, feature_column, weights[i], l2_penalty, False)
                
            # subtracting the step size times the derivative from the current weight
            weights[i] = weights[i] - (step_size * derivative)
            
    print('Done with gradient descent at iteration ', iteration)
    print('Learned weights = ', str(weights))
    return weights

