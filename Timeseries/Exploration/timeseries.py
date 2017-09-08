import numpy as np 

def simple_average(input_data):
    return [np.mean(input_data[:max(0, i-1)]) for i in range(len(input_data))]

def moving_average(input_data, window_size):
    '''
    Returns a list of lists, for each window size provided
    in the window_size list
    '''
    
    moving_averages = []
    
    for window in window_size:
        single_moving_average = [np.mean(input_data[max(i-(window+1),0):max(i-1, 0)]) 
                     for i in range(len(input_data))]
        moving_averages.append(single_moving_average)
    
    return moving_averages
    
def single_value_exponential_smoothing_1(input_data, alphas):
    '''
    Returns a list of lists, for each alpha provided in the 
    alphas list 
    This method only predicts one value into the future
    '''
    
    exponential_smoothing_alphas = []
    
    for alpha in alphas: 
        exponential_smoothing = []

        for i in range(len(input_data)):
            exponential_sum = [(((1-alpha)**index)*value) for index, value in enumerate(input_data[max(0, i-1)::-1])]
            exponential_smoothing.append(alpha*sum(exponential_sum))
        exponential_smoothing_alphas.append(exponential_smoothing)
    
    return exponential_smoothing_alphas
        
def single_value_exponential_smoothing_2(input_data, alphas, early_stopping = True):
    '''
    Returns a list of lists, for each alpha provided in the 
    alphas list 
    This method predicts many values into the future
    '''    
    
    all_future_predictions = []
    for alpha in alphas:
        future_predictions = truncated_values[:]
        for i in range(len(future_predictions), len(original_co2_data)):
            exponential_sum = [(((1-alpha)**index)*value) for index, value in enumerate(future_predictions[max(0, i-1)::-1])]
            future_predictions.append(alpha*sum(exponential_sum))
            if early_stopping:
                if len(set([int(x) for x in future_predictions[-3:]])) <= 1: 
                    break
        all_future_predictions.append(future_predictions)
    return all_future_predictions
    
    
    