import numpy as np


def calculate_standard_error(data:list[list[float]]) -> (list[float], list[float], list[float]):
    num_within = len(data)
    num_across = len(data[0])
    data_mean = [np.mean([data[i][j] for i in range(num_within)]) for j in range(num_across)]
    data_error = [np.std([data[i][j] for i in range(num_within)])/np.sqrt(num_within) for j in range(num_across)]
    neg_error = [data_mean[i]-data_error[i] for i in range(num_across)]
    pos_error = [data_mean[i]+data_error[i] for i in range(num_across)]
    return data_mean, neg_error, pos_error