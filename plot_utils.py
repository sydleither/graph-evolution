import numpy as np


lmap = lambda f,x: list(map(f,x))

#transpose a matrix (list of list)
def T(LL:list[list])->list[list]:
    return list(zip(*LL))


def calculate_standard_error(data:list[list[float]]) -> (list[float], list[float], list[float]):
    num_within = len(data)
    num_across = len(data[0])
    sqrtN = np.sqrt(num_within)
    data_t = T(data)
    data_mean = lmap(np.mean, data_t)
    data_error = lmap(lambda x:np.std(x)/sqrtN, data_t)
    neg_error = [data_mean[i]-data_error[i] for i in range(num_across)]
    pos_error = [data_mean[i]+data_error[i] for i in range(num_across)]
    return data_mean, neg_error, pos_error