from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from bintools import numBins
from eval_functions import Evaluation


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


def final_pop_histogram(eval_obj, final_pop, eval_funcs, save_loc, plot_all=True, transparent=False):
    #check if plotting one run or many replicates of a run
    plotting_replicates = isinstance(final_pop[0], list)
    #get list of properties to plot
    if plot_all:
        property_names = [func for func in dir(Evaluation) 
                          if callable(getattr(Evaluation, func)) 
                          and not (func.startswith("__") or func.endswith("distribution"))]
    else:
        property_names = [func for func in eval_funcs.keys() if not func.endswith("distribution")]
    #dynamically set size of figure
    num_plots = len(property_names)
    if num_plots == 0:
        return
    fig_col_cnt = 1 if num_plots == 1 else 2 if num_plots <= 4 else 4
    fig_row_cnt = ceil(num_plots/fig_col_cnt)
    figure, axis = plt.subplots(fig_row_cnt, fig_col_cnt, figsize=(5*fig_row_cnt, 3*fig_col_cnt), squeeze=False)
    fig_row = 0
    fig_col = 0
    #plot every property and if plotting more than the objective properties, color them differently
    for property_name in property_names:
        is_eval_func = property_name in eval_funcs.keys()
        if plot_all:
            color = "forestgreen" if is_eval_func else "sienna"
        else:
            color = "black" if plotting_replicates else "forestgreen"
        eval_func = getattr(eval_obj, property_name)
        if plotting_replicates:
            data = [[eval_func(org) for org in final_pop[run]] for run in range(len(final_pop))]
            axis[fig_row][fig_col].hist(data, bins=numBins([d for dd in data for d in dd]), stacked=True)
        else:
            data = [eval_func(org) for org in final_pop]
            axis[fig_row][fig_col].hist(data, bins=numBins(data), stacked=False, color=color)
            color = "black"
        if is_eval_func:
            ideal_val = eval_funcs[property_name]["target"] if "target" in eval_funcs[property_name].keys() else 0
            axis[fig_row][fig_col].axvline(ideal_val, color="black", linestyle="--")
        axis[fig_row][fig_col].set_title(property_name, color=color)
        fig_row += 1
        if fig_row % fig_row_cnt == 0:
            fig_col += 1
            fig_row = 0
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle("Final Population Histograms")
    fig_name = "histograms_all" if plot_all else "histograms"
    plt.savefig("{}/{}.png".format(save_loc, fig_name), bbox_inches='tight', transparent=transparent)
    plt.close()


def final_pop_distribution(eval_obj, final_pop, eval_funcs, save_loc, plot_all=True, with_error=True, transparent=False):
    #check if plotting one run or many replicates of a run
    plotting_replicates = isinstance(final_pop[0], list)
    if not plotting_replicates:
        final_pop = [final_pop]
    #get list of properties to plot
    if plot_all:
        distributions = [func for func in dir(Evaluation) if callable(getattr(Evaluation, func)) and func.endswith("distribution")]
    else:
        distributions = [func for func in eval_funcs.keys() if func.endswith("distribution")]
    #dynamically set size of figure
    num_plots = len(distributions)
    if num_plots == 0:
        return
    fig_col_cnt = 1 if num_plots == 1 else 2 if num_plots <= 4 else 4
    fig_row_cnt = ceil(num_plots/fig_col_cnt)
    figure, axis = plt.subplots(fig_row_cnt, fig_col_cnt, figsize=(4*fig_row_cnt, 3*fig_col_cnt), squeeze=False)
    fig_row = 0
    fig_col = 0
    #plot every distribution and if plotting more than the objective distributions, color them differently
    for dist_name in distributions:
        is_eval_func = dist_name in eval_funcs.keys()
        if plot_all:
            color = "forestgreen" if is_eval_func else "sienna"
        else:
            color = "black" if plotting_replicates else "forestgreen"
        for pop in final_pop:
            eval_func = getattr(eval_obj, dist_name)
            org_dists = [eval_func(org) for org in pop]
            degree_mean, neg_error, pos_error = calculate_standard_error(org_dists)
            if plotting_replicates:
                axis[fig_row][fig_col].plot(degree_mean, label=dist_name)
                axis[fig_row][fig_col].fill_between(range(len(degree_mean)), neg_error, pos_error, alpha=0.5)
            else:
                if with_error:
                    axis[fig_row][fig_col].plot(degree_mean, label=dist_name, color=color)
                    axis[fig_row][fig_col].fill_between(range(len(degree_mean)), neg_error, pos_error, alpha=0.5, color=color)
                else:
                    for org_dist in org_dists:
                        axis[fig_row][fig_col].plot(org_dist, color=color)
                color = "black"
        if is_eval_func:
            goal_dist = eval_obj.dist_dict[dist_name]
            axis[fig_row][fig_col].plot(goal_dist, color="black", linewidth=2)
        axis[fig_row][fig_col].set_title(dist_name, color=color)
        fig_row += 1
        if fig_row % fig_row_cnt == 0:
            fig_col += 1
            fig_row = 0
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle("Final Population Distributions")
    fig_name = "distributions_all" if plot_all else "distributions"
    if with_error and not plotting_replicates:
        fig_name = fig_name + "_w_error"
    plt.savefig("{}/{}.png".format(save_loc, fig_name), bbox_inches='tight', transparent=transparent)
    plt.close()