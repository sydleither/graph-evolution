from collections import Counter
from math import ceil
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import random
from scipy.stats import sem, t

from bintools import numBins
import eval_functions as ef
from organism import Organism

lmap = lambda f,x: list(map(f,x))
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#509154", "#A9561E", "#77BCFD", "#B791D4", "#EEDD5D", 
                                                    "#738696", "#24BCA8", "#D34A4F", "#8D81FE", "#FDA949"])


#transpose a matrix (list of list)
def T(LL:list[list])->list[list]:
    return list(zip(*LL))


#Algorithm from: Deb, Kalyanmoy, et al.
#"A fast and elitist multiobjective genetic algorithm: NSGA-II."
#Constraint handling from: Kalyanmoy Deb.
#"An efficient constraint handling method for genetic algorithms"
def fast_non_dominated_sort(population):
    F = {1:[]}
    S = {}
    n = {}
    valid_population = [p for p in population if p.valid]
    invalid_population = [p for p in population if not p.valid]
    for p in population:
        S[p.id] = []
        n[p.id] = 0
        if p.valid:
            for q in invalid_population:
                S[p.id].append(q)
            for q in valid_population:
                if p > q:
                    S[p.id].append(q)
                elif q > p:
                    n[p.id] += 1
        else:
            n[p.id] = len(valid_population)
        if n[p.id] == 0:
            p.nsga_rank = 1
            F[1].append(p)
    i = 1
    while F[i]:
        Q = []
        for p in F[i]:
            for q in S[p.id]:
                n[q.id] -= 1
                if n[q.id] == 0:
                    q.nsga_rank = i+1
                    Q.append(q)
        i += 1
        F[i] = Q
    return F


def get_perfect_pop(final_pop, objectives):
    return [final_pop[i] for i in range(len(final_pop)) 
            if all([np.isclose(final_pop[i].getError(name, target), 0) for name,target in objectives.items()])
            and final_pop[i].valid]


def calculate_standard_error(data:list[list[float]]) -> tuple[list[float], list[float], list[float]]:
    num_within = len(data)
    num_across = len(data[0])
    sqrtN = np.sqrt(num_within)
    data_t = T(data)
    data_mean = lmap(np.mean, data_t)
    data_error = lmap(lambda x:np.std(x)/sqrtN, data_t)
    neg_error = [data_mean[i]-data_error[i] for i in range(num_across)]
    pos_error = [data_mean[i]+data_error[i] for i in range(num_across)]
    return data_mean, neg_error, pos_error


def calculate_confidence_interval(data:list[list[float]]) -> tuple[list[float], list[float], list[float]]:
    num_across = len(data[0])
    data_t = T(data)
    data_mean = lmap(np.mean, data_t)
    intervals = [t.interval(confidence=0.95, df=len(data_t[i])-1, loc=data_mean[i], scale=sem(data_t[i])) for i in range(num_across)]
    lower = [intervals[i][0] for i in range(num_across)]
    upper = [intervals[i][1] for i in range(num_across)]
    return data_mean, lower, upper


def entropy_diff(file_name, config, property_names, data_path):
    if os.path.exists("{}/{}".format(data_path, file_name)):
        df = pd.read_csv("{}/{}".format(data_path, file_name))
    else:
        print(f"Please save {file_name} before calling plot_entropy_diff().")
        return
    N = config["popsize"]
    random_sample = [Organism(config["network_size"], random(), config["weight_range"]) for _ in range(N)]

    df_random = []
    for func_name in property_names:
        typeCounter = Counter([organism.getProperty(func_name) 
                               if "distribution" not in func_name 
                               else tuple(organism.getProperty(func_name)) 
                               for organism in random_sample])
        entropy = -sum([(count/N)*np.log2(count/N) for count in typeCounter.values()])
        df_random_entry = dict()
        df_random_entry["sampled_unique_types"] = len(typeCounter)
        df_random_entry["sampled_entropy"] = entropy
        df_random_entry["property"] = func_name
        df_random.append(df_random_entry)
    df_random = pd.DataFrame(df_random)
    df = df_random.merge(df, on=["property"], how="inner")
    df["entropy_pct_change"] = 100*((df["entropy"] - df["sampled_entropy"]) / df["sampled_entropy"])
    print(df)


def final_pop_histogram(final_pop, eval_funcs, save_loc, plot_all=True, transparent=False):
    #check if plotting one run or many replicates of a run
    plotting_replicates = isinstance(final_pop[0], list)

    #get list of properties to plot
    if plot_all:
        property_names = ef.properties
    else:
        property_names = [func_name for func_name in eval_funcs.keys() if not func_name.endswith("distribution")]

    #dynamically set size of figure
    num_plots = len(property_names)
    if num_plots == 0:
        #distribution only runs
        return
    
    fig_col_cnt = 2 if num_plots <= 4 else 4
    fig_row_cnt = ceil(num_plots/fig_col_cnt)
    figure, axis = plt.subplots(fig_row_cnt, fig_col_cnt, figsize=(5*fig_row_cnt, 3*fig_col_cnt), squeeze=False)
    fig_row = 0
    fig_col = 0

    #plot every property and if plotting more than the objective properties, color them differently
    for property_name in property_names:
        is_eval_func = property_name in eval_funcs.keys()
        if plot_all:
            color = "#509154" if is_eval_func else "#A9561E"
        else:
            color = "black" if plotting_replicates else "#509154"
        if plotting_replicates:
            data = [[org.getProperty(property_name) for org in final_pop[run]] for run in range(len(final_pop))]
            axis[fig_row][fig_col].hist(data, bins=numBins([d for dd in data for d in dd]), stacked=True)
        else:
            data = [org.getProperty(property_name) for org in final_pop]
            axis[fig_row][fig_col].hist(data, bins=numBins(data), stacked=False, color=color)
            color = "black"
        if is_eval_func:
            ideal_val = eval_funcs[property_name]
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


def final_pop_distribution(final_pop, eval_funcs, save_loc, plot_all=True, with_error=True, transparent=False):
    #check if plotting one run or many replicates of a run
    plotting_replicates = isinstance(final_pop[0], list)
    if not plotting_replicates:
        final_pop = [final_pop]
    
    #get list of properties to plot
    if plot_all:
        distributions = ef.distributions
    else:
        distributions = [dist_name for dist_name in eval_funcs.keys() if dist_name.endswith("distribution")]
    
    #dynamically set size of figure
    num_plots = len(distributions)
    if num_plots == 0:
        #no distribution run
        return
    fig_row_cnt = 2 if num_plots <= 4 else 4
    fig_col_cnt = ceil(num_plots/fig_row_cnt)
    figure, axis = plt.subplots(fig_row_cnt, fig_col_cnt, figsize=(4*fig_row_cnt, 4*fig_col_cnt), squeeze=False)
    fig_row = 0
    fig_col = 0

    #plot every distribution and if plotting more than the objective distributions, color them differently
    for dist_name in distributions:
        is_eval_func = dist_name in eval_funcs.keys()
        if plot_all:
            color = "#509154" if is_eval_func else "#A9561E"
        else:
            color = "black" if plotting_replicates else "#509154"
        for pop in final_pop:
            if len(pop) == 0:
                continue
            org_dists = [org.getProperty(dist_name) for org in pop]
            degree_mean, neg_error, pos_error = calculate_confidence_interval(org_dists)
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
            goal_dist = eval_funcs[dist_name]
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