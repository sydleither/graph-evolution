from math import ceil
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap, sem, t

from bintools import numBins
import eval_functions as ef

lmap = lambda f,x: list(map(f,x))


#transpose a matrix (list of list)
def T(LL:list[list])->list[list]:
    return list(zip(*LL))


#Algorithm from: Deb, Kalyanmoy, et al.
#"A fast and elitist multiobjective genetic algorithm: NSGA-II."
#IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
def fast_non_dominated_sort(population):
    F = {1:[]}
    S = {}
    n = {}
    for p in population:
        S[p.id] = []
        n[p.id] = 0
        for q in population:
            if p > q:
                S[p.id].append(q)
            elif q > p:
                n[p.id] += 1
        if n[p.id] == 0:
            p.nsga_rank = 1
            F[1].append(p)
    i = 1
    while len(F[i]) > 0:
        Q = []
        for p in F[i]:
            for q in S[p.id]:
                n[q.id] -= 1
                if n[q.id] == 0:
                    q.nsga_rank = i+1
                    Q.append(q)
        i += 1
        F[i] = Q[:]
    return F


def get_perfect_pop(final_pop, objectives):
    return [final_pop[i] for i in range(len(final_pop)) 
            if all([final_pop[i].getError(name, target) == 0 
                    for name,target in objectives.items()])]


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


def calculate_confidence_interval(data:list[list[float]], bootstrapped=False) -> tuple[list[float], list[float], list[float]]:
    num_across = len(data[0])
    data_t = T(data)
    data_mean = lmap(np.mean, data_t)
    if not bootstrapped:
        intervals = [t.interval(confidence=0.95, df=len(data_t[i])-1, loc=data_mean[i], scale=sem(data_t[i])) for i in range(num_across)]
    else:
        intervals = [bootstrap((data_t[i],), np.mean, confidence_level=0.95, method="percentile").confidence_interval for i in range(num_across)]
    lower = [intervals[i][0] for i in range(num_across)]
    upper = [intervals[i][1] for i in range(num_across)]
    return data_mean, lower, upper


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
            color = "forestgreen" if is_eval_func else "sienna"
        else:
            color = "black" if plotting_replicates else "forestgreen"
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
            color = "forestgreen" if is_eval_func else "sienna"
        else:
            color = "black" if plotting_replicates else "forestgreen"
        for pop in final_pop:
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


def plot_elites_map(elites_map, eval_funcs, features_dict, save_loc, transparent=False):
    def generate_heatmap(col, row, obj_name, obj_target, constant_val=None, constant_idx=None):
        mean_heatmap = np.empty([len(row), len(col)])
        count_heatmap = np.empty([len(row), len(col)])
        two_dim = len(row) > 1
        third = constant_val is not None and constant_idx is not None
        for i in range(len(col)):
            for j in range(len(row)):
                cell_idx = [col[i], row[j]] if two_dim else [col[i],]
                if third:
                    cell_idx.insert(constant_idx, constant_val)
                cell = elites_map[tuple(cell_idx)]
                num_orgs_in_cell = len(cell)
                if num_orgs_in_cell > 0:
                    mean_heatmap[j,i] = round(mean([org.getError(obj_name, obj_target) for org in cell]), 3)
                else:
                    mean_heatmap[j,i] = None
                count_heatmap[j,i] = num_orgs_in_cell
        return mean_heatmap, count_heatmap
    
    def save_heatmaps(mean_heatmap, count_heatmap, col_name, row_name, objective, title):
        col_labels = features_dict[col_name]
        row_labels = features_dict[row_name] if row_name is not None else [""]
        figure, axis = plt.subplots(1, 2, figsize=(16,48))
        axis[0].imshow(mean_heatmap, cmap="summer_r")
        axis[0].set_xticks(np.arange(len(col_labels)), labels=col_labels)
        axis[0].set_title("Mean Cell {} Error".format(objective))
        axis[1].imshow(count_heatmap, cmap="summer")
        axis[1].set_xticks(np.arange(len(col_labels)), labels=col_labels)
        axis[1].set_title("Count of Organisms in Each Cell")
        if row_name != "genome_hash":
            axis[0].set_yticks(np.arange(len(row_labels)), labels=row_labels)
            axis[1].set_yticks(np.arange(len(row_labels)), labels=row_labels)
        for i in range(len(col_labels)):
            for j in range(len(row_labels)):
                _ = axis[0].text(i, j, mean_heatmap[j, i], ha="center", va="center", color="black", fontsize="small")
                _ = axis[1].text(i, j, count_heatmap[j, i], ha="center", va="center", color="black", fontsize="small")
        figure.supxlabel(col_name)
        figure.supylabel(row_name)
        figure.suptitle(title)
        figure.tight_layout()
        if transparent:
            figure.patch.set_alpha(0.0)
        plt.savefig("{}/heatmap_{}.png".format(save_loc, name+"_"+title))
        plt.close()

    feature_names = list(features_dict.keys())
    feature_bins = list(features_dict.values())
    for name,target in eval_funcs.items():
        num_features = len(features_dict)
        if num_features == 1:
            mean_heatmap, count_heatmap = generate_heatmap(feature_bins[0], [None], name, target)
            save_heatmaps(mean_heatmap, count_heatmap, feature_names[0], None, name, "")
        elif num_features == 2:
            mean_heatmap, count_heatmap = generate_heatmap(feature_bins[0], feature_bins[1], name, target)
            save_heatmaps(mean_heatmap, count_heatmap, feature_names[0], feature_names[1], name, "")
        elif num_features == 3:
            for fval in feature_bins[2]:
                mean_heatmap, count_heatmap = generate_heatmap(feature_bins[0], feature_bins[1], name, target, fval, 2)
                save_heatmaps(mean_heatmap, count_heatmap, feature_names[0], feature_names[1], name, f"{feature_names[2]}={fval}")
        else:
            print("Too many features to plot elites map.")
            return