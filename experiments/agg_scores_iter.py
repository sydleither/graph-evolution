from collections import Counter
import os
import pickle
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


OBJECTIVES_OF_INTEREST = ["connectance", "average_positive_interactions_strength", "average_negative_interactions_strength",
                          "number_of_competiton_pairs", "positive_interactions_proportion", "strong_components", 
                          "proportion_of_self_loops", "in_degree_distribution", "out_degree_distribution"]


def entropy_boxplot(df, x, y, hue, group, iter_path, num_obj):
    figure, axis = plt.subplots(4, 5, figsize=(32,20))
    row = 0
    col = 0
    for g in df[group].unique():
        sns.boxplot(data=df.loc[df[group] == g], x=x, y=y, hue=hue, ax=axis[row][col])
        axis[row][col].set_yscale('log')
        #axis[row][col].set_title("Experiment {}".format(row+col))
        row += 1
        if row % 4 == 0:
            col += 1
            row = 0
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle("Iteration path {}, {} objectives".format(iter_path, num_obj))
    plt.savefig("{}_{}.png".format(iter_path, num_obj))
    plt.close()


def mse_boxplot(df, x, y, hue, group, iter_path, num_obj):
    figure, axis = plt.subplots(4, 5, figsize=(32,20))
    row = 0
    col = 0
    for g in df[group].unique():
        sns.boxplot(data=df.loc[df[group] == g], x=x, y=y, hue=hue, ax=axis[row][col])
        axis[row][col].set_yscale('log')
        #axis[row][col].set_title("Experiment {}".format(row+col))
        row += 1
        if row % 4 == 0:
            col += 1
            row = 0
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle("Iteration path {}, {} objectives".format(iter_path, num_obj))
    plt.savefig("{}_{}.png".format(iter_path, num_obj))
    plt.close()


def main(data_dir):
    df_cols = ["experiment_name", "num_obj", "iter_path", "combo", "rep", "network_size", 
               "objective", "MSE", "entropy", "num_unique", "pop_size"]
    df_rows = []

    for experiment_dir in os.listdir(data_dir):
        full_obj_path = "{}/{}".format(data_dir, experiment_dir)
        if not os.path.isfile(full_obj_path):
            for rep_dir in os.listdir(full_obj_path):
                full_rep_path = "{}/{}".format(full_obj_path, rep_dir)
                if not os.path.isfile(full_rep_path):
                    #skip uncompleted experiments or read in final pop
                    if not os.path.exists("{}/final_pop.pkl".format(full_rep_path)):
                        print("Skipped {} rep {}".format(experiment_dir, rep_dir))
                        break
                    with open("{}/final_pop.pkl".format(full_rep_path), "rb") as f:
                        final_pop = pickle.load(f)
                    #get unique counts of properties in final pop
                    property_counts = {}
                    for property in OBJECTIVES_OF_INTEREST:
                        if property.endswith("distribution"):
                            orgs = [tuple(org.getProperty(property)) for org in final_pop]
                        else:
                            orgs = [org.getProperty(property) for org in final_pop]
                        unique_orgs = len(Counter(orgs))
                        property_counts[property] = unique_orgs
                    #read in fitness log
                    with open("{}/fitness_log.pkl".format(full_rep_path), "rb") as f:
                        fitness_log = pickle.load(f)
                    #read in entropy csv as a dataframe
                    entropy_df = pd.read_csv("{}/entropy.csv".format(full_rep_path))
                    #get details of experiments via experiment directory name
                    parts_of_experiment_dir_name = experiment_dir.split("_")
                    experiment_name = experiment_dir
                    num_obj = parts_of_experiment_dir_name[0]
                    iter_path = parts_of_experiment_dir_name[1]
                    combo = parts_of_experiment_dir_name[2]
                    network_size = parts_of_experiment_dir_name[3]
                    #add values of interest to a list to turn into a dataframe
                    for objective,fitnesses in fitness_log.items():
                        entropy = entropy_df.loc[entropy_df["Name"] == objective]["Entropy(bits)"].values[0]
                        num_unique = property_counts[objective]
                        df_rows.append([experiment_name, num_obj, iter_path, combo, rep_dir, int(network_size), 
                                        objective, float(fitnesses[-1]), float(entropy), num_unique, 200])

    df = pd.DataFrame(data=df_rows, columns=df_cols)
    #print(df[["iter_path", "combo", "objective", "MSE", "entropy", "Drift_Entropy_Avg"]].groupby(["iter_path", "combo", "objective"]).mean())
    # for iter_exp in df["iter_path"].unique():
    #     df_iter = df.loc[df["iter_path"] == iter_exp]
    #     for obj_num in df_iter["num_obj"].unique():
    #         df_iter_obj = df_iter.loc[df_iter["num_obj"] == obj_num]
    #         print("{}_{}".format(iter_exp, obj_num))
    #         mse_boxplot(df_iter_obj, "network_size", "MSE", "objective", "combo", iter_exp, obj_num)

    entropy_boxplot(df, "objective")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Please give a valid data directory holding experiments.')