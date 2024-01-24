import os
import pickle
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


OBJECTIVES_OF_INTEREST = ["connectance", "average_positive_interactions_strength", "average_negative_interactions_strength",
                          "number_of_competiton_pairs", "positive_interactions_proportion", "strong_components", 
                          "proportion_of_self_loops", "in_degree_distribution", "out_degree_distribution"]
                          #"pos_in_weight_distribution", "pos_out_weight_distribution", "neg_in_weight_distribution", "neg_out_weight_distribution"]


def mse_boxplot(df, x, y, hue, group, title):
    figure, axis = plt.subplots(2, 2, figsize=(12,12))
    row = 0
    col = 0
    for g in df[group].unique():
        sns.boxplot(data=df.loc[df[group] == g], x=x, y=y, hue=hue, ax=axis[row][col])
        axis[row][col].set_yscale('log')
        #axis[row][col].set_title("Experiment {}".format(row+col))
        row += 1
        if row % 2 == 0:
            col += 1
            row = 0
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle(title)
    plt.savefig("{}.png".format(title))
    plt.close()


def main(data_dir):
    df_cols = ["experiment_name", "rep", "objective", "target", "network_size", 
               "MSE", "Entropy", "Drift_Entropy_Avg"]
    df_rows = []

    for experiment_dir in os.listdir(data_dir):
        full_obj_path = "{}/{}".format(data_dir, experiment_dir)
        if not os.path.isfile(full_obj_path):
            for rep_dir in os.listdir(full_obj_path):
                full_rep_path = "{}/{}".format(full_obj_path, rep_dir)
                if not os.path.isfile(full_rep_path):
                    #skip uncompleted experiments
                    if not os.path.exists("{}/final_pop.pkl".format(full_rep_path)):
                        print("Skipped {} rep {}".format(experiment_dir, rep_dir))
                        continue
                    #read in fitness log
                    with open("{}/fitness_log.pkl".format(full_rep_path), "rb") as f:
                        fitness_log = pickle.load(f)
                    #read in entropy csv as a dataframe
                    entropy_df = pd.read_csv("{}/entropy.csv".format(full_rep_path))
                    #subset entropy dataframe to only include objectives of interest under drift
                    drift_objectives = [x for x in OBJECTIVES_OF_INTEREST if x not in list(fitness_log.keys())]
                    drift_entropies = entropy_df.loc[entropy_df["Name"].isin(drift_objectives)]
                    #get mean and std of drift entropies
                    drift_entropy_avg = drift_entropies["Entropy(bits)"].mean() #TODO change this to delta entropy
                    #get details of experiments via experiment directory name
                    parts_of_experiment_dir_name = experiment_dir.split("_")
                    if data_dir.endswith("lots"):
                        experiment_name = parts_of_experiment_dir_name[0]
                        target = parts_of_experiment_dir_name[1]
                        network_size = parts_of_experiment_dir_name[2]
                    else:
                        target = parts_of_experiment_dir_name[-2]
                        network_size = parts_of_experiment_dir_name[-1]
                        experiment_name = "_".join(parts_of_experiment_dir_name[1:-2])
                    #add values of interest to a list to turn into a dataframe
                    for objective,fitnesses in fitness_log.items():
                        entropy = entropy_df.loc[entropy_df["Name"] == objective]["Entropy(bits)"].values[0]
                        df_rows.append([experiment_name, rep_dir, objective, target, int(network_size), 
                                        float(fitnesses[-1]), float(entropy), float(drift_entropy_avg)])

    df = pd.DataFrame(data=df_rows, columns=df_cols)
    df_reduced = df[["experiment_name", "target", "objective", "MSE", "Entropy", "Drift_Entropy_Avg"]]
    print(df_reduced.groupby(["experiment_name", "target", "objective"]).mean())

    if data_dir.endswith("lots"):
        mse_boxplot(df.loc[df["experiment_name"] == "3"], "network_size", "MSE", "objective", "target", "Three Objectives")
        mse_boxplot(df.loc[df["experiment_name"] == "4"], "network_size", "MSE", "objective", "target", "Four Objectives")
        mse_boxplot(df.loc[df["experiment_name"] == "5"], "network_size", "MSE", "objective", "target", "Five Objectives")
        mse_boxplot(df.loc[df["experiment_name"] == "6"], "network_size", "MSE", "objective", "target", "Six Objectives")

        # seaborn_boxplot(df.loc[df["experiment_name"] == "3"], "network_size", "Drift_Entropy_Avg", None, "target", "Three Objectives Entropy")
        # seaborn_boxplot(df.loc[df["experiment_name"] == "4"], "network_size", "Drift_Entropy_Avg", None, "target", "Four Objectives Entropy")
        # seaborn_boxplot(df.loc[df["experiment_name"] == "5"], "network_size", "Drift_Entropy_Avg", None, "target", "Five Objectives Entropy")
        # seaborn_boxplot(df.loc[df["experiment_name"] == "6"], "network_size", "Drift_Entropy_Avg", None, "target", "Six Objectives Entropy")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Please give a valid data directory holding experiments.')