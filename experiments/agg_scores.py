import os
import pandas as pd
import pickle
import sys


def main(data_dir, experiment):
    df_cols = ["rep", "selection", "objective", "target", "network_size", 
               "MSE", "Entropy", "Drift_Entropy_Avg", "Drift_Entropy_Std"]
    df_rows = []

    for experiment_dir in os.listdir(data_dir):
        full_obj_path = "{}/{}".format(data_dir, experiment_dir)
        if not os.path.isfile(full_obj_path) and experiment in experiment_dir:
            for rep_dir in os.listdir(full_obj_path):
                full_rep_path = "{}/{}".format(full_obj_path, rep_dir)
                if not os.path.isfile(full_rep_path):
                    #read in fitness log
                    with open("{}/fitness_log.pkl".format(full_rep_path), "rb") as f:
                        fitness_log = pickle.load(f)
                    #read in entropy csv as a dataframe
                    entropy_df = pd.read_csv("{}/entropy.csv".format(full_rep_path))
                    #subset entropy dataframe to only include objectives under drift
                    drift_entropies = entropy_df.loc[~entropy_df["Name"].isin(list(fitness_log.keys()))]
                    #get mean and std of drift entropies
                    drift_entropy_avg = drift_entropies["Entropy(bits)"].mean() #TODO change this to delta entropy
                    drift_entropy_std = drift_entropies["Entropy(bits)"].std()
                    #get details of experiments via experiment name
                    selection_scheme, _, target, network_size = experiment_dir.split("_")
                    #add values of interest to a list to turn into a dataframe
                    for objective,fitnesses in fitness_log.items():
                        entropy = entropy_df.loc[entropy_df["Name"] == objective]["Entropy(bits)"].values[0]
                        df_rows.append([rep_dir, selection_scheme, objective, target, network_size, 
                                        fitnesses[-1], entropy, drift_entropy_avg, drift_entropy_std])

    df = pd.DataFrame(data=df_rows, columns=df_cols)
    print(df)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print('Please give a valid location and experiment name.')