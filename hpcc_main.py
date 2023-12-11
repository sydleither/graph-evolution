import json
import pickle
import sys

from ga import run


def main(config):
    final_pop, fitness_log = run(config)
    with open('final_pop.pkl', 'wb') as f:
        pickle.dump(final_pop, f)
    with open('fitness_log.pkl', 'wb') as f:
        pickle.dump(fitness_log, f)


if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
        config = json.load(open(config_file))
    except:
        print("Please give a valid config json to read parameters from.")
        exit()
    main(config)