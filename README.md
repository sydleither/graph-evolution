# Evolving Weighted and Directed Graphs
Using evolutionary computation to evolve graphs with constrained properties.

## References
### Methods Utilized
- Deb, K. (2000). [An efficient constraint handling method for genetic algorithms](https://www.sciencedirect.com/science/article/pii/S0045782599003898). _Computer methods in applied mechanics and engineering_, _186_(2-4), 311-338.
- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://ieeexplore.ieee.org/abstract/document/996017/). _IEEE transactions on evolutionary computation_, _6_(2), 182-197.
- Hornby, G. S. (2006, July). [ALPS: the age-layered population structure for reducing the problem of premature convergence](https://dl.acm.org/doi/abs/10.1145/1143997.1144142). In _Proceedings of the 8th annual conference on Genetic and evolutionary computation_ (pp. 815-822).

### As Seen In
- Leither, S., Ragusa, V., & Dolson, E. (2024, July). [Evolving weighted and directed graphs with constrained properties](https://dl.acm.org/doi/abs/10.1145/3638530.3654350). In _Proceedings of the Genetic and Evolutionary Computation Conference Companion_ (pp. 443-446).

## How to Use
Clone this repository, `git clone https://github.com/sydleither/graph-evolution`, to start evolving graphs with constrained properties. Graphs are generated via the parameters and constraints set within a json config file. A sample config, `sample_config.json` is provided in the repo.

### Config
Here we explain each parameter of the sample config.
- `"data_dir"`: specifies the parent directory to save runs in. Note that the directory structure is `graph-evolution/[data_dir]/[name]/[rep]`.
- `"name"`: specifies the name of the run.
- `"reps"`: specifies how many replicates to conduct of the run. Note that this will generate a directory with the number of each replicate, i.e. if `"reps": 3`, then three directories of names 0, 1, and 2 will be generated in `graph-evolution/[data_dir]/[name]`. Replicates can also instead be specified when calling `main.py` using the format `main.py sample_config.json [rep]`. Providing a replicate number when calling `main.py` will ignore the number of replicates set in the config. Giving the replicate directly to `main.py` is useful when pushing graph generation jobs to high-performance computing clusters (it allows each replicate to be a separate job).
- `"save_data"`: a boolean 0 or 1 that indicates whether or not to save the final population, fitness log, and diversity log of the run.
- `"plot_data"`: a boolean 0 or 1 that indicates whether or not to create plots of the run. The plots created include histograms of the properties of the graphs in the final population, the pareto front, fitness over time, diversity over time, and diversity statistics of the final population.
- `"tracking_frequency"`: how often to log fitness and diversity data (every "tracking_frequency" generations).
- `"track_diversity_over"`: which properties to track the diversity of over time.
- `"weight_range"`: the continuous range of values each edge may have. In the format `[min_val, max_val]`.
- `"network_size"`: the number of nodes in each graph.
- `"num_generations"`: how many generations to evolve the graphs for.
- `"popsize"`: the population size.
- `"age_gap"`: the frequency with which to reset the initial age layer.
- `"mutation_rate"`: the mutation rate. Mutations occur on individual edges of the genotype matrix.
- `"mutation_odds"`: the odds of each mutation type when a mutation happens (mutation types are point mutation, offset mutation, and flipping the pos/neg sign).
- `"crossover_rate"`: the crossover rate. Crossover occurs between genotype matrices.
- `"crossover_odds"`: the odds of each crossover type when a crossover happens (crossover types are binary node, depth-first traversal, and breadth-first traversal).
- `"tournament_probability"`: chance to conduct tournament selection vs randomly sample when choosing parents for reproduction.
- `"eval_funcs"`: the constrained properties of the graphs (the objectives of the evolutionary algorithm). In the format `"[property_name]":{"target":[target_val]}`. Multiple properties can be evolved by adding more entries to the `"eval_funcs"` dict with that same format.

### Evolving Graphs
Conduct a run with the parameters specified in the sample config file using `python3 main.py sample_config.json`. To analyze multiple replicates of a run, use `python3 replicate_analysis.py [data_dir]/[name]`. If you would like to integrate graph evolution in another python file, the evolutionary algorithm can be called by itself:
```
import json
from ga import run
config_file = json.load(open("sample_config.json"))
final_pop, fitness_log, diversity_log = run(config)
```
