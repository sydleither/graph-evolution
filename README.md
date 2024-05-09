# graph-evolution
Using evolutionary computation to evolve graphs with constrained properties.

## How to Use
Clone this repository, `git clone https://github.com/sydleither/graph-evolution`, to start evolving graphs with constrained properties. Graphs are generated via the parameters and constraints set within a json config file. A sample config, `sample_config.json` is provided in the repo.

### Config
Here we explain each parameter of the sample config.
- `"data_dir"`: specifies the parent directory to save runs in. Note that the directory structure is `graph-evolution/[data_dir]/[name]/[rep]`.
- `"name"`: specifies the name of the run.
- `"reps"`: specifies how many replicates to conduct of the run. Note that this will generate a directory with the number of each replicate, i.e. if `"reps": 3`, then three directories of names 0, 1, and 2 will be generated in `graph-evolution/[data_dir]/[name]`. Replicates can also instead be specified when calling `main.py` using the format `main.py sample_config.json [rep]`. Providing a replicate number when calling `main.py` will ignore the number of replicates set in the config. Giving the replicate directly to `main.py` is useful when pushing graph generation jobs to high-performance computing clusters (it allows each replicate to be a separate job).
- `"save_data"`: a boolean 0 or 1 that indicates whether or not to save the final population and fitness log of the run.
- `"plot_data"`: a boolean 0 or 1 that indicates whether or not to create plots of the run. The plots created include histograms of the properties of the graphs in the final population, the pareto front, fitness over time, and the diversity of the final population.
- `"popsize"`: the population size.
- `"mutation_rate"`: the mutation rate. Mutations occur on individual edges of the genotype matrix.
- `"mutation_odds"`: the odds of each mutation type when a mutation happens (mutation types are point mutation, offset mutation, and flipping the pos/neg sign).
- `"crossover_rate"`: the crossover rate. Crossover occurs between genotype matrices.
- `"crossover_odds"`: the odds of each crossover type when a crossover happens (crossover types are binary node, depth-first traversal, and breadth-first traversal)
- `"weight_range"`: the continuous range of values each edge may have. In the format `[min_val, max_val]`.
- `"network_size"`: the number of nodes in each graph.
- `"num_generations"`: how many generations to evolve the graphs for.
- `"eval_funcs"`: the constrained properties of the graphs (the objectives of the evolutionary algorithm). In the format `"[property_name]":{"target":[target_val]}`. Multiple properties can be evolved by adding more entries to the `"eval_funcs"` dict with that same format.

### Evolving Graphs
Conduct a run with the parameters specified in the sample config file using `python3 main.py sample_config.json`. To analyze multiple replicates of a run, use `python3 replicate_analysis.py [data_dir]/[name]`. If you would like to integrate graph evolution in another python file, the evolutionary algorithm can called by itself:
```
import json
from ga import run
config_file = json.load(open("sample_config.json"))
final_pop, fitness_log = run(config)
```
