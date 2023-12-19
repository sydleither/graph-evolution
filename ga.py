from random import randint, sample, shuffle
from statistics import mean
from typing import Callable

from eval_functions import Evaluation
from organism import Organism


#transpose of a matrix (list-of-list)
def T(LL:list[list]) -> list[list]:
    return list(zip(*LL))


#inspired by: Deb, Kalyanmoy, et al.
#"A fast and elitist multiobjective genetic algorithm: NSGA-II."
#IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
def getDiverseChoice(organismList:list[Organism]) -> list[float]:
    global eval_obj
    extremeOrganisms = []
    for funcName,eval_func in eval_obj.functions.items():
        objectiveScores = sorted([(org.getProperty(funcName,eval_func),org) for org in organismList],key=lambda x:x[0])
        extremeOrganisms.append(objectiveScores[0][1])
        extremeOrganisms.append(objectiveScores[-1][1])
    return sample(extremeOrganisms,k=1)[0]


def epsilonLexicase(population:list[Organism], numParents:int, popsize:int, eval_funcs:dict, epsilon:float) -> list[Organism]:
    parents:list[Organism] = []
    objectiveNames:list[str] = list(eval_funcs.keys())

    for _ in range(numParents):
        shuffle(objectiveNames) #randomize objective evaluation order
        cut:list[int] = [i for i in range(popsize)] #IDs of organisms that 'make the cut'

        for name in objectiveNames:
            minVal = min([population[i].getEvaluationScores({name:eval_funcs[name]})[name] for i in cut]) #get best w.r.t. this objective
            #keep only those organisms that are within epsilon of the best organism
            cut = [i for i in cut if population[i].getEvaluationScores({name:eval_funcs[name]})[name] <= minVal*(1+epsilon)]
            if len(cut) == 1:
                parents.append(population[cut[0]])
                break
        if len(cut) > 1:
            parents.append(getDiverseChoice([population[c] for c in cut])) #if choices remain after all objectives, choose diverse
    return parents


def run(config):
    global eval_obj
    eval_obj = Evaluation(config)
    popsize = config["popsize"]

    eval_funcs:dict[str:tuple[Callable,float]] = {}
    for eval_func_name, eval_func_params in config["eval_funcs"].items():
        if eval_func_name.endswith("distribution"):
            target = eval_obj.target_dist_dict[eval_func_name]
        else:
            target = eval_func_params["target"] if "target" in eval_func_params.keys() else 0
        eval_funcs[eval_func_name] = (eval_obj.functions[eval_func_name], target)

    population = [Organism(config["network_size"], config["network_sparsity"], config["weight_range"]) for _ in range(popsize)]
    fitnessLog = {x:[] for x in eval_funcs.keys()}

    for gen in range(config["num_generations"]*popsize):
        if gen%popsize == 0:
            print("Gen", gen//popsize)
            
            for func_name, funcPack in eval_funcs.items():
                func_fitnesses = [org.getEvaluationScores({func_name:funcPack})[func_name] for org in population]
                fitnessLog[func_name].append(mean(func_fitnesses))

        parents = epsilonLexicase(population, 2, popsize, eval_funcs, config["epsilon"])
        child = parents[0].makeCrossedCopyWith(parents[1], config["crossover_rate"], config["crossover_odds"])
        child = child.makeMutatedCopy(config["mutation_rate"], config["mutation_odds"])
        deathIndex = randint(0, popsize-1)
        population[deathIndex] = child

    #compute fitness for final population, for downstream analysis
    for func_name, funcPack in eval_funcs.items():
        func_fitnesses = [org.getEvaluationScores({func_name:funcPack})[func_name] for org in population]
        # fitnessLog[func_name].append(mean(func_fitnesses))

    return population, fitnessLog
