from random import randint, sample, shuffle
from statistics import mean
from typing import Callable

from eval_functions import Evaluation
from organism import Organism


#transpose of a matrix (list-of-list)
def T(LL:list[list]) -> list[list]:
    return list(zip(*LL))


def epsilonLexicase(population:list[Organism], numParents:int, popsize:int, eval_funcs:dict, epsilon:float = 0.05) -> list[Organism]:
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
            parents.append(population[sample(cut,k=1)[0]]) #if choices remain after all objectives, choose randomly

    return parents


def run(config):
    eval = Evaluation()
    popsize = config["popsize"]

    eval_funcs:dict[str:tuple[Callable,float]] = {}
    for eval_func_name, ideal_val in config["eval_funcs"].items():
        eval_funcs[eval_func_name] = (getattr(eval, eval_func_name), ideal_val)

    population = [Organism(config["network_size"], config["network_sparsity"]) for _ in range(popsize)]
    fitnessLog = {x:[] for x in eval_funcs.keys()}

    for gen in range(config["num_generations"]*popsize):
        if gen%popsize == 0:
            print("Gen", gen//popsize)
            
            for func_name, funcPack in eval_funcs.items():
                func_fitnesses = [org.getEvaluationScores({func_name:funcPack})[func_name] for org in population]
                fitnessLog[func_name].append(mean(func_fitnesses))

        parents = epsilonLexicase(population, 2, popsize, eval_funcs)
        child = parents[0].makeCrossedCopyWith(parents[1], config["crossover_rate"]).makeMutatedCopy(config["mutation_rate"])
        deathIndex = randint(0, popsize-1)
        population[deathIndex] = child

    return population, fitnessLog
