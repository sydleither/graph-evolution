from random import sample, shuffle, randint
from statistics import mean
from typing import Callable

import matplotlib.pyplot as plt

from eval_functions import Evaluation
from organism import Organism


#transpose of a matrix (list-of-list)
def T(LL:list[list]) -> list[list]:
    return list(zip(*LL))


def epsilonLexicase(population:list[Organism], numParents:int, epsilon:float = 0.05) -> list[Organism]:
    global EVAL_FUNCS, POPSIZE

    parents:list[Organism] = []
    objectiveNames:list[str] = list(EVAL_FUNCS.keys())

    for _ in range(numParents):
        shuffle(objectiveNames) #randomize objective evaluation order
        cut:list[int] = [i for i in range(POPSIZE)] #IDs of organisms that 'make the cut'

        for name in objectiveNames:
            minVal = min([population[i].getEvaluationScores({name:EVAL_FUNCS[name]})[name] for i in cut]) #get best w.r.t. this objective
            #keep only those organisms that are within epsilon of the best organism
            cut = [i for i in cut if population[i].getEvaluationScores({name:EVAL_FUNCS[name]})[name] <= minVal*(1+epsilon)]
            if len(cut) == 1:
                parents.append(population[cut[0]])
                break
        parents.append(population[sample(cut,k=1)[0]]) #if choices remain after all objectives, choose randomly

    return parents


if __name__ == '__main__':
    eval = Evaluation()

    POPSIZE = 100
    MUTATION_RATE = 0.005
    CROSSOVER_RATE = 0.2
    NETWORK_SIZE = 10
    NETWORK_SPARSITY = 0.1
    NUM_GENERATIONS = 100
    EVAL_FUNCS:dict[str:tuple[Callable,float]] = {
        "connectance":(eval.connectance, 0.5), 
        "positive_interactions_proportion":(eval.positive_interactions_proportion, 0.5),
        "average_positive_interactions_strength":(eval.average_positive_interactions_strength, 0.25),
        "number_of_mutualistic_pairs":(eval.number_of_mutualistic_pairs, 4),
        "number_of_competiton_pairs":(eval.number_of_competiton_pairs, 2),
        "proportion_of_self_loops_positive":(eval.proportion_of_self_loops_positive, 0.25)
    }
    eval_funcs_names = EVAL_FUNCS.keys()

    population = [Organism(NETWORK_SIZE, NETWORK_SPARSITY) for _ in range(POPSIZE)]
    fitnessLog = {x:[] for x in eval_funcs_names}
    for gen in range(NUM_GENERATIONS*POPSIZE):
        if gen%POPSIZE == 0:
            print("Gen",gen//POPSIZE)

        parents = epsilonLexicase(population,2)
        child = parents[0].makeCrossedCopyWith(parents[1],CROSSOVER_RATE).makeMutatedCopy(MUTATION_RATE)
        
        if gen%POPSIZE == 0:
            for func_name, funcPack in EVAL_FUNCS.items():
                func_fitnesses = [org.getEvaluationScores({func_name:funcPack})[func_name] for org in population]
                fitnessLog[func_name].append(mean(func_fitnesses))
        
        deathIndex = randint(0,POPSIZE-1)
        population[deathIndex] = child

    population[0].saveGraphFigure("testFigure.png")

    figure, axis = plt.subplots(1, 2, figsize=(10,6))
    for func_name in eval_funcs_names:
        axis[0].plot(fitnessLog[func_name], label=func_name)
        axis[1].plot(fitnessLog[func_name])
    axis[1].set_yscale("log")
    figure.supxlabel("Generations")
    figure.supylabel("MSE")
    figure.legend()
    plt.savefig("fitness.png")
    plt.close()
