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
def getDiverseChoice(organismList:list[Organism], objective_eval_funcs) -> list[float]:
    global eval_obj
    extremeOrganisms = []
    for funcName,eval_func in eval_obj.functions.items():
        if funcName not in objective_eval_funcs.keys():
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
            parents.append(getDiverseChoice([population[c] for c in cut], eval_funcs)) #if choices remain after all objectives, choose diverse
    return parents


def run(config):
    global eval_obj
    eval_obj = Evaluation(config)
    popsize = config["popsize"]

    eval_funcs:dict[str:tuple[Callable,float]] = {} #TODO: remove the dependancy on this object and refactor evaluationScores to take the config object
    for eval_func_name, eval_func_params in config["eval_funcs"].items():
        if eval_func_name.endswith("distribution"):
            target = eval_obj.target_dist_dict[eval_func_name]
        else:
            target = eval_func_params["target"] if "target" in eval_func_params.keys() else 0
        eval_funcs[eval_func_name] = (eval_obj.functions[eval_func_name], target)

    population = [Organism(config["network_size"], config["network_sparsity"], config["weight_range"]) for _ in range(popsize)]
    fitnessLog = {x:[] for x in eval_funcs.keys()}

    if config["scheme"] == "lexicase": ## use maran process flag
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

    elif config["scheme"] == "NSGAII": ## use nsga II flag
        #Algorithm from: Deb, Kalyanmoy, et al.
        #"A fast and elitist multiobjective genetic algorithm: NSGA-II."
        #IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
        #eval parents for initial fitness values
        for func_name, funcPack in eval_funcs.items():
            func_fitnesses = [org.getEvaluationScores({func_name:funcPack})[func_name] for org in population]
            fitnessLog[func_name].append(mean(func_fitnesses))
        #init some random children
        children = [Organism(config["network_size"], config["network_sparsity"], config["weight_range"]) for _ in range(popsize)]
        #evaluate all children
        for func_name, funcPack in eval_funcs.items():
            func_fitnesses = [org.getEvaluationScores({func_name:funcPack})[func_name] for org in children]

        for gen in range(config["num_generations"]):
            print("Gen:",gen)
            R = population+children
            F = fast_non_dominated_sort(R)
            P = []
            i = 1
            while len(P) + len(F[i]) <= popsize:
                crowding_distance_assignment(F[i])
                P.extend(F[i])
                i += 1
            if len(P) < popsize:
                crowding_distance_assignment(F[i])
                F[i].sort(key=lambda org: org.nsga_distance,reverse=True)
                P.extend(F[i][:popsize-len(P)])
            population = P
            parents = nsga_tournament(population,2*popsize)
            children = [parents[i].makeCrossedCopyWith(
                parents[i+popsize],config["crossover_rate"], config["crossover_odds"]).makeMutatedCopy(
                config["mutation_rate"], config["mutation_odds"]) for i in range(popsize)]
            #re-evaluate parents to log fitness (should be cheep due to score memo)
            for func_name, funcPack in eval_funcs.items():
                func_fitnesses = [org.getEvaluationScores({func_name:funcPack})[func_name] for org in population]
                fitnessLog[func_name].append(mean(func_fitnesses))
            #evaluate all children
            for func_name, funcPack in eval_funcs.items():
                func_fitnesses = [org.getEvaluationScores({func_name:funcPack})[func_name] for org in children]

    else:
        print("Please give a valid selection scheme: lexicase or NSGAII")
        exit()

    return population, fitnessLog


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


#Algorithm from: Deb, Kalyanmoy, et al.
#"A fast and elitist multiobjective genetic algorithm: NSGA-II."
#IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
def crowding_distance_assignment(I):
    l = len(I)
    if l == 0: return []
    for i in I:
        i.nsga_distance = 0
    for m in I[0].evaluationScores.keys():
        I.sort(key=lambda org: org.evaluationScores[m])
        I[0].nsga_distance = float("inf")
        I[-1].nsga_distance = float('inf')
        rng = I[-1].evaluationScores[m]-I[0].evaluationScores[m]
        if rng == 0: continue
        for i in range(1,l-2):
            I[i].nsga_distance += (I[i+1].evaluationScores[m]-I[i-1].evaluationScores[m])/rng


#Algorithm from: Deb, Kalyanmoy, et al.
#"A fast and elitist multiobjective genetic algorithm: NSGA-II."
#IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
def nsga_tournament(population,numOffspring):
    parents = []
    for _ in range(numOffspring):
        choices = sample(population,k=2)
        if choices[0].nsga_rank > choices[1].nsga_rank:
            parents.append(choices[1])
        elif choices[1].nsga_rank > choices[0].nsga_rank:
            parents.append(choices[0])
        elif choices[0].nsga_distance < choices[1].nsga_distance:
            parents.append(choices[1])
        elif choices[1].nsga_distance < choices[0].nsga_distance:
            parents.append(choices[0])
        else:
            parents.append(sample(choices,k=1)[0])
    return parents