#pragma once

#include "instance.cpp"
#include "logger.cpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

using namespace std;

struct Individual {
    vector<int> chromosome;
    double fitness = numeric_limits<double>::max();

    Individual() = default;
    Individual(vector<int> c, double f) : chromosome(move(c)), fitness(f) {}
    bool operator<(const Individual& o) const { return fitness < o.fitness; }
};

struct EAConfig {
    int    popSize           = 100;
    int    budget            = 50000;
    int    generations       = 500;
    double px                = 0.7;
    double pm                = 0.3;
    int    tournamentSize    = 3;
    int    eliteCount        = 2;
    double chaitinFraction   = 0.3;
    int    repairStrength    = 3;
    double smartBias         = 0.75;
    double immigrantFraction = 0.0;
    string mutationType      = "repair";   // repair | change | swap
    string crossoverType     = "smart";    // smart | uniform | onepoint
    int    seed              = 42;
};


void repairConflictsBySpilling(const Instance& inst, vector<int>& x) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto [u, v] : inst.edges) {
            if (x[u] != 0 && x[v] != 0 && x[u] == x[v]) {
                int victim = (inst.weights[u] <= inst.weights[v]) ? u : v;
                x[victim] = 0;
                changed = true;
            }
        }
    }
}

double evalRAObjective(const Instance& inst, const vector<int>& x) {
    vector<int> decoded = x;
    repairConflictsBySpilling(inst, decoded);
    return calculateSpillCost(inst, decoded);
}

bool isFeasible(const Instance& inst, const vector<int>& x) {
    for (auto [u, v] : inst.edges)
        if (x[u] != 0 && x[v] != 0 && x[u] == x[v])
            return false;
    return true;
}

int spillCount(const Instance& inst, const vector<int>& x) {
    int cnt = 0;
    for (int i = 0; i < inst.n; i++)
        if (x[i] == 0) cnt++;
    return cnt;
}


struct EAResult {
    vector<int> assignment;
    double cost;
    bool feasible;
    int spills;
    int evals;
};

EAResult evolutionaryAlgorithm(
    const Instance& inst,
    const vector<int>& chaitin_solution,
    Logger& logger,
    const EAConfig& cfg
) {
    mt19937 rng(cfg.seed);
    uniform_int_distribution    geneDist(0, inst.k);
    uniform_real_distribution coin(0.0, 1.0);

    int evals = 0;

    vector<vector<int>> adj = buildAdjacencyList(inst);


    auto makeRandom = [&]() {
        vector<int> c(inst.n);
        for (int i = 0; i < inst.n; i++) c[i] = geneDist(rng);
        return Individual(c, evalRAObjective(inst, c));
    };

    auto makeFromChaitin = [&]() {
        vector<int> c = chaitin_solution;
        vector<int> order(inst.n);
        for (int i = 0; i < inst.n; i++) order[i] = i;
        shuffle(order.begin(), order.end(), rng);

        int perturb = max(1, inst.n / 10);
        for (int i = 0; i < perturb; i++) {
            int v = order[i];
            if (coin(rng) < 0.7) {
                vector<int> freeRegisters;
                vector<bool> used(inst.k + 1, false);

                for (int nb : adj[v]) {
                    if (c[nb] > 0) {
                        used[c[nb]] = true;
                    }
                }

                for (int r = 1; r <= inst.k; r++) {
                    if (!used[r]) {
                        freeRegisters.push_back(r);
                    }
                }

                if (!freeRegisters.empty()) {
                    uniform_int_distribution<int> regDist(0, (int)freeRegisters.size() - 1);
                    c[v] = freeRegisters[regDist(rng)];
                } else {
                    c[v] = 0;
                }
            } else {
                c[v] = geneDist(rng);
            }
        }
        return Individual(c, evalRAObjective(inst, c));
    };

    vector<Individual> pop;
    pop.reserve(cfg.popSize);
    int chaitin_count = (int)(cfg.popSize * cfg.chaitinFraction);
    for (int i = 0; i < chaitin_count; i++)  pop.push_back(makeFromChaitin());
    while ((int)pop.size() < cfg.popSize)     pop.push_back(makeRandom());
    evals += cfg.popSize;

    Individual bestEver = *min_element(pop.begin(), pop.end());

    auto tournament = [&]() -> const Individual& {
        uniform_int_distribution<int> d(0, (int)pop.size() - 1);
        int best = d(rng);
        for (int i = 1; i < cfg.tournamentSize; i++) {
            int idx = d(rng);
            if (pop[idx].fitness < pop[best].fitness) best = idx;
        }
        return pop[best];
    };

    auto elite = [&](int count) {
        vector<Individual> sorted = pop;
        sort(sorted.begin(), sorted.end());
        sorted.resize(min(count, (int)sorted.size()));
        return sorted;
    };

    auto crossoverSmart = [&](const Individual& p1, const Individual& p2) {
        vector<int> c(inst.n);
        for (int i = 0; i < inst.n; i++) {
            int conf1 = 0, conf2 = 0;
            for (int nb : adj[i]) {
                if (p1.chromosome[nb] != 0 && p1.chromosome[nb] == p1.chromosome[i]) conf1++;
                if (p2.chromosome[nb] != 0 && p2.chromosome[nb] == p2.chromosome[i]) conf2++;
            }
            if (conf1 == conf2) {
                c[i] = (coin(rng) < 0.5) ? p1.chromosome[i] : p2.chromosome[i];
            } else if (conf1 < conf2) {
                c[i] = (coin(rng) < cfg.smartBias) ? p1.chromosome[i] : p2.chromosome[i];
            } else {
                c[i] = (coin(rng) < cfg.smartBias) ? p2.chromosome[i] : p1.chromosome[i];
            }
        }
        return Individual(c, numeric_limits<double>::max());
    };

    auto crossoverUniform = [&](const Individual& p1, const Individual& p2) {
        vector<int> c(inst.n);
        for (int i = 0; i < inst.n; i++)
            c[i] = (coin(rng) < 0.5) ? p1.chromosome[i] : p2.chromosome[i];
        return Individual(c, numeric_limits<double>::max());
    };

    auto crossoverOnePoint = [&](const Individual& p1, const Individual& p2) {
        uniform_int_distribution<int> d(1, inst.n - 1);
        int pt = d(rng);
        vector<int> c(inst.n);
        for (int i = 0;  i < pt;     i++) c[i] = p1.chromosome[i];
        for (int i = pt; i < inst.n; i++) c[i] = p2.chromosome[i];
        return Individual(c, numeric_limits<double>::max());
    };

    auto applyCrossover = [&](const Individual& p1, const Individual& p2) {
        if (coin(rng) >= cfg.px) return (coin(rng) < 0.5) ? p1 : p2;
        if (cfg.crossoverType == "uniform")   return crossoverUniform(p1, p2);
        if (cfg.crossoverType == "onepoint")  return crossoverOnePoint(p1, p2);
        return crossoverSmart(p1, p2);
    };

    auto mutationRepair = [&](Individual ind) {
        auto assignFreeRegister = [&](int vertex) {
            vector<int> freeRegisters;
            vector<bool> used(inst.k + 1, false);

            for (int nb : adj[vertex]) {
                if (ind.chromosome[nb] > 0) {
                    used[ind.chromosome[nb]] = true;
                }
            }

            for (int r = 1; r <= inst.k; r++) {
                if (!used[r]) {
                    freeRegisters.push_back(r);
                }
            }

            if (freeRegisters.empty()) {
                return false;
            }

            uniform_int_distribution<int> regDist(0, (int)freeRegisters.size() - 1);
            ind.chromosome[vertex] = freeRegisters[regDist(rng)];
            return true;
        };

        int repairs = max(1, cfg.repairStrength);
        for (int step = 0; step < repairs; step++) {
            vector<int> conflicted;
            for (auto [u, v] : inst.edges) {
                if (ind.chromosome[u] != 0 &&
                    ind.chromosome[v] != 0 &&
                    ind.chromosome[u] == ind.chromosome[v]) {
                    conflicted.push_back((inst.weights[u] < inst.weights[v]) ? u : v);
                }
            }

            if (!conflicted.empty()) {
                uniform_int_distribution<int> pick(0, (int)conflicted.size() - 1);
                int victim = conflicted[pick(rng)];
                if (!assignFreeRegister(victim)) {
                    ind.chromosome[victim] = 0;
                }
                continue;
            }

            vector<int> spilled;
            for (int i = 0; i < inst.n; i++) {
                if (ind.chromosome[i] == 0) spilled.push_back(i);
            }

            if (spilled.empty()) break;

            uniform_int_distribution<int> pick(0, (int)spilled.size() - 1);
            assignFreeRegister(spilled[pick(rng)]);
        }

        if (coin(rng) < 0.1) {
            uniform_int_distribution<int> pos(0, inst.n - 1);
            ind.chromosome[pos(rng)] = geneDist(rng);
        }

        ind.fitness = numeric_limits<double>::max();
        return ind;
    };

    auto mutationChange = [&](Individual ind) {
        uniform_int_distribution<int> pos(0, inst.n - 1);
        ind.chromosome[pos(rng)] = geneDist(rng);
        ind.fitness = numeric_limits<double>::max();
        return ind;
    };

    auto mutationSwap = [&](Individual ind) {
        uniform_int_distribution<int> d(0, inst.n - 1);
        int i = d(rng), j;
        do { j = d(rng); } while (j == i);
        swap(ind.chromosome[i], ind.chromosome[j]);
        ind.fitness = numeric_limits<double>::max();
        return ind;
    };

    auto applyMutation = [&](Individual ind) {
        if (coin(rng) >= cfg.pm) return ind;
        if (cfg.mutationType == "change") return mutationChange(move(ind));
        if (cfg.mutationType == "swap")   return mutationSwap(move(ind));
        return mutationRepair(move(ind));
    };

    auto computeStats = [&](const vector<Individual>& p) {
        double sum = 0.0;
        double worst = numeric_limits<double>::lowest();
        for (const auto& ind : p) {
            sum += ind.fitness;
            worst = max(worst, ind.fitness);
        }

        double avg = sum / (double)p.size();
        double sq = 0.0;
        for (const auto& ind : p) {
            sq += (ind.fitness - avg) * (ind.fitness - avg);
        }

        double stddev = sqrt(sq / (double)p.size());
        return tuple<double, double, double>(avg, stddev, worst);
    };

    int generation = 0;
    const int totalBudget = cfg.popSize * cfg.generations;

    auto [avg0, std0, worst0] = computeStats(pop);
    logger.logIteration(0, bestEver.fitness, worst0, bestEver.fitness, avg0, std0, "init");
    cout << "Gen 0"
         << " / " << (cfg.generations - 1)
         << " | popSize=" << cfg.popSize
         << " | best=" << bestEver.fitness
         << " | global=" << bestEver.fitness
         << " | evals=" << evals
         << " / " << totalBudget << "\n";

    while (generation + 1 < cfg.generations && evals < totalBudget) {
        generation++;

        vector<Individual> candidates = elite(cfg.eliteCount);
        candidates.reserve(cfg.eliteCount + cfg.popSize);

        int immigrants = min(
            cfg.popSize,
            max(0, (int)(cfg.popSize * cfg.immigrantFraction))
        );
        int offspringTarget = max(0, cfg.popSize - immigrants);

        int offspringCreated = 0;
        double currentGenerationBest = numeric_limits<double>::max();
        while (offspringCreated < offspringTarget && evals < totalBudget) {
            Individual offspring = applyCrossover(tournament(), tournament());
            offspring = applyMutation(move(offspring));
            offspring.fitness = evalRAObjective(inst, offspring.chromosome);
            evals++;
            offspringCreated++;

            currentGenerationBest = min(currentGenerationBest, offspring.fitness);

            if (offspring.fitness < bestEver.fitness)
                bestEver = offspring;

            candidates.push_back(move(offspring));
        }

        for (int i = 0; i < immigrants && evals < totalBudget; i++) {
            Individual immigrant = makeRandom();
            if (cfg.mutationType == "repair") {
                immigrant = mutationRepair(move(immigrant));
                immigrant.fitness = evalRAObjective(inst, immigrant.chromosome);
            }
            evals++;
            currentGenerationBest = min(currentGenerationBest, immigrant.fitness);

            if (immigrant.fitness < bestEver.fitness)
                bestEver = immigrant;

            candidates.push_back(move(immigrant));
        }

        sort(candidates.begin(), candidates.end());
        if ((int)candidates.size() > cfg.popSize) {
            candidates.resize(cfg.popSize);
        }
        pop = move(candidates);

        auto [avg, stddev, worst] = computeStats(pop);
        logger.logIteration(generation, currentGenerationBest, worst, bestEver.fitness, avg, stddev, "generation");

        if (generation <= 10 || generation % 10 == 0 || generation + 1 == cfg.generations)
            cout << "Gen " << generation
                 << " / " << (cfg.generations - 1)
                 << " | popSize=" << cfg.popSize
                 << " | current=" << currentGenerationBest
                 << " | global=" << bestEver.fitness
                 << " | evals=" << evals
                 << " / " << totalBudget << "\n";
    }

    vector<int> decodedBest = bestEver.chromosome;
    repairConflictsBySpilling(inst, decodedBest);

    EAResult result;
    result.assignment = decodedBest;
    result.cost       = calculateSpillCost(inst, decodedBest);
    result.feasible   = isFeasible(inst, decodedBest);
    result.spills     = spillCount(inst, decodedBest);
    result.evals      = evals;
    return result;
}
