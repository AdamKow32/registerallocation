#pragma once

#include "instance.cpp"
#include "logger.cpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

struct Individual {
    vector<int> chromosome;
    double fitness = numeric_limits<double>::max();

    Individual() = default;
    Individual(vector<int> c, double f) : chromosome(move(c)), fitness(f) {}

    bool operator<(const Individual& o) const { return fitness < o.fitness; }
};

struct EAConfig {
    int    popSize        = 50;
    int    budget         = 100000;
    double px             = 0.7;
    double pm             = 0.3;
    int    tournamentSize = 3;
    int    eliteCount     = 2;
    string mutationType   = "change";
    string crossoverType  = "uniform";
    int    seed           = 42;
};

double evalPenalized(const Instance& inst, const vector<int>& x, double M) {
    double cost = 0.0;
    for (int i = 0; i < inst.n; i++)
        if (x[i] == 0) cost += inst.weights[i];
    for (auto [u, v] : inst.edges)
        if (x[u] != 0 && x[v] != 0 && x[u] == x[v])
            cost += M;
    return cost;
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
    Logger& logger,
    const EAConfig& cfg
) {
    mt19937 rng(cfg.seed);
    uniform_int_distribution<int> geneDist(0, inst.k);
    uniform_real_distribution<double> coin(0.0, 1.0);

    double M = inst.n * (*max_element(inst.weights.begin(), inst.weights.end())) + 1.0;
    int evals = 0;

    auto makeRandom = [&]() {
        vector<int> c(inst.n);
        for (int i = 0; i < inst.n; i++) c[i] = geneDist(rng);
        return Individual(c, evalPenalized(inst, c, M));
    };

    vector<Individual> pop;
    pop.reserve(cfg.popSize);
    for (int i = 0; i < cfg.popSize; i++) pop.push_back(makeRandom());
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
        for (int i = 0; i < pt;     i++) c[i] = p1.chromosome[i];
        for (int i = pt; i < inst.n; i++) c[i] = p2.chromosome[i];
        return Individual(c, numeric_limits<double>::max());
    };

    auto applyCrossover = [&](const Individual& p1, const Individual& p2) {
        if (coin(rng) >= cfg.px) return (coin(rng) < 0.5) ? p1 : p2;
        if (cfg.crossoverType == "onepoint") return crossoverOnePoint(p1, p2);
        return crossoverUniform(p1, p2);
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
        if (cfg.mutationType == "swap") return mutationSwap(move(ind));
        return mutationChange(move(ind));
    };

    int generation = 0;
    logger.logIteration(0, bestEver.fitness, bestEver.fitness, "init");

    while (evals < cfg.budget) {
        generation++;
        vector<Individual> next = elite(cfg.eliteCount);

        while ((int)next.size() < cfg.popSize && evals < cfg.budget) {
            Individual offspring = applyCrossover(tournament(), tournament());
            offspring = applyMutation(move(offspring));
            offspring.fitness = evalPenalized(inst, offspring.chromosome, M);
            evals++;

            if (offspring.fitness < bestEver.fitness)
                bestEver = offspring;

            next.push_back(move(offspring));
        }

        while ((int)next.size() < cfg.popSize)
            next.push_back(elite(1)[0]);

        pop = move(next);

        double best = min_element(pop.begin(), pop.end())->fitness;
        logger.logIteration(generation, best, bestEver.fitness, "generation");

        if (generation % 10 == 0)
            cout << "Gen " << generation
                 << " | best=" << best
                 << " | global=" << bestEver.fitness
                 << " | evals=" << evals << "\n";
    }

    EAResult result;
    result.assignment = bestEver.chromosome;
    result.cost       = calculateSpillCost(inst, bestEver.chromosome);
    result.feasible   = isFeasible(inst, bestEver.chromosome);
    result.spills     = spillCount(inst, bestEver.chromosome);
    result.evals      = evals;
    return result;
}