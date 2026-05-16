#pragma once

#include "instance.cpp"
#include "logger.cpp"

#include <limits>
#include <vector>

using namespace std;

AllocationResult chaitinBaseline(
    const Instance& instance,
    Logger& logger
) {
    vector<vector<int>> adjacency = buildAdjacencyList(instance);

    vector<int>  degree(instance.n);
    vector assignment(instance.n, 0);
    vector active(instance.n, true);
    vector spilled(instance.n, false);

    for (int i = 0; i < instance.n; i++)
        degree[i] = adjacency[i].size();

    vector<int> simplify_stack;

    int    remaining    = instance.n;
    int    iteration    = 0;
    double current_cost = 0.0;
    double best_cost    = 0.0;
    double score;

    while (remaining > 0) {
        int    chosen_node = -1;
        string note;

        for (int i = 0; i < instance.n; i++) {
            if (active[i] && degree[i] < instance.k) {
                chosen_node = i;
                break;
            }
        }

        if (chosen_node != -1) {
            simplify_stack.push_back(chosen_node);
            note = "simplify";
        } else {
            double best_score = numeric_limits<double>::max();
            for (int i = 0; i < instance.n; i++) {
                if (active[i]) {
                    if (degree[i] == 0)
                        score = numeric_limits<double>::max();
                    else
                        score = instance.weights[i] / static_cast<double>(degree[i]);;
                    if (score < best_score) {
                        best_score = score;
                        chosen_node = i;
                    }
                }
            }

            spilled[chosen_node] = true;
            current_cost += instance.weights[chosen_node];
            best_cost     = current_cost;
            note          = "spill_candidate";
        }

        active[chosen_node] = false;
        remaining--;

        for (int neighbor : adjacency[chosen_node])
            if (active[neighbor])
                degree[neighbor]--;

        logger.logIteration(iteration, current_cost, best_cost, note);
        iteration++;
    }

    for (int i = (int)simplify_stack.size() - 1; i >= 0; i--) {
        int node = simplify_stack[i];

        vector used_register(instance.k + 1, false);
        for (int neighbor : adjacency[node]) {
            int reg = assignment[neighbor];
            if (reg > 0) used_register[reg] = true;
        }

        int selected_register = 0;
        for (int reg = 1; reg <= instance.k; reg++) {
            if (!used_register[reg]) {
                selected_register = reg;
                break;
            }
        }

        assignment[node] = selected_register;

        if (selected_register == 0) {
            spilled[node]  = true;
            current_cost  += instance.weights[node];
            best_cost       = current_cost;
            logger.logIteration(iteration, current_cost, best_cost, "late_spill");
        } else {
            logger.logIteration(iteration, current_cost, best_cost, "color");
        }

        iteration++;
    }

    for (int i = 0; i < instance.n; i++)
        if (spilled[i]) assignment[i] = 0;

    AllocationResult result;
    result.assignment = assignment;
    result.cost       = calculateSpillCost(instance, assignment);
    return result;
}