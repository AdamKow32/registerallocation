#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

struct Instance {
    int n;
    int k;
    int m;

    vector<double> weights;          // double (nie int)
    vector<pair<int, int>> edges;    // 0-indexed
};

struct AllocationResult {
    vector<int> assignment;
    double cost;
};

Instance readInstance(const string& filename) {
    ifstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open instance file");
    }

    Instance instance;
    file >> instance.n >> instance.k >> instance.m;

    instance.weights.resize(instance.n);
    for (int i = 0; i < instance.n; i++)
        file >> instance.weights[i];

    instance.edges.resize(instance.m);
    for (int i = 0; i < instance.m; i++) {
        file >> instance.edges[i].first >> instance.edges[i].second;
        instance.edges[i].first--;
        instance.edges[i].second--;
    }

    return instance;
}

vector<vector<int>> buildAdjacencyList(const Instance& instance) {
    vector<vector<int>> adjacency(instance.n);

    for (auto [u, v] : instance.edges) {
        adjacency[u].push_back(v);
        adjacency[v].push_back(u);
    }

    return adjacency;
}

double calculateSpillCost(
    const Instance& instance,
    const vector<int>& assignment
) {
    double cost = 0.0;
    for (int i = 0; i < instance.n; i++)
        if (assignment[i] == 0)
            cost += instance.weights[i];
    return cost;
}

void saveAllocation(
    const Instance& instance,
    const vector<int>& assignment,
    const string& filename
) {
    ofstream file(filename);
    file << "variable;register;weight\n";
    for (int i = 0; i < instance.n; i++) {
        file << i + 1 << ";"
             << assignment[i] << ";"
             << instance.weights[i] << "\n";
    }
}