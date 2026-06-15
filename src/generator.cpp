#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>

int    N            = 50;
int    K            = 8;
double DENSITY      = 0.3;
std::string GRAPH   = "interval";
unsigned int SEED   = 42;
std::string OUTPUT  = "data/instance.txt";


std::vector<std::pair<int,int>> generate_erdos_renyi(int n, double p, std::mt19937& rng) {
    std::vector<std::pair<int,int>> edges;
    std::uniform_real_distribution U(0.0, 1.0);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (U(rng) < p)
                edges.emplace_back(i, j);
    return edges;
}

std::vector<std::pair<int,int>> generate_interval(int n, double density, std::mt19937& rng) {
    double L = n / std::max(density, 0.05);
    std::uniform_real_distribution Upos(0.0, L);
    std::uniform_real_distribution Ulen(1.0, L * 0.2);

    std::vector<std::pair<double,double>> ivals(n);
    for (int i = 0; i < n; ++i) {
        double s = Upos(rng);
        ivals[i] = {s, s + Ulen(rng)};
    }

    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (!(ivals[i].second < ivals[j].first || ivals[j].second < ivals[i].first))
                edges.emplace_back(i, j);
    return edges;
}

std::vector<std::pair<int,int>> generate_chordal(int n, double density, std::mt19937& rng) {
    std::vector<std::vector<int>> adj(n);
    std::uniform_real_distribution U(0.0, 1.0);

    for (int i = 1; i < n; ++i) {
        std::uniform_int_distribution pick(0, i - 1);
        std::vector clique = { pick(rng) };

        std::vector<int> candidates = adj[clique[0]];
        while (!candidates.empty() && U(rng) < density) {
            std::uniform_int_distribution pc(0, (int)candidates.size() - 1);
            int chosen = candidates[pc(rng)];
            clique.push_back(chosen);
            std::vector<int> next;
            for (int c : candidates)
                if (c != chosen &&
                    std::find(adj[chosen].begin(), adj[chosen].end(), c) != adj[chosen].end())
                    next.push_back(c);
            candidates = next;
        }
        for (int v : clique) {
            adj[i].push_back(v);
            adj[v].push_back(i);
        }
    }

    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < n; ++i)
        for (int j : adj[i])
            if (j > i) edges.emplace_back(i, j);
    return edges;
}


std::vector<double> generate_weights(int n, std::mt19937& rng) {
    std::poisson_distribution    uses_dist(3.0);
    std::poisson_distribution    defs_dist(1.0);
    std::discrete_distribution   depth_dist({70, 20, 10}); // P(d=0,1,2)

    std::vector<double> w(n);
    for (int i = 0; i < n; ++i) {
        int uses = std::max(1, uses_dist(rng));
        int defs = std::max(1, defs_dist(rng));
        double weight = 0.0;
        for (int u = 0; u < uses; ++u) weight += std::pow(10.0, depth_dist(rng));
        for (int d = 0; d < defs; ++d) weight += std::pow(10.0, depth_dist(rng));
        w[i] = weight;
    }
    return w;
}

int main() {
    unsigned int seed = SEED
        ? SEED
        : (unsigned int)std::chrono::high_resolution_clock::now()
                           .time_since_epoch().count();
    std::mt19937 rng(seed);

    std::vector<std::pair<int,int>> edges;
    if      (GRAPH == "erdos")    edges = generate_erdos_renyi(N, DENSITY, rng);
    else if (GRAPH == "interval") edges = generate_interval   (N, DENSITY, rng);
    else                          edges = generate_chordal    (N, DENSITY, rng);

    std::vector<double> weights = generate_weights(N, rng);

    std::ofstream out(OUTPUT);
    out << N << " " << K << " " << edges.size() << "\n";
    for (int i = 0; i < N; ++i)
        out << weights[i] << (i + 1 == N ? "\n" : " ");
    for (auto [u, v] : edges)
        out << (u + 1) << " " << (v + 1) << "\n";

    std::cerr << "seed=" << seed << "  n=" << N << "  K=" << K
              << "  edges=" << edges.size() << "  graph=" << GRAPH << "\n";
    return 0;
}
