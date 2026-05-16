#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

struct Instance {
    int n;
    int k;
    int m;

    vector<int> weights;
    vector<pair<int, int>> edges;
};

struct RunConfig {
    string algorithm;
    string instance_name;
    string output_dir;

    int n;
    int k;
    int m;

    int seed;
    int max_iterations;
};

class Logger {
private:
    using Clock = chrono::steady_clock;

    RunConfig config;
    Clock::time_point start_time;

    ofstream iterations_file;
    ofstream summary_file;

    string run_id;
    string iterations_path;
    string summary_path;

    char separator = ';';

public:
    Logger(const RunConfig& cfg) {
        config = cfg;
        start_time = Clock::now();

        filesystem::create_directories(config.output_dir);

        run_id = createRunId(config.algorithm);

        iterations_path = config.output_dir + "/" + run_id + "_iterations.csv";
        summary_path = config.output_dir + "/summary.csv";

        iterations_file.open(iterations_path);

        iterations_file
            << "run_id" << separator
            << "algorithm" << separator
            << "instance" << separator
            << "n" << separator
            << "k" << separator
            << "m" << separator
            << "seed" << separator
            << "max_iterations" << separator
            << "iteration" << separator
            << "time_ms" << separator
            << "current_cost" << separator
            << "best_cost" << separator
            << "note\n";

        bool summary_exists = filesystem::exists(summary_path);

        summary_file.open(summary_path, ios::app);

        if (!summary_exists) {
            summary_file
                << "run_id" << separator
                << "algorithm" << separator
                << "instance" << separator
                << "n" << separator
                << "k" << separator
                << "m" << separator
                << "seed" << separator
                << "max_iterations" << separator
                << "time_ms" << separator
                << "best_cost" << separator
                << "note\n";
        }
    }

    void logIteration(
        int iteration,
        double current_cost,
        double best_cost,
        const string& note
    ) {
        iterations_file
            << run_id << separator
            << config.algorithm << separator
            << config.instance_name << separator
            << config.n << separator
            << config.k << separator
            << config.m << separator
            << config.seed << separator
            << config.max_iterations << separator
            << iteration << separator
            << elapsedMs() << separator
            << current_cost << separator
            << best_cost << separator
            << note << "\n";
    }

    void logFinal(
        double best_cost,
        const string& note
    ) {
        summary_file
            << run_id << separator
            << config.algorithm << separator
            << config.instance_name << separator
            << config.n << separator
            << config.k << separator
            << config.m << separator
            << config.seed << separator
            << config.max_iterations << separator
            << elapsedMs() << separator
            << best_cost << separator
            << note << "\n";

        iterations_file.flush();
        summary_file.flush();
    }

    string getRunId() const {
        return run_id;
    }

private:
    long long elapsedMs() const {
        return chrono::duration_cast<chrono::milliseconds>(
            Clock::now() - start_time
        ).count();
    }

    string createRunId(const string& algorithm) {
        auto now = chrono::system_clock::now();
        auto time = chrono::system_clock::to_time_t(now);

        tm tm_data{};

#ifdef _WIN32
        localtime_s(&tm_data, &time);
#else
        localtime_r(&time, &tm_data);
#endif

        stringstream ss;

        ss
            << put_time(&tm_data, "%Y%m%d_%H%M%S")
            << "_"
            << algorithm;

        return ss.str();
    }
};

struct AllocationResult {
    vector<int> assignment;
    double cost;
};

Instance readInstance(const string& filename) {
    ifstream file(filename);

    Instance instance;

    file >> instance.n >> instance.k >> instance.m;

    instance.weights.resize(instance.n);

    for (int i = 0; i < instance.n; i++) {
        file >> instance.weights[i];
    }

    instance.edges.resize(instance.m);

    for (int i = 0; i < instance.m; i++) {
        file >> instance.edges[i].first >> instance.edges[i].second;
    }

    return instance;
}

vector<vector<int>> buildAdjacencyList(const Instance& instance) {
    vector<vector<int>> adjacency(instance.n);

    for (auto edge : instance.edges) {
        int u = edge.first - 1;
        int v = edge.second - 1;

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

    for (int i = 0; i < instance.n; i++) {
        if (assignment[i] == 0) {
            cost += instance.weights[i];
        }
    }

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
        file
            << i + 1 << ";"
            << assignment[i] << ";"
            << instance.weights[i] << "\n";
    }
}

AllocationResult chaitinBaseline(
    const Instance& instance,
    Logger& logger
) {
    vector<vector<int>> adjacency = buildAdjacencyList(instance);

    vector<int> degree(instance.n);

    for (int i = 0; i < instance.n; i++) {
        degree[i] = adjacency[i].size();
    }

    vector<int> simplify_stack;
    vector<int> assignment(instance.n, 0);

    vector<bool> active(instance.n, true);
    vector<bool> spilled(instance.n, false);

    int remaining = instance.n;
    int iteration = 0;

    double current_cost = 0.0;
    double best_cost = 0.0;

    while (remaining > 0) {
        int chosen_node = -1;
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
                    double score =
                        static_cast<double>(instance.weights[i]) /
                        static_cast<double>(degree[i]);

                    if (score < best_score) {
                        best_score = score;
                        chosen_node = i;
                    }
                }
            }

            spilled[chosen_node] = true;
            current_cost += instance.weights[chosen_node];
            best_cost = current_cost;

            note = "spill_candidate";
        }

        active[chosen_node] = false;
        remaining--;

        for (int neighbor : adjacency[chosen_node]) {
            if (active[neighbor]) {
                degree[neighbor]--;
            }
        }

        logger.logIteration(
            iteration,
            current_cost,
            best_cost,
            note
        );

        iteration++;
    }

    for (int i = static_cast<int>(simplify_stack.size()) - 1; i >= 0; i--) {
        int node = simplify_stack[i];

        vector<bool> used_register(instance.k + 1, false);

        for (int neighbor : adjacency[node]) {
            int neighbor_register = assignment[neighbor];

            if (neighbor_register > 0) {
                used_register[neighbor_register] = true;
            }
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
            spilled[node] = true;
            current_cost += instance.weights[node];
            best_cost = current_cost;

            logger.logIteration(
                iteration,
                current_cost,
                best_cost,
                "late_spill"
            );
        } else {
            logger.logIteration(
                iteration,
                current_cost,
                best_cost,
                "color"
            );
        }

        iteration++;
    }

    for (int i = 0; i < instance.n; i++) {
        if (spilled[i]) {
            assignment[i] = 0;
        }
    }

    AllocationResult result;

    result.assignment = assignment;
    result.cost = calculateSpillCost(instance, assignment);

    return result;
}

int main() {
    string instance_file = "instance.txt";

    Instance instance = readInstance(instance_file);

    RunConfig config;

    config.algorithm = "chaitin_baseline";
    config.instance_name = instance_file;
    config.output_dir = "logs";

    config.n = instance.n;
    config.k = instance.k;
    config.m = instance.m;

    config.seed = 0;
    config.max_iterations = instance.n * 2;

    Logger logger(config);

    AllocationResult result = chaitinBaseline(instance, logger);

    string solution_file =
        config.output_dir + "/" + logger.getRunId() + "_solution.csv";

    saveAllocation(
        instance,
        result.assignment,
        solution_file
    );

    logger.logFinal(
        result.cost,
        "koniec chaitin baseline"
    );

    cout << "Chaitin baseline zakonczony" << endl;
    cout << "Best cost: " << result.cost << endl;
    cout << "Run ID: " << logger.getRunId() << endl;
    cout << "Solution file: " << solution_file << endl;

    return 0;
}