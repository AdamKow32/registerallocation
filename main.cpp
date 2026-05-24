#include "src/chaitin.cpp"
#include "src/ea.cpp"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;

int main() {
    fs::create_directories("logs");

    string instance_file = "data/instance.txt";
    Instance instance = readInstance(instance_file);

    cout << "Instance: n=" << instance.n
         << " k=" << instance.k
         << " m=" << instance.m << "\n\n";

    {
        RunConfig cfg;
        cfg.algorithm      = "chaitin_baseline";
        cfg.instance_name  = instance_file;
        cfg.output_dir     = "logs";
        cfg.n = instance.n; cfg.k = instance.k; cfg.m = instance.m;
        cfg.seed           = 0;
        cfg.max_iterations = instance.n * 2;

        Logger logger(cfg);
        AllocationResult result = chaitinBaseline(instance, logger);

        string sol_file = "logs/" + logger.getRunId() + "_solution.csv";
        saveAllocation(instance, result.assignment, sol_file);
        logger.logFinal(result.cost, "done");

        cout << "[Chaitin] cost=" << result.cost
             << "  run_id=" << logger.getRunId() << "\n\n";
    }

    {
        EAConfig eaCfg;
        eaCfg.popSize        = 200;
        eaCfg.budget         = 100000;
        eaCfg.px             = 0.7;
        eaCfg.pm             = 0.4;
        eaCfg.tournamentSize = 2;
        eaCfg.eliteCount     = 1;
        eaCfg.mutationType   = "change";
        eaCfg.crossoverType  = "uniform";
        eaCfg.seed           = 42;

        RunConfig cfg;
        cfg.algorithm      = "ea";
        cfg.instance_name  = instance_file;
        cfg.output_dir     = "logs";
        cfg.n = instance.n; cfg.k = instance.k; cfg.m = instance.m;
        cfg.seed           = eaCfg.seed;
        cfg.max_iterations = eaCfg.budget;

        Logger logger(cfg);
        EAResult result = evolutionaryAlgorithm(instance, logger, eaCfg);

        string sol_file = "logs/" + logger.getRunId() + "_solution.csv";
        saveAllocation(instance, result.assignment, sol_file);
        logger.logFinal(result.cost, "done");

        cout << "\n[EA] cost="     << result.cost
             << "  spills="        << result.spills
             << "  feasible="      << (result.feasible ? "yes" : "NO")
             << "  evals="         << result.evals
             << "  run_id="        << logger.getRunId() << "\n";
    }

    return 0;
}