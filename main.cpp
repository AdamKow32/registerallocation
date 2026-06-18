#include "src/chaitin.cpp"
#include "src/EA.cpp"

#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

int main() {
    fs::create_directories("logs");

    string instance_file = "data/instance.txt";
    Instance instance = readInstance(instance_file);

    cout << "Instance: n=" << instance.n
         << " k=" << instance.k
         << " m=" << instance.m << "\n\n";

    AllocationResult chaitin_result;
    {
        RunConfig cfg;
        cfg.algorithm      = "chaitin_baseline";
        cfg.instance_name  = instance_file;
        cfg.output_dir     = "logs";
        cfg.n = instance.n;
        cfg.k = instance.k;
        cfg.m = instance.m;
        cfg.seed           = 0;
        cfg.max_iterations = instance.n * 2;

        Logger logger(cfg);
        chaitin_result = chaitinBaseline(instance, logger);

        string sol_file = "logs/" + logger.getRunId() + "_solution.csv";
        saveAllocation(instance, chaitin_result.assignment, sol_file);
        logger.logFinal(chaitin_result.cost, "done");

        cout << "[Chaitin] cost=" << chaitin_result.cost
             << "  run_id=" << logger.getRunId() << "\n\n";
    }

    {
        EAConfig eaCfg;
        eaCfg.popSize         = 200;
        eaCfg.budget          = 50000;
        eaCfg.px              = 0.7;
        eaCfg.pm              = 0.3;
        eaCfg.tournamentSize  = 2;
        eaCfg.eliteCount      = 2;
        eaCfg.chaitinFraction = 0.0;
        eaCfg.mutationType    = "change";
        eaCfg.crossoverType   = "uniform";
        eaCfg.seed            = 42;

        RunConfig cfg;
        cfg.algorithm      = "ea_general";
        cfg.instance_name  = instance_file;
        cfg.output_dir     = "logs";
        cfg.n = instance.n;
        cfg.k = instance.k;
        cfg.m = instance.m;
        cfg.seed           = eaCfg.seed;
        cfg.max_iterations = eaCfg.budget;

        Logger logger(cfg);

        EAResult result = evolutionaryAlgorithm(
            instance,
            chaitin_result.assignment,
            logger,
            eaCfg
        );

        string sol_file = "logs/" + logger.getRunId() + "_solution.csv";
        saveAllocation(instance, result.assignment, sol_file);
        logger.logFinal(result.cost, "done");

        cout << "\n[EA general] cost=" << result.cost
             << "  spills="            << result.spills
             << "  feasible="          << (result.feasible ? "yes" : "NO")
             << "  evals="             << result.evals
             << "  run_id="            << logger.getRunId() << "\n";
    }

    {
        EAConfig eaCfg;
        eaCfg.popSize         = 50;
        eaCfg.budget          = 50000;
        eaCfg.px              = 0.9;
        eaCfg.pm              = 0.3;
        eaCfg.tournamentSize  = 3;
        eaCfg.eliteCount      = 2;
        eaCfg.chaitinFraction = 0.3;
        eaCfg.mutationType    = "repair";
        eaCfg.crossoverType   = "smart";
        eaCfg.seed            = 42;

        RunConfig cfg;
        cfg.algorithm      = "ea_personalized";
        cfg.instance_name  = instance_file;
        cfg.output_dir     = "logs";
        cfg.n = instance.n;
        cfg.k = instance.k;
        cfg.m = instance.m;
        cfg.seed           = eaCfg.seed;
        cfg.max_iterations = eaCfg.budget;

        Logger logger(cfg);

        EAResult result = evolutionaryAlgorithm(
            instance,
            chaitin_result.assignment,
            logger,
            eaCfg
        );

        string sol_file = "logs/" + logger.getRunId() + "_solution.csv";
        saveAllocation(instance, result.assignment, sol_file);
        logger.logFinal(result.cost, "done");

        cout << "\n[EA personalized] cost=" << result.cost
             << "  spills="                  << result.spills
             << "  feasible="                << (result.feasible ? "yes" : "NO")
             << "  evals="                   << result.evals
             << "  run_id="                  << logger.getRunId() << "\n";
    }

    return 0;
}