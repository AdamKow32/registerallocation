#include "src/chaitin.cpp"
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

int main() {
    fs::create_directories("logs");

    string instance_file = "data/instance.txt";

    Instance instance = readInstance(instance_file);

    RunConfig config;
    config.algorithm      = "chaitin_baseline";
    config.instance_name  = instance_file;
    config.output_dir     = "logs";
    config.n              = instance.n;
    config.k              = instance.k;
    config.m              = instance.m;
    config.seed           = 0;
    config.max_iterations = instance.n * 2;

    Logger logger(config);

    AllocationResult result = chaitinBaseline(instance, logger);

    string solution_file = "logs/" + logger.getRunId() + "_solution.csv";
    saveAllocation(instance, result.assignment, solution_file);

    logger.logFinal(result.cost, "done");

    cout << "Best cost:     " << result.cost       << "\n";
    cout << "Run ID:        " << logger.getRunId()  << "\n";
    cout << "Solution file: " << solution_file      << "\n";

    return 0;
}