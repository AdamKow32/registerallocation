#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std;

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
        summary_path    = config.output_dir + "/summary.csv";

        iterations_file.open(iterations_path);
        iterations_file
            << "run_id"          << separator
            << "algorithm"       << separator
            << "instance"        << separator
            << "n"               << separator
            << "k"               << separator
            << "m"               << separator
            << "seed"            << separator
            << "max_iterations"  << separator
            << "iteration"       << separator
            << "time_ms"         << separator
            << "current_cost"    << separator
            << "best_cost"       << separator
            << "note\n";

        bool summary_exists = filesystem::exists(summary_path);
        summary_file.open(summary_path, ios::app);

        if (!summary_exists) {
            summary_file
                << "run_id"         << separator
                << "algorithm"      << separator
                << "instance"       << separator
                << "n"              << separator
                << "k"              << separator
                << "m"              << separator
                << "seed"           << separator
                << "max_iterations" << separator
                << "time_ms"        << separator
                << "best_cost"      << separator
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
            << run_id               << separator
            << config.algorithm     << separator
            << config.instance_name << separator
            << config.n             << separator
            << config.k             << separator
            << config.m             << separator
            << config.seed          << separator
            << config.max_iterations<< separator
            << iteration            << separator
            << elapsedMs()          << separator
            << current_cost         << separator
            << best_cost            << separator
            << note << "\n";
    }

    void logFinal(double best_cost, const string& note) {
        summary_file
            << run_id                << separator
            << config.algorithm      << separator
            << config.instance_name  << separator
            << config.n              << separator
            << config.k              << separator
            << config.m              << separator
            << config.seed           << separator
            << config.max_iterations << separator
            << elapsedMs()           << separator
            << best_cost             << separator
            << note << "\n";

        iterations_file.flush();
        summary_file.flush();
    }

    string getRunId() const { return run_id; }

private:
    long long elapsedMs() const {
        return chrono::duration_cast<chrono::milliseconds>(
            Clock::now() - start_time
        ).count();
    }

    string createRunId(const string& algorithm) {
        auto now  = chrono::system_clock::now();
        auto time = chrono::system_clock::to_time_t(now);

        tm tm_data{};
#ifdef _WIN32
        localtime_s(&tm_data, &time);
#else
        localtime_r(&time, &tm_data);
#endif
        stringstream ss;
        ss << put_time(&tm_data, "%Y%m%d_%H%M%S") << "_" << algorithm;
        return ss.str();
    }
};