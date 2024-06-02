#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "cub/cub.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#define PHANTOM_CHECK_CUDA(val) check((val), #val, __FILE__, __LINE__)
#define PHANTOM_CHECK_CUDA_LAST() checkLast(__FILE__, __LINE__)

template<typename T>
inline void check(T err, const char *const func, const char *const file,
                  const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

inline void checkLast(const char *const file, const int line) {
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

// A function to return a seeded random number generator.
inline std::mt19937 &generator() {
    // the generator will only be seeded once (per thread) since it's static
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

// A function to generate integers in the range [min, max]
inline int my_rand_int(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(generator());
}

inline void print_timer_banner() {
    // print header
    std::cout << "function,trials,median time (us),mean time (us),std. dev." << std::endl;
}

class CUDATimer {
public:
    explicit CUDATimer(std::string func_name) {
        func_name_ = std::move(func_name);

        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
    }

    ~CUDATimer() {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
        auto n_trials = time_.size();
        auto mean_time = mean(time_);
        auto min_time = min(time_);
        auto median_time = median(time_);
        auto stddev = std_dev(time_);
        std::cout << func_name_ << ","
                  << n_trials << ","
                  << median_time << ","
                  << mean_time << std::endl;
    }

    inline void start() const {
        cudaEventRecord(start_event_);
    }

    inline void stop() {
        cudaEventRecord(stop_event_);
        cudaEventSynchronize(stop_event_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event_, stop_event_);
        time_.push_back(milliseconds);
    }

private:
    std::string func_name_;
    cudaEvent_t start_event_{}, stop_event_{};
    std::vector<float> time_;

    static float mean(std::vector<float> const &v) {
        if (v.empty())
            return 0;

        auto const count = static_cast<float>(v.size());
        return std::reduce(v.begin(), v.end()) / count * 1000;
    }

    static float median(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v[size / 2] * 1000;
    }

    static float min(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v.front() * 1000;
    }

    static float max(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v.back() * 1000;
    }

    static double std_dev(std::vector<float> const &v) {
        if (v.empty())
            return 0;

        auto const count = static_cast<float>(v.size());
        float mean = std::reduce(v.begin(), v.end()) / count;

        std::vector<double> diff(v.size());

        std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        return std::sqrt(sq_sum / count);
    }
};

class ChronoTimer {
public:
    explicit ChronoTimer(std::string func_name) {
        func_name_ = std::move(func_name);
    }

    ~ChronoTimer() {
        auto n_trials = time_.size();
        auto mean_time = mean(time_);
        auto median_time = median(time_);
        auto min_time = min(time_);
        auto stddev = std_dev(time_);
        std::cout << func_name_ << ","
                  << n_trials << ","
                  << median_time << ","
                  << mean_time << std::endl;
    }

    inline void start() {
        start_point_ = std::chrono::steady_clock::now();
    }

    inline void stop() {
        stop_point_ = std::chrono::steady_clock::now();
        std::chrono::duration<float, std::micro> elapsed_time = stop_point_ - start_point_;
        time_.emplace_back(elapsed_time.count());
    }

private:
    std::string func_name_;

    std::chrono::time_point<std::chrono::steady_clock> start_point_, stop_point_;
    std::vector<float> time_;

    static float mean(std::vector<float> const &v) {
        if (v.empty())
            return 0;

        auto const count = static_cast<float>(v.size());
        return std::reduce(v.begin(), v.end()) / count;
    }

    static float median(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;
        else {
            sort(v.begin(), v.end());
            if (size % 2 == 0)
                return (v[size / 2 - 1] + v[size / 2]) / 2;
            else
                return v[size / 2];
        }
    }

    static float min(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v.front();
    }

    static float max(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v.back();
    }

    static double std_dev(std::vector<float> const &v) {
        if (v.empty())
            return 0;

        auto const count = static_cast<float>(v.size());
        float mean = std::reduce(v.begin(), v.end()) / count;

        std::vector<double> diff(v.size());

        std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        return std::sqrt(sq_sum / count);
    }
};
