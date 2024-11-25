#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <omp.h>

#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Matrix = std::vector<float>;
using Batch = std::vector<Matrix>;
using d_Batch = std::vector<float*>;

Matrix generateRandomMatrix(const size_t matrix_size) {
    static std::random_device rd;
    static std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    Matrix matrix;
    matrix.reserve(matrix_size * matrix_size);

    for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
        matrix.emplace_back(dis(rng));
    }

    return matrix;
}

Batch generateRandomBatch(const size_t batch_size, const size_t matrix_size) {
    Batch batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        batch.emplace_back(generateRandomMatrix(matrix_size));
    }

    return batch;
}

Batch convLayerSeq(const Batch &input_batch, const Matrix &filter,
                   const size_t n, const size_t k) {
    const auto c = n - k + 1;

    Batch output_batch;
    output_batch.reserve(input_batch.size());

    for (const Matrix &input_matrix : input_batch) {
        Matrix output_matrix;
        output_matrix.reserve(c * c);

        for (size_t i = 0; i < c; ++i) {
            for (size_t j = 0; j < c; ++j) {
                float val = 0.0f;
                for (size_t e = 0; e < k; ++e) {
                    for (size_t f = 0; f < k; ++f) {
                        val += filter[e * k + f] *
                               input_matrix[(i + e) * n + j + f];
                    }
                }
                output_matrix.emplace_back(val);
            }
        }

        output_batch.emplace_back(std::move(output_matrix));
    }

    return output_batch;
}

Batch convLayerPar(const Batch &input_batch, const Matrix &filter,
                   const size_t n, const size_t k) {
    Batch output_batch(input_batch.size());
    const auto c = n - k + 1;

#pragma omp parallel for
    for (size_t b = 0; b < output_batch.size(); ++b) {
        const Matrix &input_matrix = input_batch[b];
        Matrix output_matrix;
        output_matrix.reserve(c * c);

        for (size_t i = 0; i < c; ++i) {
            for (size_t j = 0; j < c; ++j) {
                float val = 0.0f;
                for (size_t e = 0; e < k; ++e) {
                    for (size_t f = 0; f < k; ++f) {
                        val += filter[e * k + f] *
                               input_matrix[(i + e) * n + j + f];
                    }
                }
                output_matrix.emplace_back(val);
            }
        }

        output_batch[b] = std::move(output_matrix);
    }

    return output_batch;
}

d_Batch getCudaBatch(const Batch &batch) {
    d_Batch d_batch;
    d_batch.reserve(batch.size());
    const auto matrix_size = batch[0].size() * sizeof(float);

    for (const Matrix &matrix : batch) {
        float *d_matrix = nullptr;
        cudaMalloc((void**)&d_matrix, matrix_size);
        cudaMemcpy((void*)d_matrix, (void*)matrix.data(), matrix_size, cudaMemcpyHostToDevice);
        d_batch.emplace_back(d_matrix);
    }

    return d_batch;
}

Batch convLayerCUDA(const Batch &input_batch, const Matrix &filter, const size_t n, const size_t k) {
    d_Batch d_input_batch = getCudaBatch(input_batch);

    
}

float maxDifference(const Batch &batch1, const Batch &batch2) {
    float max_difference = 0.0f;

    for (size_t i = 0; i < batch1.size(); ++i) {
        const Matrix &m1 = batch1[i];
        const Matrix &m2 = batch2[i];

        for (size_t j = 0; j < m1.size(); ++j) {
            const float difference = std::abs(m1[j] - m2[j]);
            if (difference > max_difference) {
                max_difference = difference;
            }
        }
    }

    return max_difference;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix_size> <filter_size> <batch_size>" << std::endl;
        return 1;
    }
    const auto n = std::stoull(argv[1]);
    const auto k = std::stoull(argv[2]);
    const auto m = std::stoull(argv[3]);

    try {
        const Batch batch = generateRandomBatch(m, n);
        const Matrix filter = generateRandomMatrix(k);

        const double seq_start = omp_get_wtime();
        const Batch seq_result = convLayerSeq(batch, filter, n, k);
        const double seq_end = omp_get_wtime();
        std::cout << "Sequential time: " << seq_end - seq_start << std::endl;

        {
            const double par_start = omp_get_wtime();
            const Batch par_result = convLayerPar(batch, filter, n, k);
            const double par_end = omp_get_wtime();
            std::cout << "Parallel time: " << par_end - par_start << " ";
            std::cout << "(diff: " << maxDifference(seq_result, par_result)
                      << ")" << std::endl;
        }
    } catch (const std::bad_alloc &e) {
        std::cerr << "Error alloc memory: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
 