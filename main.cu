#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

using Matrix = std::vector<float>;
using Batch = std::vector<Matrix>;
using d_Batch = std::vector<float *>;

Matrix generateRandomMatrix(const size_t n) {
  static std::random_device rd;
  static std::mt19937 rng(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  Matrix matrix;
  matrix.reserve(n * n);

  for (size_t i = 0; i < n * n; ++i) {
    // matrix.emplace_back(dis(rng));
    matrix.emplace_back(1);
  }

  return matrix;
}

Batch generateRandomBatch(const size_t m, const size_t n) {
  Batch batch;
  batch.reserve(m);

  for (size_t i = 0; i < m; ++i) {
    batch.emplace_back(generateRandomMatrix(n));
  }

  return batch;
}

Batch convLayerSeq(const Batch &input_batch, const Matrix &filter,
                   const size_t m, const size_t n, const size_t k) {
  const auto c = n - k + 1;

  Batch output_batch;
  output_batch.reserve(m);

  for (const auto &input_matrix : input_batch) {
    Matrix output_matrix;
    output_matrix.reserve(c * c);

    for (size_t i = 0; i < c; ++i) {
      for (size_t j = 0; j < c; ++j) {
        float val = 0.0f;
        for (size_t e = 0; e < k; ++e) {
          for (size_t f = 0; f < k; ++f) {
            val += filter[e * k + f] * input_matrix[(i + e) * n + j + f];
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
                   const size_t m, const size_t n, const size_t k) {
  const auto c = n - k + 1;
  Batch output_batch(m);

#pragma omp parallel for
  for (int b = 0; b < m; ++b) {
    const auto &input_matrix = input_batch[b];
    Matrix output_matrix;
    output_matrix.reserve(c * c);

    for (size_t i = 0; i < c; ++i) {
      for (size_t j = 0; j < c; ++j) {
        float val = 0.0f;
        for (size_t e = 0; e < k; ++e) {
          for (size_t f = 0; f < k; ++f) {
            val += filter[e * k + f] * input_matrix[(i + e) * n + j + f];
          }
        }
        output_matrix.emplace_back(val);
      }
    }

    output_batch[b] = std::move(output_matrix);
  }

  return output_batch;
}

float *matrixToDevice(const Matrix &matrix) {
  const auto matrix_size = matrix.size() * sizeof(float);
  float *d_matrix = nullptr;

  cudaMalloc(reinterpret_cast<void **>(&d_matrix), matrix_size);
  cudaMemcpy(d_matrix, matrix.data(), matrix_size, cudaMemcpyHostToDevice);

  return d_matrix;
}

d_Batch batchToDevice(const Batch &batch) {
  d_Batch d_batch;
  d_batch.reserve(batch.size());

  for (const auto &matrix : batch) {
    d_batch.emplace_back(matrixToDevice(matrix));
  }

  return d_batch;
}

d_Batch initDeviceBatch(const size_t batch_size, const size_t c) {
  const auto matrix_size = c * c * sizeof(float);
  d_Batch d_batch(batch_size);

  for (auto &d_matrix : d_batch) {
    cudaMalloc(reinterpret_cast<void **>(&d_matrix), matrix_size);
  }

  return d_batch;
}

void freeDeviceBatch(const d_Batch &d_batch) {
  for (auto &d_matrix : d_batch) {
    cudaFree(d_matrix);
  }
}

Batch batchToHost(const d_Batch &d_batch, const size_t c) {
  Batch batch;
  batch.reserve(d_batch.size());

  for (const auto &d_matrix : d_batch) {
    Matrix matrix(c * c);
    cudaMemcpy(matrix.data(), d_matrix, c * c * sizeof(float),
               cudaMemcpyDeviceToHost);
    batch.emplace_back(std::move(matrix));
  }

  return batch;
}

__global__ void convLayerKernel(const float *input_matrix, float *output_matrix,
                                const float *filter, const size_t n,
                                const size_t c, const size_t k) {
  // extern __shared__ float filter_cache[];
  // for (auto i = threadIdx.x; i < k * k; i += blockDim.x) {
  //   filter_cache[i] = filter[i];
  // }
  // __syncthreads();

  const auto row = blockIdx.x;
  const auto col = threadIdx.x;

  float val = 0.0f;
  for (size_t e = 0; e < k; ++e) {
    for (size_t f = 0; f < k; ++f) {
      val += filter[e * k + f] * input_matrix[(row + e) * n + col + f];
    }
  }
  output_matrix[row * c + col] = val;
}

std::pair<Batch, float> convLayerCuda(const Batch &input_batch,
                                      const Matrix &filter, const size_t m,
                                      const size_t n, const size_t k) {
  const auto c = n - k + 1;
  const auto d_input_batch = batchToDevice(input_batch);
  const auto d_filter = matrixToDevice(filter);
  const auto d_output_batch = initDeviceBatch(m, c);

  const auto blocks_per_grid = c;
  const auto threads_per_block = c;
  const auto shared_memory_size = k * k * sizeof(float);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  for (size_t i = 0; i < m; ++i) {
    convLayerKernel<<<static_cast<unsigned>(blocks_per_grid),
                      static_cast<unsigned>(threads_per_block)>>>(
        d_input_batch[i], d_output_batch[i], d_filter, n, c, k);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float sec = 0.0f;
  cudaEventElapsedTime(&sec, start, stop);
  sec /= 1000.0f;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  const auto output_batch = batchToHost(d_output_batch, n - k + 1);

  freeDeviceBatch(d_input_batch);
  freeDeviceBatch(d_output_batch);
  cudaFree(d_filter);

  return {output_batch, sec};
}

float maxDifference(const Batch &batch1, const Batch &batch2) {
  float max_difference = 0.0f;

  for (size_t i = 0; i < batch1.size(); ++i) {
    const auto &m1 = batch1[i];
    const auto &m2 = batch2[i];

    for (size_t j = 0; j < m1.size(); ++j) {
      const auto difference = std::abs(m1[j] - m2[j]);
      if (difference > max_difference) {
        max_difference = difference;
      }
    }
  }

  return max_difference;
}

void print_result(const Batch &batch, const size_t n) {
  const Matrix &matrix = batch[0];

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      std::cout << matrix[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
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
    const auto batch = generateRandomBatch(m, n);
    const auto filter = generateRandomMatrix(k);

    const auto seq_start = omp_get_wtime();
    const auto seq_result = convLayerSeq(batch, filter, m, n, k);
    const auto seq_end = omp_get_wtime();
    std::cout << "Sequential time: " << seq_end - seq_start << " sec."
              << std::endl;
    // print_result(seq_result, n - k + 1);

    {
      // const auto par_start = omp_get_wtime();
      // const auto par_result = convLayerPar(batch, filter, m, n, k);
      // const auto par_end = omp_get_wtime();
      // std::cout << "Parallel time: " << par_end - par_start << " sec. ";
      // std::cout << "(diff: " << maxDifference(seq_result, par_result) << ")"
      //           << std::endl;
    }

    {
      const auto [cuda_result, cuda_time] =
          convLayerCuda(batch, filter, m, n, k);
      std::cout << "Cuda time: " << cuda_time << " sec. ";
      std::cout << "(diff: " << maxDifference(seq_result, cuda_result) << ")"
                << std::endl;
      // print_result(cuda_result, n - k + 1);
    }

  } catch (const std::bad_alloc &e) {
    std::cerr << "Error alloc memory: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
