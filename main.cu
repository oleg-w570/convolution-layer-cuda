#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#define CHECK_CUDA_ERROR(call)                                                                                       \
  {                                                                                                                  \
    cudaError_t err = call;                                                                                          \
    if (err != cudaSuccess) {                                                                                        \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
      exit(EXIT_FAILURE);                                                                                            \
    }                                                                                                                \
  }

using Matrix = std::vector<float>;
using Batch = std::vector<Matrix>;

constexpr size_t BLOCK_SIZE = 32;

Matrix generateRandomMatrix(const size_t rows, const size_t cols) {
  static std::random_device rd;
  static std::mt19937 rng(rd());
  std::uniform_real_distribution dis(0.0f, 1.0f);

  Matrix matrix;
  matrix.reserve(rows * cols);

  for (size_t i = 0; i < rows * cols; ++i) {
    matrix.emplace_back(dis(rng));
  }

  return matrix;
}

Batch generateRandomBatch(const size_t batch_size, const size_t matrix_rows, const size_t matrix_cols) {
  Batch batch;
  batch.reserve(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    batch.emplace_back(generateRandomMatrix(matrix_rows, matrix_cols));
  }

  return batch;
}

Batch convLayerSeq(const Batch &inputs, const Batch &filters, const size_t m, const size_t n, const size_t k) {
  const auto c = n - k + 1;

  Batch output_batch;
  output_batch.reserve(m);

  for (size_t batch_i = 0; batch_i < m; ++batch_i) {
    const auto &input_matrix = inputs[batch_i];
    const auto &filter = filters[batch_i];

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

Batch convLayerPar(const Batch &inputs, const Batch &filters, const size_t m, const size_t n, const size_t k) {
  const auto c = n - k + 1;
  Batch output_batch(m);

#pragma omp parallel for
  for (size_t batch_i = 0; batch_i < m; ++batch_i) {
    const auto &input_matrix = inputs[batch_i];
    const auto &filter = filters[batch_i];

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

    output_batch[batch_i] = std::move(output_matrix);
  }

  return output_batch;
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

__global__ void convLayerKernel(const float *input, const float *filter, float *output, const size_t n, const size_t k,
                                const size_t c) {
  __shared__ float filter_cache[BLOCK_SIZE * BLOCK_SIZE];

  auto sum = 0.0f;
  const auto out_col = blockIdx.x * blockDim.x + threadIdx.x;
  const auto out_row = blockIdx.y * blockDim.y + threadIdx.y;
  const auto valid_thread = out_row < c && out_col < c;
  const auto tid = threadIdx.y * blockDim.x + threadIdx.x;
  const auto num_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (size_t block_i = 0; block_i < num_blocks; ++block_i) {
    for (size_t block_j = 0; block_j < num_blocks; ++block_j) {
      const auto filter_i = block_i * BLOCK_SIZE + threadIdx.y;
      const auto filter_j = block_j * BLOCK_SIZE + threadIdx.x;
      filter_cache[tid] = filter_i < k && filter_j < k ? filter[filter_i * k + filter_j] : 0.0f;
      __syncthreads();

      const auto valid_cache_rows = min(BLOCK_SIZE, k - block_i * BLOCK_SIZE);
      const auto valid_cache_cols = min(BLOCK_SIZE, k - block_j * BLOCK_SIZE);

      for (size_t cache_i = 0; cache_i < valid_cache_rows; ++cache_i) {
        for (size_t cache_j = 0; cache_j < valid_cache_cols; ++cache_j) {
          const auto e = block_i * BLOCK_SIZE + cache_i;
          const auto f = block_j * BLOCK_SIZE + cache_j;
          if (valid_thread) {
            sum += filter_cache[cache_i * BLOCK_SIZE + cache_j] * input[(out_row + e) * n + out_col + f];
          }
        }
      }
      __syncthreads();
    }
  }

  if (valid_thread) {
    output[out_row * c + out_col] = sum;
  }
}

auto convLayerCudaSingle(const Batch &inputs, const Batch &filters, const size_t m, const size_t n, const size_t k) {
  const auto c = n - k + 1;
  const auto input_size = n * n * sizeof(float);
  const auto filter_size = k * k * sizeof(float);
  const auto output_size = c * c * sizeof(float);

  Batch outputs(m, Matrix(c * c));

  dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks_per_grid((c + BLOCK_SIZE - 1) / BLOCK_SIZE, (c + BLOCK_SIZE - 1) / BLOCK_SIZE);

  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  CHECK_CUDA_ERROR(cudaEventRecord(start));

  for (size_t batch_i = 0; batch_i < m; ++batch_i) {
    const auto &input = inputs[batch_i];
    const auto &filter = filters[batch_i];
    auto &output = outputs[batch_i];

    float *d_input = nullptr;
    float *d_filter = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter, filter_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_filter, filter.data(), filter_size, cudaMemcpyHostToDevice));

    convLayerKernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_filter, d_output, n, k, c);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_output, output_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_filter));
    CHECK_CUDA_ERROR(cudaFree(d_input));
  }

  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  float execution_time = 0.0f;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&execution_time, start, stop));
  execution_time /= 1000.0f;

  CHECK_CUDA_ERROR(cudaEventDestroy(stop));
  CHECK_CUDA_ERROR(cudaEventDestroy(start));

  return std::make_pair(outputs, execution_time);
}

auto convLayerCudaStreams(const Batch &inputs, const Batch &filters, const size_t m, const size_t n, const size_t k) {
  const auto c = n - k + 1;
  const auto input_size = n * n * sizeof(float);
  const auto filter_size = k * k * sizeof(float);
  const auto output_size = c * c * sizeof(float);

  Batch outputs(m, Matrix(c * c));

  dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks_per_grid((c + BLOCK_SIZE - 1) / BLOCK_SIZE, (c + BLOCK_SIZE - 1) / BLOCK_SIZE);

  constexpr size_t num_streams = 4;
  std::array<cudaStream_t, num_streams> streams{};
  for (size_t s = 0; s < num_streams; ++s) {
    CHECK_CUDA_ERROR(cudaStreamCreate(&streams[s]));
  }

  const auto batch_size_per_stream = (m + num_streams - 1) / num_streams;

  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  CHECK_CUDA_ERROR(cudaEventRecord(start));

  for (size_t s = 0; s < num_streams; ++s) {
    const auto &stream = streams[s];
    const auto batch_start = s * batch_size_per_stream;
    const auto batch_end = min(batch_start + batch_size_per_stream, m);

    for (auto batch_i = batch_start; batch_i < batch_end; ++batch_i) {
      const auto &input = inputs[batch_i];
      const auto &filter = filters[batch_i];
      auto &output = outputs[batch_i];

      float *d_input = nullptr;
      float *d_filter = nullptr;
      float *d_output = nullptr;

      CHECK_CUDA_ERROR(cudaMallocAsync(&d_input, input_size, stream));
      CHECK_CUDA_ERROR(cudaMallocAsync(&d_filter, filter_size, stream));
      CHECK_CUDA_ERROR(cudaMallocAsync(&d_output, output_size, stream));

      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input, input.data(), input_size, cudaMemcpyHostToDevice, stream));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_filter, filter.data(), filter_size, cudaMemcpyHostToDevice, stream));

      convLayerKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_input, d_filter, d_output, n, k, c);
      CHECK_CUDA_ERROR(cudaGetLastError());

      CHECK_CUDA_ERROR(cudaMemcpyAsync(output.data(), d_output, output_size, cudaMemcpyDeviceToHost, stream));

      CHECK_CUDA_ERROR(cudaFreeAsync(d_output, stream));
      CHECK_CUDA_ERROR(cudaFreeAsync(d_filter, stream));
      CHECK_CUDA_ERROR(cudaFreeAsync(d_input, stream));
    }
  }

  for (size_t s = 0; s < num_streams; ++s) {
    CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[s]));
    CHECK_CUDA_ERROR(cudaStreamDestroy(streams[s]));
  }

  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  float execution_time = 0.0f;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&execution_time, start, stop));
  execution_time /= 1000.0f;

  CHECK_CUDA_ERROR(cudaEventDestroy(stop));
  CHECK_CUDA_ERROR(cudaEventDestroy(start));

  return std::make_pair(outputs, execution_time);
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <matrix_size> <filter_size> <batch_size>" << std::endl;
    return 1;
  }
  const auto n = std::stoull(argv[1]);
  const auto k = std::stoull(argv[2]);
  const auto m = std::stoull(argv[3]);

  const auto inputs = generateRandomBatch(m, n, n);
  const auto filters = generateRandomBatch(m, k, k);

  const auto seq_start = omp_get_wtime();
  const auto seq_result = convLayerSeq(inputs, filters, m, n, k);
  const auto seq_end = omp_get_wtime();
  std::cout << "Sequential time: " << seq_end - seq_start << " sec." << std::endl;
  {
    const auto par_start = omp_get_wtime();
    const auto par_result = convLayerPar(inputs, filters, m, n, k);
    const auto par_end = omp_get_wtime();
    std::cout << "Parallel time: " << par_end - par_start << " sec. ";
    std::cout << "(diff: " << maxDifference(seq_result, par_result) << ")" << std::endl;
  }

  {
    const auto [cuda_result, cuda_time] = convLayerCudaSingle(inputs, filters, m, n, k);
    std::cout << "Cuda (Single) time: " << cuda_time << " sec. ";
    std::cout << "(diff: " << maxDifference(seq_result, cuda_result) << ")" << std::endl;
  }

  {
    const auto [cuda_result, cuda_time] = convLayerCudaStreams(inputs, filters, m, n, k);
    std::cout << "Cuda (Streams) time: " << cuda_time << " sec. ";
    std::cout << "(diff: " << maxDifference(seq_result, cuda_result) << ")" << std::endl;
  }

  return 0;
}
