#include "tsyplakov_k_mul_double_crs_matrix/all/include/ops_all.hpp"

#include <mpi.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tsyplakov_k_mul_double_crs_matrix/common/include/common.hpp"

namespace tsyplakov_k_mul_double_crs_matrix {

namespace {

struct Distribution {
  int start_row{};
  int local_rows{};
};

Distribution GetDistribution(int rows, int rank, int size) {
  const int rows_per_process = rows / size;
  const int remainder = rows % size;

  Distribution distribution;

  distribution.start_row = (rank * rows_per_process) + std::min(rank, remainder);

  distribution.local_rows = rows_per_process + ((rank < remainder) ? 1 : 0);

  return distribution;
}

void ComputeRow(const SparseMatrixCRS &a, const SparseMatrixCRS &b, int row, std::vector<double> &values,
                std::vector<int> &cols) {
  std::unordered_map<int, double> accumulator;

  for (int idx_a = a.row_ptr[row]; idx_a < a.row_ptr[row + 1]; ++idx_a) {
    const int k = a.col_index[idx_a];
    const double val_a = a.values[idx_a];

    for (int idx_b = b.row_ptr[k]; idx_b < b.row_ptr[k + 1]; ++idx_b) {
      const int col = b.col_index[idx_b];

      accumulator[col] += val_a * b.values[idx_b];
    }
  }

  values.reserve(accumulator.size());
  cols.reserve(accumulator.size());

  for (const auto &[col, val] : accumulator) {
    if (std::fabs(val) > 1e-12) {
      cols.push_back(col);
      values.push_back(val);
    }
  }
}

void ComputeLocalRows(const SparseMatrixCRS &a, const SparseMatrixCRS &b, int start_row, int local_rows,
                      std::vector<std::vector<double>> &local_values, std::vector<std::vector<int>> &local_cols) {
  tbb::parallel_for(tbb::blocked_range<int>(0, local_rows), [&](const tbb::blocked_range<int> &range) {
    for (int local_row = range.begin(); local_row < range.end(); ++local_row) {
      const int global_row = start_row + local_row;

      ComputeRow(a, b, global_row, local_values[local_row], local_cols[local_row]);
    }
  });
}

SparseMatrixCRS BuildLocalMatrix(int local_rows, int cols, const std::vector<std::vector<double>> &local_values,
                                 const std::vector<std::vector<int>> &local_cols) {
  SparseMatrixCRS local_matrix(local_rows, cols);

  for (int i = 0; i < local_rows; ++i) {
    local_matrix.row_ptr[i + 1] = local_matrix.row_ptr[i] + static_cast<int>(local_values[i].size());
  }

  const int local_nnz = local_matrix.row_ptr[local_rows];

  local_matrix.values.reserve(local_nnz);
  local_matrix.col_index.reserve(local_nnz);

  for (int i = 0; i < local_rows; ++i) {
    local_matrix.values.insert(local_matrix.values.end(), local_values[i].begin(), local_values[i].end());

    local_matrix.col_index.insert(local_matrix.col_index.end(), local_cols[i].begin(), local_cols[i].end());
  }

  return local_matrix;
}

void BuildDisplacements(const std::vector<int> &recv_counts, std::vector<int> &displs) {
  for (std::size_t i = 1; i < recv_counts.size(); ++i) {
    displs[i] = displs[i - 1] + recv_counts[i - 1];
  }
}

void BuildRowPtr(SparseMatrixCRS &matrix, const std::vector<int> &row_sizes) {
  for (int i = 0; i < matrix.rows; ++i) {
    matrix.row_ptr[i + 1] = matrix.row_ptr[i] + row_sizes[i];
  }
}

}  // namespace

TsyplakovKTestTaskALL::TsyplakovKTestTaskALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TsyplakovKTestTaskALL::ValidationImpl() {
  const auto &input = GetInput();

  return input.a.cols == input.b.rows;
}

bool TsyplakovKTestTaskALL::PreProcessingImpl() {
  return true;
}

bool TsyplakovKTestTaskALL::RunImpl() {
  const auto &input = GetInput();

  const auto &a = input.a;
  const auto &b = input.b;

  int rank = 0;
  int size = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const Distribution distribution = GetDistribution(a.rows, rank, size);

  std::vector<std::vector<double>> local_values(distribution.local_rows);

  std::vector<std::vector<int>> local_cols(distribution.local_rows);

  ComputeLocalRows(a, b, distribution.start_row, distribution.local_rows, local_values, local_cols);

  SparseMatrixCRS local_matrix = BuildLocalMatrix(distribution.local_rows, b.cols, local_values, local_cols);

  const int local_nnz = local_matrix.row_ptr[distribution.local_rows];

  std::vector<int> recv_nnz(size);
  std::vector<int> recv_rows(size);

  MPI_Gather(&local_nnz, 1, MPI_INT, recv_nnz.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Gather(&distribution.local_rows, 1, MPI_INT, recv_rows.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs_nnz(size, 0);
  std::vector<int> displs_rows(size, 0);

  if (rank == 0) {
    BuildDisplacements(recv_nnz, displs_nnz);
    BuildDisplacements(recv_rows, displs_rows);
  }

  int global_nnz = 0;

  if (rank == 0 && !recv_nnz.empty()) {
    global_nnz = displs_nnz.back() + recv_nnz.back();
  }

  SparseMatrixCRS result_matrix;

  if (rank == 0) {
    result_matrix.rows = a.rows;
    result_matrix.cols = b.cols;

    result_matrix.values.resize(global_nnz);
    result_matrix.col_index.resize(global_nnz);
    result_matrix.row_ptr.resize(a.rows + 1, 0);
  }

  MPI_Gatherv(local_matrix.values.data(), local_nnz, MPI_DOUBLE, rank == 0 ? result_matrix.values.data() : nullptr,
              recv_nnz.data(), displs_nnz.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Gatherv(local_matrix.col_index.data(), local_nnz, MPI_INT, rank == 0 ? result_matrix.col_index.data() : nullptr,
              recv_nnz.data(), displs_nnz.data(), MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_row_sizes(distribution.local_rows);

  for (int i = 0; i < distribution.local_rows; ++i) {
    local_row_sizes[i] = static_cast<int>(local_values[i].size());
  }

  std::vector<int> gathered_row_sizes;

  if (rank == 0) {
    gathered_row_sizes.resize(a.rows);
  }

  MPI_Gatherv(local_row_sizes.data(), distribution.local_rows, MPI_INT, rank == 0 ? gathered_row_sizes.data() : nullptr,
              recv_rows.data(), displs_rows.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    BuildRowPtr(result_matrix, gathered_row_sizes);

    GetOutput() = std::move(result_matrix);
  }

  return true;
}

bool TsyplakovKTestTaskALL::PostProcessingImpl() {
  return true;
}

}  // namespace tsyplakov_k_mul_double_crs_matrix
