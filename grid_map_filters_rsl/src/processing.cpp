/**
 * @file        processing.cpp
 * @authors     Fabian Jenelten
 * @date        04.08, 2021
 * @affiliation ETH RSL
 * @brief       Processing filter (everything that is not smoothing or
 * inpainting).
 */

// grid map filters rsl.
#include <grid_map_filters_rsl/processing.hpp>

// stl.
#include <algorithm>
#include <cmath>


namespace grid_map::processing {

namespace {
/** Kernel window centered at the cell, clipped to the map. */
struct KernelWindow {
  int startRow;
  int startCol;
  int rows;
  int cols;
};

KernelWindow getKernelWindow(int rowId, int colId, int maxKernelId, int numRows, int numCols) {
  const int startRow = std::max(rowId - maxKernelId, 0);
  const int startCol = std::max(colId - maxKernelId, 0);
  const int endRow = std::min(rowId + maxKernelId, numRows - 1);
  const int endCol = std::min(colId + maxKernelId, numCols - 1);
  return {startRow, startCol, endRow - startRow + 1, endCol - startCol + 1};
}
}  // namespace

void dilate(grid_map::GridMap& map, const std::string& layerIn,
            const std::string& layerOut, const grid_map::Matrix& mask,
            int kernelSize, bool inpaint) {
  // Create new layer if missing
  if (!map.exists(layerOut)) {
    map.add(layerOut, map.get(layerIn));
  }

  // Reference to in and out maps.
  const grid_map::Matrix& H_in = map.get(layerIn);
  grid_map::Matrix& H_out = map.get(layerOut);

  // Apply mask.
  grid_map::Matrix H_in_masked;
  if (mask.cols() == 0 || mask.rows() == 0) {
    H_in_masked = H_in;
  } else {
    H_in_masked = H_in.cwiseProduct(mask);
  }
  const auto maxKernelId = (kernelSize - 1) / 2;

  for (auto colId = 0; colId < H_in.cols(); ++colId) {
    for (auto rowId = 0; rowId < H_in.rows(); ++rowId) {
      // Kernel window around the cell, at the map border only the part inside of the map.
      const auto window = getKernelWindow(rowId, colId, maxKernelId, H_in.rows(), H_in.cols());

      // Find maximum in region.
      if (inpaint || !std::isnan(H_in(rowId, colId))) {
        const auto maxInKernel =
            H_in_masked.block(window.startRow, window.startCol, window.rows, window.cols)
                .maxCoeffOfFinites();
        H_out(rowId, colId) =
            std::isnan(maxInKernel) ? H_in(rowId, colId) : maxInKernel;
      } else {
        H_out(rowId, colId) = NAN;
      }
    }
  }
}

void erode(grid_map::GridMap& map, const std::string& layerIn,
           const std::string& layerOut, const grid_map::Matrix& mask,
           int kernelSize, bool inpaint) {
  // Create new layer if missing
  if (!map.exists(layerOut)) {
    map.add(layerOut, map.get(layerIn));
  }

  // Reference to in and out maps.
  const grid_map::Matrix& H_in = map.get(layerIn);
  grid_map::Matrix& H_out = map.get(layerOut);

  // Apply mask.
  grid_map::Matrix H_in_masked;
  if (mask.cols() == 0 || mask.rows() == 0) {
    H_in_masked = H_in;
  } else {
    H_in_masked = H_in.cwiseProduct(mask);
  }
  const auto maxKernelId = (kernelSize - 1) / 2;

  for (auto colId = 0; colId < H_in.cols(); ++colId) {
    for (auto rowId = 0; rowId < H_in.rows(); ++rowId) {
      // Kernel window around the cell, at the map border only the part inside of the map.
      const auto window = getKernelWindow(rowId, colId, maxKernelId, H_in.rows(), H_in.cols());

      // Find minimum in region.
      if (inpaint || !std::isnan(H_in(rowId, colId))) {
        const auto minInKernel =
            H_in_masked.block(window.startRow, window.startCol, window.rows, window.cols)
                .minCoeffOfFinites();
        H_out(rowId, colId) =
            std::isnan(minInKernel) ? H_in(rowId, colId) : minInKernel;
      } else {
        H_out(rowId, colId) = NAN;
      }
    }
  }
}

void outline(grid_map::GridMap& map, const std::string& layerIn,
             const std::string& layerOut) {
  // Create new layer if missing
  if (!map.exists(layerOut)) {
    map.add(layerOut, map.get(layerIn));
  }

  // Reference to in and out maps.
  const grid_map::Matrix& H_in = map.get(layerIn);
  grid_map::Matrix& H_out = map.get(layerOut);

  constexpr auto kernelSize = 3;
  constexpr auto maxKernelId = (kernelSize - 1) / 2;

  for (auto colId = 0; colId < H_in.cols(); ++colId) {
    for (auto rowId = 0; rowId < H_in.rows(); ++rowId) {
      // Kernel window around the cell, at the map border only the part inside of the map.
      const auto window = getKernelWindow(rowId, colId, maxKernelId, H_in.rows(), H_in.cols());

      // Check if grid cell touches the nan grid cell.
      if (H_in.block(window.startRow, window.startCol, window.rows, window.cols)
              .hasNaN()) {
        H_out(rowId, colId) = H_in(rowId, colId);
      } else {
        H_out(rowId, colId) = NAN;
      }
    }
  }
}

void applyKernelFunction(
    grid_map::GridMap& map, const std::string& layerIn,
    const std::string& layerOut, int kernelSize,
    std::function<float(const Eigen::Ref<const grid_map::GridMap::Matrix>&)>
        func) {
  // Create new layer if missing
  if (!map.exists(layerOut)) {
    map.add(layerOut, map.get(layerIn));
  }

  // Reference to in and out maps.
  const grid_map::Matrix& H_in = map.get(layerIn);
  grid_map::Matrix& H_out = map.get(layerOut);

  const auto maxKernelId = (kernelSize - 1) / 2;

  // Pad with nan, so the kernel window is always centered at the cell, also at the map border.
  const auto padding = kernelSize - 1;
  grid_map::Matrix H_padded = grid_map::Matrix::Constant(
      H_in.rows() + padding, H_in.cols() + padding, NAN);
  H_padded.block(maxKernelId, maxKernelId, H_in.rows(), H_in.cols()) = H_in;

  for (auto colId = 0; colId < H_in.cols(); ++colId) {
    for (auto rowId = 0; rowId < H_in.rows(); ++rowId) {
      // Apply user defined function
      H_out(rowId, colId) =
          func(H_padded.block(rowId, colId, kernelSize, kernelSize));
    }
  }
}
} // namespace grid_map::processing

