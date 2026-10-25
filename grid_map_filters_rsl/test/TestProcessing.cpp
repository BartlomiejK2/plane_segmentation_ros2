/**
 * @brief       Tests for grid map processing filters.
 */

#include <gtest/gtest.h>

#include <cmath>

#include <grid_map_filters_rsl/processing.hpp>

using namespace grid_map;

namespace {
/** 10 x 10 cells map with layer "input" */
GridMap getMap(float value) {
  GridMap map;
  map.setGeometry(Length(1.0, 1.0), 0.1, Position(0.0, 0.0));
  map.add("input", value);
  return map;
}
}  // namespace

TEST(TestProcessing, applyKernelFunction_centeredAtBorder) {  // NOLINT
  auto map = getMap(0.0);
  auto& input = map.get("input");
  for (int col = 0; col < input.cols(); ++col) {
    for (int row = 0; row < input.rows(); ++row) {
      input(row, col) = row + 100 * col;  // unique value per cell
    }
  }

  // Middle of the window is the cell itself, also at the map border
  constexpr int kernelSize = 5;
  processing::applyKernelFunction(map, "input", "center", kernelSize,
                                  [](const Eigen::Ref<const GridMap::Matrix>& data) -> float {
                                    return data(kernelSize / 2, kernelSize / 2);
                                  });
  EXPECT_TRUE(map.get("center").isApprox(map.get("input")));

  // Part of the window outside of the map is nan
  processing::applyKernelFunction(map, "input", "nanCount", kernelSize,
                                  [](const Eigen::Ref<const GridMap::Matrix>& data) -> float {
                                    return static_cast<float>(data.size() - data.array().isFinite().count());
                                  });
  EXPECT_FLOAT_EQ(map.get("nanCount")(0, 0), 16.0);
  EXPECT_FLOAT_EQ(map.get("nanCount")(1, 5), 5.0);
  EXPECT_FLOAT_EQ(map.get("nanCount")(5, 5), 0.0);
}

TEST(TestProcessing, dilateAndErode_windowAtBorder) {  // NOLINT
  auto map = getMap(0.0);
  map.get("input")(2, 5) = 1.0;
  processing::dilate(map, "input", "dilated", Matrix(), 3);
  EXPECT_FLOAT_EQ(map.get("dilated")(1, 5), 1.0);  // neighbour
  EXPECT_FLOAT_EQ(map.get("dilated")(0, 5), 0.0);  // two cells away, outside of the kernel

  map.get("input")(2, 5) = -1.0;
  processing::erode(map, "input", "eroded", Matrix(), 3);
  EXPECT_FLOAT_EQ(map.get("eroded")(1, 5), -1.0);
  EXPECT_FLOAT_EQ(map.get("eroded")(0, 5), 0.0);
}

TEST(TestProcessing, outline_windowAtBorder) {  // NOLINT
  auto map = getMap(1.0);
  map.get("input")(2, 5) = NAN;
  processing::outline(map, "input", "outline");
  EXPECT_FLOAT_EQ(map.get("outline")(1, 5), 1.0);  // touches nan cell
  EXPECT_TRUE(std::isnan(map.get("outline")(0, 5)));  // two cells away
}
