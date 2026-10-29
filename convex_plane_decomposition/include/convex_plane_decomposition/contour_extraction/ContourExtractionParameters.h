//
// Created by rgrandia on 07.06.20.
//

#pragma once

namespace convex_plane_decomposition {
namespace contour_extraction {

struct ContourExtractionParameters {
  /// Size of the kernel creating the boundary offset. In number of (sub) pixels.
  int marginSize = 1;

  /// Maximum deviation of the simplified region boundary (Douglas-Peucker) in number of (sub) pixels, 0 disables.
  /// Fewer boundary edges make convex region growing faster. Keep below 1, so insets stay inside of the boundary.
  double boundarySimplificationEpsilon = 0.0;
};

}  // namespace contour_extraction
}  // namespace convex_plane_decomposition
