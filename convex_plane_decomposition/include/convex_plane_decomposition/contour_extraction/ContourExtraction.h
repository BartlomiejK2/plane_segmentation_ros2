//
// Created by rgrandia on 04.06.20.
//

#pragma once

#include <opencv2/core/mat.hpp>

#include "convex_plane_decomposition/PlanarRegion.h"
#include "convex_plane_decomposition/PolygonTypes.h"
#include "convex_plane_decomposition/SegmentedPlanesMap.h"

#include "ContourExtractionParameters.h"

namespace convex_plane_decomposition {
namespace contour_extraction {

/**
 * Extracts the contours in map resolution, but with the x and y axis flipped.
 * This way all contours are in counter clockwise direction.
 */
class ContourExtraction {
 public:
  ContourExtraction(const ContourExtractionParameters& parameters);

  std::vector<PlanarRegion> extractPlanarRegions(const SegmentedPlanesMap& segmentedPlanesMap);

 private:
  ContourExtractionParameters parameters_;
  cv::Mat insetKernel_;
  cv::Mat marginKernel_;

  // Memory to reuse between calls
  cv::Mat binaryImage_;
};

/// Modifies the image in-place! Boundaries are simplified with boundarySimplificationEpsilon [(sub) pixels], 0 disables.
std::vector<BoundaryWithInset> extractBoundaryAndInset(cv::Mat& binary_image, const cv::Mat& erosionKernel,
                                                       double boundarySimplificationEpsilon = 0.0);

/// Contours are simplified with simplificationEpsilon [pixels] (Douglas-Peucker), 0 disables.
std::vector<CgalPolygonWithHoles2d> extractPolygonsFromBinaryImage(const cv::Mat& binary_image, double simplificationEpsilon = 0.0);

CgalPolygon2d cgalPolygonFromOpenCv(const std::vector<cv::Point>& openCvPolygon);

CgalPoint2d pixelToWorldFrame(const CgalPoint2d& pixelspaceCgalPoint2d, double resolution, const Eigen::Vector2d& mapOffset);

}  // namespace contour_extraction
}  // namespace convex_plane_decomposition
