# spot-detection

## Description
Detects spots in a grayscale image by using the a trous wavelet transform described in [Olivo-Marin (2002)](doi.org/10.1016/S0031-3203(01)00127-3). Performs localization using the radial symmetry method described in [Parthasarathy (2012)](doi.org/10.1038/nmeth.2071).

## Procedure
The general procedure to obtain the spot locations is as follows:
  1. spot_detection(scales, threshold)
     * **scales** is the number of times the wavelet decomposition is performed. For larger spots, choose larger integer for scales. 
     * **threshold** is the multiplier (threshold * median absolute deviation) to filter the last wavelet plane. Choosing larger threshold decreases sensitivity to spot detection.
  2. localization(region_size, min_distance)
     * **region_size** is the n by n square region where the radial symmetry method will be performed for each spot. This should be set to slightly larger than the spot.
     * **min_distance** is the minimum distance between two local maxima of the distance transform (used to detect overlapping spots).
  3. view_locations()

