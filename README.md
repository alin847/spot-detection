# spot-detection

## Description
Detects spots in a grayscale image by using a modified version of the a trous wavelet transform described in [Olivo-Marin (2002)](doi.org/10.1016/S0031-3203(01)00127-3). Performs localization using the radial symmetry method described in [Parthasarathy (2012)](doi.org/10.1038/nmeth.2071).

## Procedure
Run "pip install -r requirements.txt" on your terminal if you don't have the following libraries: numpy, cv2, scipy, and skimage.

The general procedure to obtain the spot locations is as follows (look at example.py for guidance):
  1. Create a CellDetection Object. Image must be in grayscale for the code to work.
  2. Call spot_detection(scales, threshold) function.
     * **scales** is the number of times the wavelet decomposition is performed. For larger spots, choose a larger integer for scales. 
     * **threshold** is the multiplier (threshold * median absolute deviation) to filter the last wavelet plane. Choosing larger threshold decreases sensitivity to spot detection. Generally, 4.5 (default) is good enough in most cases and is used in the paper.
     * **Visually check** if binary image properly seperated the cells from background. If there are ring-like structures, try increasing scales. If it is detecting too many spots, try increasing threshold.
  3. Call localization(region_size, min_distance) function.
     * **region_size** is the n by n square region where the radial symmetry method will be performed for each spot. This should be set to slightly larger than the spot.
     * **min_distance** is the minimum distance between two local maxima of the distance transform (used to detect overlapping spots). This should be set to slightly smaller than the cell size.
     * **Visually check** if the cell localization looks good. If a single cell is being detected as multiple, try increasing min_distance. If overlapping cells are being detected as a single cell, try decreasing min_distance. If there is over/under detection, go back to step 2 and make sure your binary image looks good.
  4. Call count function.
  

