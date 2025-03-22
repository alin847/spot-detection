import cv2
import numpy as np
from scipy.stats import median_abs_deviation
from scipy.ndimage import convolve
from skimage.feature import peak_local_max


class CellDetection:
    def __init__(self, gray_image):
        """
        Creates a CellDetection object. The image must be grayscale.
        The general procedure to obtain the cell locations is as follows:
        1. spot_detection(scales, threshold)
        2. localization(region_size, min_distance)
        3. view_locations()

        Inputs:
        gray_image: grayscale image (numpy array)
        """
        self.gray_image = gray_image
        self.binary_image = None
        self.cell_locations = None


    # MAIN FUNCTIONS
    def spot_detection(self, scales, threshold=4.5):
        """
        Utilizes the a_trou_transform method to detect spots. It filters
        the last wavelet plane by thresholding. If Wi(x,y) is less than
        threshold * MAD, then Wi(x,y) = 0, else Wi(x,y) = 255.
        Returns nothing, updates binary image.

        Inputs:
        scales: the number of times the decomposition is performed (J)
        threshold: the threshold multiplier to filter the wavelet plane.

        Output:
        binary_image: binary image of the spots. 255 for spots and 0 for
        background.
        """

        # HELPER FUNCTIONS
        def a_trous_transform(gray_image, scales):
            """
            Performs the a trous wavelet transform described in
            https://doi.org/10.1016/S0031-3203(01)00127-3

            Inputs:
            gray_image: grayscale image
            scales: refers to the number of times the decomposition is performed (J)

            Outputs:
            wavelet_planes: refers to Wi [W1, W2, ... WJ]
            smooth_approximation: refers to AJ
            """
            kernel_values = [1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16]
            wavelet_planes = [0] * scales
            previous_approximation = gray_image.astype(np.float64)

            def update_kernel():
                for i in reversed(range(1, len(kernel_values))):
                    kernel_values.insert(i, 0)

            for i in range(scales):
                kernel_1D = np.array([kernel_values])
                kernel_2D = np.dot(np.transpose(kernel_1D), kernel_1D)

                smooth_approximation = convolve(
                    previous_approximation, kernel_2D, mode="mirror"
                )
                wavelet_planes[i] = np.subtract(
                    previous_approximation, smooth_approximation
                )

                previous_approximation = smooth_approximation
                update_kernel()

            return np.array(wavelet_planes), smooth_approximation

        # MAIN
        wavelet_planes, _ = a_trous_transform(self.gray_image, scales)
        wavelet_plane = wavelet_planes[-1]
        median_AD = median_abs_deviation(wavelet_plane, axis=None)
        _, binary_image = cv2.threshold(
            wavelet_plane, threshold * median_AD, 255, cv2.THRESH_BINARY
        )
        # removing specs
        binary_image = cv2.morphologyEx(
            binary_image,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        self.binary_image = binary_image.astype(np.uint8)
        return self.binary_image


    def localization(self, region_size, min_distance):
        """
        Performs localization on the image to obtain subpixel level
        accurary of cell location. Does this by calculating the
        centroids of the binary image. Then, uses radial symmetry
        method on a region_size by region_size grid to refine results.

        Input:
        region_size (int): the size of the square region to extract.
        This number must be odd.
        min_distance (int): the minimum distance between two local maxima
        for centroid detection.

        Output:
        cell_locations: [[i0,j0],...,[in,jn]]
        """
        # HELPER FUNCTIONS
        def extract_region(image, center, size):
            """
            Extracts a n by n region (n = size) of an image from
            the center of the image. Pads edges with 0s.
            """
            height, width = image.shape
            i, j = center
            region = np.zeros((size, size))
            # raw region indices (including out of ranges)
            top = i - size // 2
            bottom = i + size // 2 + size % 2
            left = j - size // 2
            right = j + size // 2 + size % 2

            # indices for image to extract
            top_i = max(0, top)
            bottom_i = min(height, bottom)
            left_i = max(0, left)
            right_i = min(width, right)

            # indices for region to add onto
            top_r = top_i - top
            bottom_r = top_r + (bottom_i - top_i)
            left_r = left_i - left
            right_r = left_r + (right_i - left_i)

            region[top_r:bottom_r, left_r:right_r] = image[
                top_i:bottom_i, left_i:right_i
            ]
            return region


        def get_centroids(binary_image, min_distance):
            """
            Performs the distance transform on the binary image. Then,
            detects the local maxima of the distance transform to obtain
            the cell centriods.

            Input:
            min_distance: the search distance for the local maxima

            Output:
            centroids: [[i0,j0],[i1,j1],...,[in,jn]]
            """
            dist = cv2.distanceTransform(binary_image, cv2.DIST_L2, 3)
            dist = cv2.GaussianBlur(dist, (3, 3), 0)
            return peak_local_max(dist, min_distance)


        def radial_symmetry(image_patch):
            """
            Performs the radial symmetry method from doi.org/10.1038/nmeth.2071

            Inputs:
            image_patch: n by n array of a patch of the original gray image

            Outputs:
            ic: i coordinate of the particle's center
            jc: j coordinate of the particle's center
            """

            def calc_mk(i, j):
                """
                Calculates mk of the bottom right of (i,j)
                """
                if (
                    (image_patch[i, j + 1] - image_patch[i + 1, j])
                    - (image_patch[i, j] - image_patch[i + 1, j + 1])
                ) == 0:
                    if (image_patch[i, j + 1] - image_patch[i + 1, j]) < 0:
                        mk = -(10**10)
                    else:
                        mk = 10**10
                else:
                    mk = (
                        (image_patch[i, j + 1] - image_patch[i + 1, j])
                        + (image_patch[i, j] - image_patch[i + 1, j + 1])
                    ) / (
                        (image_patch[i, j + 1] - image_patch[i + 1, j])
                        - (image_patch[i, j] - image_patch[i + 1, j + 1])
                    )
                return mk

            def calc_gradient(i, j):
                """
                Calculates gradient of the bottom right of (i,j)
                """
                nabla_Ik = (image_patch[i, j + 1] - image_patch[i + 1, j]) ** 2 + (
                    image_patch[i, j] - image_patch[i + 1, j + 1]
                ) ** 2
                return nabla_Ik ** (1 / 2)

            image_patch = image_patch.astype(float)
            region_size = image_patch.shape[0]
            n = (
                region_size - 1
            ) ** 2  # n refers to the length of the following variables

            # initializing variables
            m = np.zeros(n)  # slope
            nabla_I = np.zeros(n)  # gradient
            x = np.zeros(n)
            y = np.zeros(n)

            # calculating m, nabla_I, x, y
            k = 0
            for i in range(region_size - 1):
                for j in range(region_size - 1):
                    m[k] = calc_mk(i, j)
                    nabla_I[k] = calc_gradient(i, j)
                    x[k] = j + 0.5
                    y[k] = region_size - 1 - i - 0.5
                    k += 1

            # initial centroid guess
            xc = np.sum(x * nabla_I) / np.sum(nabla_I)
            yc = np.sum(y * nabla_I) / np.sum(nabla_I)

            # calculating weights
            w = np.zeros(n)  # weights
            d_kc = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5  # distance to guess
            w = nabla_I**2 / d_kc

            # setting up system of equations
            matrixA_1 = np.sum(-((m**2) * w) / (m**2 + 1))
            matrixA_2 = np.sum((m * w) / (m**2 + 1))
            matrixA_3 = np.sum(-(m * w) / (m**2 + 1))
            matrixA_4 = np.sum(w / (m**2 + 1))
            matrixA = np.array([[matrixA_1, matrixA_2], [matrixA_3, matrixA_4]])

            matrixB_1 = np.sum((m * w * (y - m * x)) / (m**2 + 1))
            matrixB_2 = np.sum((w * (y - m * x)) / (m**2 + 1))
            matrixB = np.array([[matrixB_1], [matrixB_2]])

            # solving system of equations
            solutions = np.linalg.inv(matrixA) @ matrixB
            jc = solutions[0][0]
            ic = region_size - 1 - solutions[1][0]
            return ic, jc


        # MAIN
        cell_locations = []
        centroids = get_centroids(self.binary_image, min_distance)

        for i, j in centroids:
            local_region = extract_region(self.gray_image, (i, j), region_size)
            ic, jc = radial_symmetry(local_region)
            ic += i - region_size // 2
            jc += j - region_size // 2
            cell_locations.append([ic, jc])
        self.cell_locations = np.array(cell_locations)
        return self.cell_locations


    def count(self):
        return self.cell_locations.shape[0]


    def view_locations(self):
        """
        Returns the image with the cell locations overlaid.
        """
        color_image = cv2.cvtColor(
            self.gray_image, cv2.COLOR_GRAY2RGB
        )
        color_image = color_image.astype(np.uint8)

        for i, j in self.cell_locations:
            color_image[int(i), int(j)] = [0, 0, 255]

        return color_image
