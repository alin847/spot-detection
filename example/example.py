import cv2
from CellDetection import CellDetection


# Step 1: create CellDetection Object
gray_image = cv2.imread("example.tif", cv2.IMREAD_ANYDEPTH)
image = CellDetection(gray_image=gray_image)

# Step 2a: call spot detection function
binary_image = image.spot_detection(scales=4, threshold=6)

# Step 2b: check if binary image looks good
cv2.imshow("binary image", binary_image)
cv2.waitKey()

# Step 3a: call localization
locations = image.localization(region_size=50, min_distance=5)

# Step 3b: check if cell detection looks good
outlined_image = image.view_locations()
cv2.imshow("outlined image", outlined_image)
cv2.waitKey()

# Step 4: call count
cell_count = image.count()
print(f"Cell count: {cell_count}")
