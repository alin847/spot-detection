import cv2
from SpotDetection import SpotDetection


# Step 1: create SpotDetection Object
gray_image = cv2.imread("example.tif", cv2.IMREAD_ANYDEPTH)
image = SpotDetection(gray_image=gray_image)

# Step 2a: call spot detection function
binary_image = image.detection(scales=4, threshold=6)

# Step 2b: check if binary image looks good
cv2.imshow("binary image", binary_image)
cv2.waitKey()

# Step 3a: call localization
locations = image.localization(region_size=50, min_distance=5)

# Step 3b: check if cell detection looks good
outlined_image = image.view_locations(auto=False)
cv2.imshow("outlined image", outlined_image)
cv2.waitKey()

# Step 4: call count
cell_count = image.count()
print(f"Cell count: {cell_count}")
