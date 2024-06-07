# 1. Import the libraries
import cv2 
import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A
import numpy as np
import json
import open3d as o3d

# Filtering values
MIN_DEPTH = 0
MAX_DEPTH = 1000

lower_green = np.array([18, 7, 30])
upper_green = np.array([94, 255, 148])

# 2. Image processing functions
def filter_color(image, lower_bound, upper_bound):
    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Create a mask using the given bounds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return mask

def filter_depth(image, min_distance, max_distance):
    # Create a mask using the given bounds
    mask = ((image > min_distance) & (image < max_distance)).astype(np.uint8) * 255
    return mask

# Trackbar callback functions
def update_lower_h(val):
    global lower_green
    lower_green[0] = val

def update_lower_s(val):
    global lower_green
    lower_green[1] = val

def update_lower_v(val):
    global lower_green
    lower_green[2] = val

def update_upper_h(val):
    global upper_green
    upper_green[0] = val

def update_upper_s(val):
    global upper_green
    upper_green[1] = val

def update_upper_v(val):
    global upper_green
    upper_green[2] = val

# Create a window for the trackbars
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)

# Create the trackbars
cv2.createTrackbar("Lower H", "Trackbars", lower_green[0], 179, update_lower_h)
cv2.createTrackbar("Lower S", "Trackbars", lower_green[1], 255, update_lower_s)
cv2.createTrackbar("Lower V", "Trackbars", lower_green[2], 255, update_lower_v)
cv2.createTrackbar("Upper H", "Trackbars", upper_green[0], 179, update_upper_h)
cv2.createTrackbar("Upper S", "Trackbars", upper_green[1], 255, update_upper_s)
cv2.createTrackbar("Upper V", "Trackbars", upper_green[2], 255, update_upper_v)

# 3. Real time Point Cloud visualization
def get_point_cloud_vectorized(depth_image, intrinsic_matrix):
    height, width = depth_image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    x_cam = (x - intrinsic_matrix[0, 2]) * depth_image / intrinsic_matrix[0, 0]
    y_cam = (y - intrinsic_matrix[1, 2]) * depth_image / intrinsic_matrix[1, 1]

    points = np.stack((x_cam, y_cam, depth_image), axis=-1)
    mask = (depth_image > MIN_DEPTH) & (depth_image < MAX_DEPTH) & (depth_image != 0)
    return points[mask]

transformation_matrix = np.array([[1, 0, 0, 0], 
                                  [0, -1, 0, 0], 
                                  [0, 0, -1, 0], 
                                  [0, 0, 0, 1]])

# Main function
def main():
    with open('intrinsic.json', 'r') as f:
        intrinsic_json = json.load(f)
    
    intrinsic_matrix_flat = intrinsic_json['intrinsic_matrix']
    intrinsic_matrix = [
        intrinsic_matrix_flat[0:3],
        intrinsic_matrix_flat[3:6],
        intrinsic_matrix_flat[6:9],
    ]
    intrinsic_matrix = np.array(intrinsic_matrix)

    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        )
    )
    k4a.start()

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()

    try: 
        while True: 
            capture = k4a.get_capture()

            if capture.color is not None:
                filtered_color_mask = filter_color(capture.color, lower_green, upper_green)
                cv2.imshow("Color Mask", filtered_color_mask)
            
            if capture.transformed_depth is not None:
                depth_filtered_mask = filter_depth(capture.transformed_depth, MIN_DEPTH, MAX_DEPTH)
                cv2.imshow("Depth Mask", depth_filtered_mask)
                cv2.imshow("Depth Image", colorize(capture.transformed_depth, (MIN_DEPTH, MAX_DEPTH)))
            
            fused_mask = cv2.bitwise_and(filtered_color_mask, depth_filtered_mask)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fused_mask, 4, cv2.CV_32S)
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_label = np.argmax(areas) + 1
            largest_component = np.zeros_like(labels, dtype=np.uint8)
            largest_component[labels == max_label] = 255

            result_color = cv2.bitwise_and(capture.color, capture.color, mask=largest_component)
            result_depth = cv2.bitwise_and(capture.transformed_depth, capture.transformed_depth, mask=largest_component)

            centroid = tuple(int(c) for c in centroids[max_label])
            result_color_with_mark = cv2.circle(result_color, centroid, 10, (0, 255, 0), -1)
            capture_color_with_mark = capture.color.copy()
            cv2.circle(capture_color_with_mark, centroid, 10, (0, 255, 0), -1)

            points = get_point_cloud_vectorized(result_depth, intrinsic_matrix=intrinsic_matrix)
            pcd.points = o3d.utility.Vector3dVector(points)

            pcd.transform(transformation_matrix)
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            cv2.imshow("Original image with centroid", capture_color_with_mark)

            key = cv2.waitKey(10)
            if key != -1:
                break
    finally: 
        cv2.destroyAllWindows()
        vis.destroy_window()
        k4a.stop()

if __name__ == "__main__":
    main()

