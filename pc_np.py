import numpy as np
import open3d as o3d
#from open3d import geometry, visualization
from pyk4a import PyK4A
import json
import cv2

MIN_DEPTH = 300  # Example minimum depth in millimeters
MAX_DEPTH = 500  # Example maximum depth in millimeters


# Point cloud visualization in real time using numpy
def get_point_cloud_vectorized(depth_image, intrinsic_matrix):
    height, width = depth_image.shape
    
    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Convert to camera coordinates
    x_cam = (x - intrinsic_matrix[0, 2]) * depth_image / intrinsic_matrix[0, 0]
    y_cam = (y - intrinsic_matrix[1, 2]) * depth_image / intrinsic_matrix[1, 1]
    
    # Stack the coordinates together
    points = np.stack((x_cam, y_cam, depth_image), axis=-1)

    # Apply depth range threshold and remove points with zero depth
    mask = (depth_image > MIN_DEPTH) & (depth_image < MAX_DEPTH) & (depth_image != 0)
    return points[mask]

transformation_matrix = [[1, 0, 0, 0], 
                         [0, -1, 0, 0], 
                         [0, 0, -1, 0], 
                         [0, 0, 0, 1]]

def main():
    # Load camera intrinsics
    with open("C:\\Users\\gusta\\Desktop\\ITESM_Desktop\\maestria\\tesis\\TercerSemestre\\wunsch\\Notebooks\\Open3D-master\\Open3D-master\\examples\\python\\reconstruction_system\\sensors\\late-night-test\\intrinsic.json", 'r') as f:
        intrinsic_json = json.load(f)

    # Convert flat list to 3x3 nested list
    intrinsic_matrix_flat = intrinsic_json['intrinsic_matrix']
    intrinsic_matrix = [
        intrinsic_matrix_flat[0:3],
        intrinsic_matrix_flat[3:6],
        intrinsic_matrix_flat[6:9],
    ]
    intrinsic_matrix = np.array(intrinsic_matrix)

    k4a = PyK4A()
    k4a.start()

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()

    try:
        while True:
            capture = k4a.get_capture()
            depth_image = capture.depth
            color_image = capture.color 
            # Display color image using OpenCV
            cv2.imshow('Color Image', color_image)
            key = cv2.waitKey(1)

            if key == 27:
                break

            points = get_point_cloud_vectorized(depth_image, intrinsic_matrix)
            pcd.points = o3d.utility.Vector3dVector(points)

            pcd.transform(transformation_matrix)

            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
    finally:
        cv2.destroyAllWindows() # Close OpenCV window
        vis.destroy_window()
        k4a.stop()

if __name__ == "__main__":
    main()

