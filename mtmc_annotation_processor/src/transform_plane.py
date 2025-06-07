import numpy as np
import cv2
import json

class CoordinateTransformer:
    def __init__(self, camera_points=None, map_points=None):
        """
        Initialize the transformer with corresponding points from camera and map.
        If no points provided, matrices must be loaded from file.
        
        Args:
            camera_points: numpy array of shape (4, 2) containing camera frame coordinates
            map_points: numpy array of shape (4, 2) containing map coordinates
        """
        if camera_points is not None and map_points is not None:
            if len(camera_points) != 4 or len(map_points) != 4:
                raise ValueError("Exactly 4 points are required for each coordinate system")
                
            self.camera_points = np.float32(camera_points)
            self.map_points = np.float32(map_points)
            
            # Calculate the perspective transformation matrix
            self.transform_matrix = cv2.getPerspectiveTransform(
                self.camera_points,
                self.map_points
            )
            
    @classmethod
    def load_matrices(cls, data):
        """
        Load transformation matrices from a file.
        
        Args:
            data: dict of the camera, map points and homography matrices, respectively
            
        Returns:
            CoordinateTransformer instance with loaded matrices
        """
            
        transformer = cls()
        transformer.transform_matrix = np.array(data['transform_matrix'])
        transformer.camera_points = np.array(data['camera_points'])
        transformer.map_points = np.array(data['map_points'])
        
        return transformer
    
    def camera_to_map(self, points):
        """
        Transform points from camera coordinates to map coordinates.
        
        Args:
            points: numpy array of shape (N, 2) containing points in camera coordinates
            
        Returns:
            numpy array of shape (N, 2) containing transformed points in map coordinates
        """
        points = np.float32(points).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, self.transform_matrix)
        return transformed_points.reshape(-1, 2)


# Example usage:
if __name__ == "__main__":
    # Example coordinates
    camera_points = np.load("./101/101_human_points.npy")
    print(camera_points)
    map_points = np.load("./101/101_points.npy")
    print(map_points)  
    # Create and save transformer
    transformer = CoordinateTransformer(camera_points, map_points)
    transformer.save_matrices("./101/101_transform_matrices.json")
    
    # Load transformer from file
    loaded_transformer = CoordinateTransformer.load_matrices("./101/101_transform_matrices.json")
    
    # Test with a new point
    new_camera_point = np.array([[250, 200]])
    mapped_point = loaded_transformer.camera_to_map(new_camera_point)
    print(f"Camera point {new_camera_point} maps to {mapped_point} on the topology map")
