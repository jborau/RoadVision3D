import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import List, Any
import numpy as np
import cv2
import logging
from typing import Any, Dict, List, Optional, Tuple

class Visualizer:
    def __init__(self, calib: Any = None, pitch: float = 0.0):
        """
        Initializes the Visualizer with optional calibration data.

        Parameters:
        - calib: Calibration data for projecting 3D bounding boxes.
        - pitch: Camera pitch angle for 3D projections.
        """
        self.calib = calib
        self.pitch = pitch

    def draw_2d_bboxes(self, image: Image.Image, objects: List[Any],
                       color: str = 'red', width: int = 2,
                       display: bool = False, save_path: str = None) -> Image.Image:
        """
        Draws 2D bounding boxes on a copy of the image.

        Parameters:
        - image: PIL.Image object to draw on.
        - objects: List of objects with a 'box2d' attribute containing bounding box coordinates.
        - color: Color of the bounding box edges.
        - width: Line width of the bounding box edges.
        - display: If True, displays the image after drawing.
        - save_path: If provided, saves the image to the specified path.

        Returns:
        - image_with_boxes: PIL.Image object with the bounding boxes drawn.
        """
        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes)
        for obj in objects:
            draw.rectangle(obj.box2d, outline=color, width=width)
        if save_path:
            image_with_boxes.save(save_path)
        if display:
            plt.imshow(image_with_boxes)
            plt.axis('off')
            plt.show()
        return image_with_boxes

    def draw_3d_bbox(self, image: Image.Image, label: Any, color: str = 'red', 
                     color_front: str = 'green', width: int = 2) -> Image.Image:
        """
        Draws a single 3D bounding box on the image.

        Parameters:
        - image: PIL.Image object to draw on.
        - label: An object containing 3D bounding box data.
        - color: Color of the bounding box edges.
        - width: Line width of the bounding box edges.

        Returns:
        - image_with_box: PIL.Image object with the bounding box drawn.
        """
        if not self.calib:
            raise ValueError("Calibration data is required for drawing 3D bounding boxes.")

        image_with_box = image.copy()
        draw = ImageDraw.Draw(image_with_box)

        corners3d = label.generate_corners3d(camera_pitch=self.pitch)
        corners2d, _ = self.calib.rect_to_img(corners3d)
        corners2d = [tuple(coord) for coord in corners2d]

        # Define the edges of the bounding box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]

        for edge in edges:
            draw.line([corners2d[edge[0]], corners2d[edge[1]]], fill=color, width=width)

        # Add a green cross on the front face to indicate car direction
        # Assuming the front face is defined by corners [0, 1, 5, 4]

        # Indices of the front face corners
        front_face_indices = [0, 1, 5, 4]

        # Draw the green cross on the front face
        # Line from corner 0 to corner 5
        draw.line([corners2d[front_face_indices[0]], corners2d[front_face_indices[2]]],
                fill=color_front, width=width)
        # Line from corner 1 to corner 4
        draw.line([corners2d[front_face_indices[1]], corners2d[front_face_indices[3]]],
                fill=color_front, width=width)

        return image_with_box

    def draw_3d_bboxes(self, image: Image.Image, labels: List[Any],
                       color: str = 'red', color_front: str = 'green', width: int = 2,
                       display: bool = False, save_path: str = None) -> Image.Image:
        """
        Draws multiple 3D bounding boxes on the image.

        Parameters:
        - image: PIL.Image object to draw on.
        - labels: List of labels, each containing 3D bounding box data.
        - color: Color of the bounding box edges.
        - width: Line width of the bounding box edges.
        - display: If True, displays the image after drawing.
        - save_path: If provided, saves the image to the specified path.

        Returns:
        - image_with_boxes: PIL.Image object with the bounding boxes drawn.
        """
        image_with_boxes = image.copy()

        for label in labels:
            # Draw each bounding box on the image
            image_with_boxes = self.draw_3d_bbox(image_with_boxes, label, color=color, color_front=color_front, width=width)

        if save_path:
            image_with_boxes.save(save_path)
        if display:
            plt.imshow(image_with_boxes)
            plt.axis('off')
            plt.show()
        return image_with_boxes


class Visualizer_dataloader:
    def __init__(self, calib: Any = None, pitch: float = 0.0, cfg: Dict[str, Any] = None):
        """
        Initializes the Visualizer with optional calibration data.

        Parameters:
        - calib: Calibration data for projecting 3D bounding boxes.
        - pitch: Camera pitch angle for 3D projections (in radians).
        - cfg: Configuration dictionary containing parameters like mean, std, class names, etc.
        """
        self.calib = calib
        self.pitch = pitch
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def _denormalize_image(self, inputs: np.ndarray) -> np.ndarray:
        """
        Denormalizes the input image tensor using the mean and std from cfg.

        Parameters:
        - inputs: Input image tensor (C, H, W).

        Returns:
        - img: Denormalized image as a NumPy array (H, W, C).
        """
        if self.cfg is None or 'mean' not in self.cfg or 'std' not in self.cfg:
            raise ValueError("Mean and std must be provided in cfg for image denormalization.")

        mean = np.array(self.cfg['mean'], dtype=np.float32)
        std = np.array(self.cfg['std'], dtype=np.float32)
        img = inputs.transpose(1, 2, 0)  # C, H, W -> H, W, C
        img = (img * std + mean) * 255  # Denormalize
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def draw_2d_bboxes(
        self,
        inputs: np.ndarray,
        targets: Dict[str, Any],
        class_names: Optional[List[str]] = None,
        with_center: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize the image and plot both the 2D bounding boxes, center points, and class names of the objects.

        Parameters:
        - inputs: Input image tensor of shape (C, H, W).
        - targets: Dictionary containing target components.
        - class_names: Optional list of class names, corresponding to the class IDs (cls_ids) in targets.
        - with_center: Whether to plot the center points of the objects.
        - save_path: Optional path to save the visualized image.
        """
        self.logger.debug("Drawing 2D bounding boxes.")
        img = self._denormalize_image(inputs)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        # Validate targets
        required_keys = ['mask_2d', 'indices', 'offset_2d', 'size_2d', 'cls_ids']
        for key in required_keys:
            if key not in targets:
                raise KeyError(f"Key '{key}' missing in targets.")

        if self.cfg is None or 'downsample' not in self.cfg:
            raise ValueError("Configuration 'cfg' with 'downsample' is required.")

        valid_mask = targets['mask_2d'] == 1
        valid_indices = targets['indices'][valid_mask]
        valid_offset_2d = targets['offset_2d'][valid_mask]
        valid_size_2d = targets['size_2d'][valid_mask]  # Width and height of each box
        valid_cls_ids = targets['cls_ids'][valid_mask]  # Class IDs of valid objects

        # Default class names (if not provided)
        if class_names is None:
            class_names = self.cfg.get('eval_cls', [])

        # Calculate the center locations in the original image
        downsample_factor = self.cfg['downsample']
        feature_map_width = inputs.shape[2] // downsample_factor  # Width of the downsampled feature map

        centers_x = ((valid_indices % feature_map_width) + valid_offset_2d[:, 0]) * downsample_factor
        centers_y = ((valid_indices // feature_map_width) + valid_offset_2d[:, 1]) * downsample_factor

        # Scale up the size_2d to match the original image dimensions
        valid_size_2d *= downsample_factor

        # Draw bounding boxes, center points, and class labels for valid objects
        for i in range(len(valid_size_2d)):
            w, h = valid_size_2d[i]  # Scaled width and height of the object
            x_min = centers_x[i] - w / 2  # Top-left corner x
            y_min = centers_y[i] - h / 2  # Top-left corner y

            # Get the class name for this object
            cls_id = valid_cls_ids[i]
            class_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

            # Draw the bounding box
            rect = plt.Rectangle((x_min, y_min), w, h, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)

            # Draw the center point
            if with_center:
                ax.scatter(centers_x[i], centers_y[i], c='red', s=50, marker='.', label='Object Center' if i == 0 else "")

            # Annotate with the class name
            ax.text(x_min, y_min - 5, class_name, color='yellow', fontsize=12, weight='bold')

        if with_center and len(valid_size_2d) > 0:
            ax.legend()
        ax.axis('off')
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close(fig)

    def draw_heatmaps(
        self,
        inputs: np.ndarray,
        targets: Dict[str, Any],
        class_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize the heatmap overlaid on the image.

        Parameters:
        - inputs: Input image tensor of shape (C, H, W).
        - targets: Dictionary containing target components.
        - class_idx: Class index for which to visualize the heatmap.
        - save_path: Optional path to save the visualized image.
        """
        self.logger.debug("Drawing heatmaps.")
        img = self._denormalize_image(inputs)

        # Validate class_idx
        if 'heatmap' not in targets:
            raise KeyError("Key 'heatmap' missing in targets.")
        if class_idx < 0 or class_idx >= targets['heatmap'].shape[0]:
            raise ValueError(f"class_idx {class_idx} is out of bounds.")

        # Extract and resize the heatmap
        heatmap = targets['heatmap'][class_idx]
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]  # Take only RGB channels
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Blend the heatmap with the image
        alpha = 0.6
        blended = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

        # Compute center positions
        # valid_mask = targets['mask_2d'] == 1
        # valid_indices = targets['indices'][valid_mask]
        # valid_offset_2d = targets['offset_2d'][valid_mask]

        downsample_factor = self.cfg['downsample']
        feature_map_width = inputs.shape[2] // downsample_factor

        # centers_x = ((valid_indices % feature_map_width) + valid_offset_2d[:, 0]) * downsample_factor
        # centers_y = ((valid_indices // feature_map_width) + valid_offset_2d[:, 1]) * downsample_factor

        # Plot the image with centers
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(blended)
        # ax.scatter(centers_x, centers_y, c='red', s=50, marker='.', label='Object Center')
        ax.legend()
        ax.axis('off')
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close(fig)

    def display_centers_dimensions_depth(
        self,
        inputs: np.ndarray,
        targets: Dict[str, Any],
        num_bins: int = 12,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize the image and plot the center points of the objects with their IDs on the image.
        Also, print the ID, depth, 3D dimensions, rotation_y, and heading_bin + heading_res in the console.

        Parameters:
        - inputs: Input image tensor of shape (C, H, W).
        - targets: Dictionary containing target components.
        - num_bins: Number of heading bins used (default is 12).
        - save_path: Optional path to save the visualized image.
        """
        self.logger.debug("Displaying centers, dimensions, and depth.")
        img = self._denormalize_image(inputs)

        # Validate targets
        required_keys = ['mask_2d', 'indices', 'offset_2d', 'size_3d', 'depth', 'cls_ids', 'rotation_y', 'heading_bin', 'heading_res']
        for key in required_keys:
            if key not in targets:
                raise KeyError(f"Key '{key}' missing in targets.")

        if self.cfg is None or 'cls_mean_size' not in self.cfg or 'downsample' not in self.cfg:
            raise ValueError("Configuration 'cfg' with 'cls_mean_size' and 'downsample' is required.")

        valid_mask = targets['mask_2d'] == 1
        valid_indices = targets['indices'][valid_mask]
        valid_offset_2d = targets['offset_2d'][valid_mask]
        valid_size_3d = targets['size_3d'][valid_mask]
        valid_depth = targets['depth'][valid_mask]
        valid_cls_ids = targets['cls_ids'][valid_mask]
        valid_rotation_y = targets['rotation_y'][valid_mask]
        valid_heading_bin = targets['heading_bin'][valid_mask]
        valid_heading_res = targets['heading_res'][valid_mask]

        downsample_factor = self.cfg['downsample']
        feature_map_width = inputs.shape[2] // downsample_factor

        centers_x = ((valid_indices % feature_map_width) + valid_offset_2d[:, 0]) * downsample_factor
        centers_y = ((valid_indices // feature_map_width) + valid_offset_2d[:, 1]) * downsample_factor

        # Plot the image
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        # Iterate through each object and plot the centers, bounding boxes, IDs, and print details
        for i in range(len(valid_size_3d)):
            cls_id = valid_cls_ids[i]
            mean_size = self.cfg['cls_mean_size'][cls_id]

            size_3d = valid_size_3d[i] + mean_size  # [w, h, l]
            w_3d, h_3d, l_3d = size_3d  # Assuming order is [w, h, l]
            depth = float(valid_depth[i])  # Ensure depth is a scalar
            rotation_y = float(valid_rotation_y[i])
            heading_bin = int(valid_heading_bin[i])
            heading_res = float(valid_heading_res[i])

            # Convert heading_bin and heading_res back to rotation_y
            bin_size = (2 * np.pi) / num_bins
            reconstructed_rotation_y = bin_size * heading_bin + heading_res

            # Plot the center point
            ax.scatter(centers_x[i], centers_y[i], c='red', s=50, marker='.', label='Object Center' if i == 0 else "")

            # Annotate the object ID next to the center
            ax.text(centers_x[i] + 5, centers_y[i], f'ID: {i}', color='red', fontsize=12, weight='bold')

            # Print the object details
            print(f'Object ID: {i}')
            print(f'  Depth: {depth:.2f} meters')
            print(f'  3D Size (width, height, length): {w_3d:.2f}, {h_3d:.2f}, {l_3d:.2f}')
            print(f'  Rotation Y (Yaw): {rotation_y:.2f} radians')
            print(f'  Reconstructed Rotation Y: {reconstructed_rotation_y:.2f} radians')
            print(f'  Heading Bin: {heading_bin}, Heading Residual: {heading_res:.2f}')
            print('-' * 60)

        if len(valid_size_3d) > 0:
            ax.legend()
        ax.axis('off')
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close(fig)

    def project_to_image(self, pts_3d: np.ndarray) -> np.ndarray:
        """
        Projects 3D points to the 2D image plane using the calibration matrix.

        Parameters:
        - pts_3d: 3D points in camera coordinates of shape (N, 3).

        Returns:
        - pts_2d: 2D points in image coordinates of shape (N, 2).
        """
        if self.calib is None:
            raise ValueError("Calibration data is required for projecting 3D points to 2D.")

        # Add an extra column of ones to the 3D points for homogeneous coordinates
        pts_3d_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))

        # Project 3D points to 2D using the camera matrix P
        pts_2d_homo = pts_3d_homo.dot(self.calib.T)

        # Convert from homogeneous coordinates to 2D
        pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]

        return pts_2d

    def compute_box_3d(
        self,
        dimensions: Tuple[float, float, float],
        location: Tuple[float, float, float],
        rotation_y: float
    ) -> np.ndarray:
        """
        Computes the 3D bounding box corners in camera coordinates.

        Parameters:
        - dimensions: Tuple of (h, w, l).
        - location: Tuple of (x, y, z) representing the object center in camera coordinates.
        - rotation_y: Rotation around the y-axis (yaw).

        Returns:
        - corners_3d: Array of shape (8, 3) representing the 3D corners of the bounding box.
        """
        h, w, l = dimensions
        x, y, z = location

        # Create rotation matrix around the y-axis
        R_y = np.array([
            [ np.cos(rotation_y), 0, np.sin(rotation_y)],
            [               0, 1,              0],
            [-np.sin(rotation_y), 0, np.cos(rotation_y)]
        ])

        # Rotation matrix around the x-axis (camera pitch)
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(self.pitch), -np.sin(self.pitch)],
            [0, np.sin(self.pitch), np.cos(self.pitch)]
        ])

        # 3D bounding box corners in the object coordinate system
        x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
        y_corners = [    0,    0,     0,     0,    -h,    -h,    -h,    -h]
        z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]

        # Stack corners and apply rotation
        corners = np.vstack([x_corners, y_corners, z_corners])  # Shape: (3, 8)
        corners_3d = np.dot(R_y, corners)  # Apply yaw rotation (heading of the vehicle)
        corners_3d = np.dot(R_pitch, corners_3d)  # Apply pitch rotation (camera pitch)

        # Translate corners to the object's location
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z

        return corners_3d.transpose()  # Shape: (8, 3)

    def draw_3d_bboxes(
        self,
        inputs: np.ndarray,
        targets: Dict[str, Any],
        class_names: Optional[List[str]] = None,
        color: str = 'red',
        color_front: str = 'green',
        width: int = 2,
        display: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Draws 3D bounding boxes on the image, with the front face highlighted to indicate direction.

        Parameters:
        - inputs: Input image tensor of shape (C, H, W).
        - targets: Dictionary containing target components.
        - class_names: Optional list of class names, corresponding to the class IDs (cls_ids) in targets.
        - color: Color of the bounding box edges.
        - color_front: Color of the front face edges.
        - width: Line width of the bounding box edges.
        - display: If True, displays the image after drawing.
        - save_path: Optional path to save the visualized image.
        """
        self.logger.debug("Drawing 3D bounding boxes.")
        img = self._denormalize_image(inputs)

        # Validate targets
        required_keys = ['mask_2d', 'size_3d', 'position', 'rotation_y', 'cls_ids']
        for key in required_keys:
            if key not in targets:
                raise KeyError(f"Key '{key}' missing in targets.")

        if self.cfg is None or 'cls_mean_size' not in self.cfg:
            raise ValueError("Configuration 'cfg' with 'cls_mean_size' is required.")

        valid_mask = targets['mask_2d'] == 1
        valid_size_3d = targets['size_3d'][valid_mask]
        valid_position = targets['position'][valid_mask]
        valid_rotation_y = targets['rotation_y'][valid_mask]
        valid_cls_ids = targets['cls_ids'][valid_mask]

        # Compute depths for sorting (assuming Z-axis is depth)
        object_depths = valid_position[:, 2]  # Depth (z-coordinate)

        # Sort objects by depth (farther objects first)
        sorted_indices = np.argsort(-object_depths)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        for idx in sorted_indices:
            cls_id = valid_cls_ids[idx]
            mean_size = self.cfg['cls_mean_size'][cls_id]

            # Reconstruct actual 3D size [w, h, l]
            size_3d = valid_size_3d[idx] + mean_size  # [w, h, l]
            h_3d, w_3d, l_3d = size_3d  # Assuming order is [w, h, l]

            # Get position
            position = valid_position[idx]  # [x, y, z]

            rotation_y = valid_rotation_y[idx]

            # Compute the 3D corners of the bounding box
            corners_3d = self.compute_box_3d([h_3d, w_3d, l_3d], position, rotation_y)

            # Project the 3D corners to 2D image plane
            corners_2d = self.project_to_image(corners_3d)  # Shape: (8, 2)

            # Check if any of the projected points are within the image bounds
            if np.any((corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < img.shape[1]) &
                      (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < img.shape[0])):
                # Draw the edges
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                    (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
                ]
                for edge in edges:
                    x_coords = [corners_2d[edge[0], 0], corners_2d[edge[1], 0]]
                    y_coords = [corners_2d[edge[0], 1], corners_2d[edge[1], 1]]
                    ax.plot(x_coords, y_coords, color=color, linewidth=width)

                # Draw the front face with a different color
                front_face_indices = [0, 1, 5, 4]

                front_edges = [
                    (front_face_indices[0], front_face_indices[1]),
                    (front_face_indices[1], front_face_indices[2]),
                    (front_face_indices[2], front_face_indices[3]),
                    (front_face_indices[3], front_face_indices[0])
                ]
                for edge in front_edges:
                    x_coords = [corners_2d[edge[0], 0], corners_2d[edge[1], 0]]
                    y_coords = [corners_2d[edge[0], 1], corners_2d[edge[1], 1]]
                    ax.plot(x_coords, y_coords, color=color_front, linewidth=width)

                # Draw a cross on the front face to indicate direction
                ax.plot([corners_2d[front_face_indices[0], 0], corners_2d[front_face_indices[2], 0]],
                        [corners_2d[front_face_indices[0], 1], corners_2d[front_face_indices[2], 1]],
                        color=color_front, linewidth=width)
                ax.plot([corners_2d[front_face_indices[1], 0], corners_2d[front_face_indices[3], 0]],
                        [corners_2d[front_face_indices[1], 1], corners_2d[front_face_indices[3], 1]],
                        color=color_front, linewidth=width)

                # Optionally, draw the class name
                if class_names is not None and cls_id < len(class_names):
                    class_name = class_names[cls_id]
                    # Place the text at the center of the front face
                    front_face_center = np.mean(corners_2d[front_face_indices], axis=0)
                    ax.text(front_face_center[0], front_face_center[1], class_name, color='yellow', fontsize=12, weight='bold')
            else:
                self.logger.info(f"Object {idx} is outside the image bounds and will not be drawn.")

        ax.axis('off')
        if save_path:
            plt.savefig(save_path)
        if display:
            plt.show()
        plt.close(fig)

    