import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

def draw_2d_bboxes(image, objects):
    for car in objects:
        image = draw_bbox_on_image(image, car.box2d)
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def draw_bbox_on_image(image, box2d):
    """
    Draws a 2D bounding box on a Pillow image object and returns the modified image.

    Parameters:
    - image: PIL.Image object, the image on which to draw.
    - bbox: tuple of (xmin, ymin, xmax, ymax), the coordinates of the bounding box.
    
    Returns:
    - image: PIL.Image object, the modified image with the bounding box drawn on it.
    """
    # Initialize the drawing context with the image as background
    draw = ImageDraw.Draw(image)
    
    # Draw the bounding box
    draw.rectangle(box2d, outline='red', width=2)

    # Return the modified image
    return image

def draw_3d_bbox_on_image(image, bbox, calib, camera_pitch):
    # Get the 3D bbox
    corners3d = bbox.generate_corners3d(camera_pitch=camera_pitch)
    
    # Project the 3D bbox to the image
    corners2d, depths = calib.rect_to_img(corners3d)  # Assuming calib.rect_to_img returns two arrays

    # Ensure corners2d is a list of tuples, not an array of arrays
    corners2d = [tuple(coord) for coord in corners2d]

    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Connect the points: 0-1-2-3-0 (bottom face) and 4-5-6-7-4 (top face)
    # Connect vertical edges: 0-4, 1-5, 2-6, 3-7
    bottom_face = [0, 1, 2, 3, 0]
    top_face = [4, 5, 6, 7, 4]
    vertical_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
    
    # Draw bottom and top faces
    for i in range(4):
        draw.line((corners2d[bottom_face[i]], corners2d[bottom_face[i+1]]), fill='red', width=4)
        draw.line((corners2d[top_face[i]], corners2d[top_face[i+1]]), fill='red', width=4)
    
    # Draw vertical edges
    for edge in vertical_edges:
        draw.line((corners2d[edge[0]], corners2d[edge[1]]), fill='red', width=4)

    return image

# Updated draw_3d_bboxes to include pitch information

def draw_3d_bboxes(image, labels, calib, pitch):
    for label in labels:
        image = draw_3d_bbox_on_image(image, label, calib, pitch)
    #     print(f"  Size 3D (h, w, l): {label.h, label.w, label.l}")
    #     print(f"  Position: {label.pos}")
    #     print(f"  Rotation Y (Yaw): {label.ry}")
    # print(calib.P2)
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()