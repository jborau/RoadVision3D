# Object3D and Calibration in RoadVision3D

This document explains the **Object3D** and **Calibration** classes used in RoadVision3D. These modules handle 3D object representation and coordinate transformations between different reference frames.

> **Note:** This implementation is inspired by [MonoLSS](https://github.com/Traffic-X/MonoLSS).

---

## **1. Coordinate Systems in RoadVision3D**

RoadVision3D deals with multiple coordinate systems:
- **LiDAR Coordinate System**: Origin at the LiDAR sensor, with X-axis forward, Y-axis left, and Z-axis up.
- **Camera Coordinate System**: Origin at the camera, with X-axis right, Y-axis down, and Z-axis forward.
- **Image Coordinate System**: Pixel-based coordinate system in the 2D image plane.

Transformations between these coordinate systems are managed using calibration matrices.

---

## **2. Object3D Class**
The `Object3D` class represents a detected object in **3D space**, storing its position, orientation, and dimensions.

### **Attributes**
```python
def __init__(self, cls_type, alpha, box2d, h, w, l, pos, ry, trucation=None, occlusion=None, score=None):
```
- **cls_type**: Object category (e.g., 'Car', 'Pedestrian').
- **alpha**: Observation angle relative to the camera.
- **box2d**: 2D bounding box `[x_min, y_min, x_max, y_max]` in image space.
- **h, w, l**: Object dimensions (height, width, length).
- **pos**: Position in 3D space.
- **ry**: Rotation around the Y-axis (yaw).

### **Coordinate Transformations in Object3D**

#### **Convert LiDAR to Camera Coordinates**
```python
pos_cam = calib.V2C @ pos_lidar  # Transform LiDAR position to camera frame
```
- Uses the **V2C (LiDAR to Camera)** matrix.
- Converts **[x, y, z]** from **LiDAR** to **Camera frame**.

#### **From KITTI and DAIR-V2X Data**
- **KITTI**: Uses `from_kitti_line()` to parse labels.
- **DAIR-V2X**: Uses `from_dair_json()` with **calibration transformation**.

---

## **3. Calibration Class**
The `Calibration` class manages **transformation matrices** between coordinate systems.

### **Key Matrices**
```python
def __init__(self, P2, R0, V2C, C2V):
```
- **P2**: Camera **projection matrix** (maps 3D points to image plane).
- **R0**: **Rectification matrix**
- **V2C**: **LiDAR to Camera transformation matrix**.
- **C2V**: **Inverse Camera to LiDAR transformation**.

### **Coordinate Transformations in Calibration**

#### **LiDAR → Camera**
```python
def lidar_to_rect(self, pts_lidar):
    pts_rect = np.dot(self.V2C, self.cart_to_hom(pts_lidar).T).T
```
- Converts **LiDAR points** to **Camera frame** using **V2C**.

#### **Camera → Image**
```python
def rect_to_img(self, pts_rect):
    pts_2d_hom = np.dot(self.P2, self.cart_to_hom(pts_rect).T).T
```
- Projects **3D points** from the **camera frame** onto the **image plane**.

#### **Image → 3D** (Depth Map Backprojection)
```python
def img_to_rect(self, u, v, depth_rect):
    x = ((u - self.cu) * depth_rect) / self.fu
    y = ((v - self.cv) * depth_rect) / self.fv
```
- Reconstructs **3D world coordinates** from **2D image points** and depth.

---

## **4. Parsing Calibration Files**

#### **From KITTI**
```python
@classmethod
def from_kitti_calib_file(cls, calib_file):
    calib = get_calib_from_file(calib_file)
    return cls(calib['P2'], calib['R0'], calib['Tr_velo2cam'], cls.inverse_rigid_trans(calib['Tr_velo2cam']))
```
- Reads **KITTI calibration** and extracts transformation matrices.

#### **From DAIR-V2X**
```python
@classmethod
def from_dair_calib_file(cls, camera_calib_file, lidar_calib_file):
```
- Reads **camera and LiDAR JSON** calibration from **DAIR-V2X**.

#### **From ROPE3D**
```python
@classmethod
def from_rope3d_calib_file(cls, calib_file):
```
- Extracts **projection and transformation matrices** from **ROPE3D**.
---

