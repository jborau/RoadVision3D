import numpy as np
import cv2
import json
import math


# def get_ry_cam(ry_lidar, calib):
#     Ry_lidar = np.array([
#             [np.cos(ry_lidar), 0, np.sin(ry_lidar)],
#             [0, 1, 0],
#             [-np.sin(ry_lidar), 0, np.cos(ry_lidar)]
#         ])
    
#     R_v2c = calib.V2C[:, :3]  # Extract rotation matrix

#     # Transform the rotation matrix to camera coordinates
#     R_y_cam = R_v2c @ Ry_lidar

#     # Compute the rotation angle around Y-axis in camera coordinates
#     ry_cam = np.arctan2(R_y_cam[2, 0], R_y_cam[0, 0]) # + np.pi / 2

#     return ry_cam

def normalize_angle(angle):
    # make angle in range [-0.5pi, 1.5pi]
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan


def get_camera_3d_8points(obj_size, yaw_lidar, center_lidar, center_in_cam, r_velo2cam, t_velo2cam):
    liadr_r = np.matrix(
        [[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]]
    )
    l, w, h = obj_size
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar[:3]).reshape(3, 1)
    corners_3d_cam = r_velo2cam * corners_3d_lidar + t_velo2cam.reshape(3, 1)

    x0, z0 = corners_3d_cam[0, 0], corners_3d_cam[2, 0]
    x3, z3 = corners_3d_cam[0, 3], corners_3d_cam[2, 3]
    dx, dz = x0 - x3, z0 - z3
    yaw = math.atan2(-dz, dx)

    alpha = yaw - math.atan2(center_in_cam[0], center_in_cam[2])

    # add transfer
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi

    alpha_arctan = normalize_angle(alpha)

    return alpha_arctan, yaw


class Object3d(object):
    def __init__(self, cls_type, alpha, box2d, h, w, l, pos, ry, trucation=None, occlusion=None, score=None):
        self.cls_type = cls_type
        self.trucation = trucation
        self.occlusion = occlusion # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = alpha
        self.box2d = np.array(box2d, dtype=np.float32)
        self.h = h
        self.w = w
        self.l = l
        self.pos = np.array(pos, dtype=np.float32)
        self.ry = ry
        self.score = score if score is not None else -1.0
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.level_str = None
        self.level = self.get_obj_level()

    @classmethod
    def from_kitti_line(cls, line):
        label = line.strip().split(' ')
        cls_type = label[0]
        trucation = float(label[1])
        occlusion = float(label[2])
        alpha = float(label[3])
        box2d = (float(label[4]), float(label[5]), float(label[6]), float(label[7]))
        h = float(label[8])
        w = float(label[9])
        l = float(label[10])
        pos = (float(label[11]), float(label[12]), float(label[13]))
        ry = float(label[14])
        score = float(label[15]) if len(label) == 16 else -1.0

        return cls(cls_type, alpha, box2d, h, w, l, pos, ry, trucation, occlusion, score)
    
    @classmethod
    def from_dair_json(cls, obj, calib):
        cls_type = obj['type']
        trucation = float(obj['truncated_state'])
        occlusion = float(obj['occluded_state'])
        alpha = 0
        box2d = (obj['2d_box']['xmin'],obj['2d_box']['ymin'], obj['2d_box']['xmax'], obj['2d_box']['ymax'])
        h = float(obj['3d_dimensions']['h'])
        w = float(obj['3d_dimensions']['w'])
        l = float(obj['3d_dimensions']['l'])
        pos_lidar = np.array([
            float(obj['3d_location']['x']),
            float(obj['3d_location']['y']),
            float(obj['3d_location']['z']) - h / 2,  # Adjust for the height of the object
            1.0  # This is already a float
        ])
        ry = float(obj['rotation'])

        score = 0
        # Transform position to camera coordinates
        pos_cam = calib.V2C @ pos_lidar  # Shape: (3,)

        obj_size = (h, w, l)
        r_velo2cam = calib.V2C[:3, :3]  
        t_velo2cam = calib.V2C[:3, 3]
        alpha, ry_cam = get_camera_3d_8points(obj_size, ry, pos_lidar, pos_cam, r_velo2cam, t_velo2cam)

        return cls(cls_type, alpha, box2d, h, w, l, pos_cam, ry_cam, trucation, occlusion, score)
    
    @classmethod
    def from_RCooper_dair_json(cls, obj, calib):
        cls_type = obj['type']
        trucation = float(obj['truncated_state'])
        occlusion = float(obj['occluded_state'])
        alpha = 0
        box2d = (0.0, 0.0, 0.0, 0.0)
        h = float(obj['3d_dimensions']['h'])
        w = float(obj['3d_dimensions']['w'])
        l = float(obj['3d_dimensions']['l'])
        pos_lidar = np.array([
            float(obj['3d_location']['x']),
            float(obj['3d_location']['y']),
            float(obj['3d_location']['z']) - h / 2,  # Adjust for the height of the object
            1.0  # This is already a float
        ])
        ry = float(obj['rotation'])

        score = 0
        # Transform position to camera coordinates
        pos_cam = calib.V2C @ pos_lidar  # Shape: (3,)

        obj_size = (h, w, l)
        r_velo2cam = calib.V2C[:3, :3]  
        t_velo2cam = calib.V2C[:3, 3]
        alpha, ry_cam = get_camera_3d_8points(obj_size, ry, pos_lidar, pos_cam, r_velo2cam, t_velo2cam)

        return cls(cls_type, alpha, box2d, h, w, l, pos_cam, ry_cam, trucation, occlusion, score)

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if self.trucation == -1:
            self.level_str = 'DontCare'
            return 0

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4


    def generate_corners3d(self, camera_pitch=0.0):
        """
        Generate corners3d representation for this object, considering camera pitch.
        :param camera_pitch: Pitch rotation of the camera in radians.
        :return corners_3d: (8, 3) corners of box3d in camera coordinate system.
        """
        # Box dimensions
        l, h, w = self.l, self.h, self.w
        
        # Define the 8 corners of the bounding box in local coordinates
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # Rotation matrix around the y-axis (heading angle of the vehicle)
        R_y = np.array([
            [np.cos(self.ry), 0, np.sin(self.ry)],
            [0, 1, 0],
            [-np.sin(self.ry), 0, np.cos(self.ry)]
        ])

        # Rotation matrix around the x-axis (camera pitch)
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(camera_pitch), -np.sin(camera_pitch)],
            [0, np.sin(camera_pitch), np.cos(camera_pitch)]
        ])

        # Stack the corners and apply the rotations
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R_y, corners3d)  # Apply yaw rotation (heading of the vehicle)
        corners3d = np.dot(R_pitch, corners3d)  # Apply pitch rotation (camera pitch)
        
        # Translate the corners to the object's position
        corners3d = corners3d.T + self.pos
        
        return corners3d


    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m
        :param oblique:
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            box2d[:, 1] = Object3d.BEV_SHAPE[0] - 1 - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            cv = Object3d.BEV_SHAPE[0] - 1 - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d


    def __str__(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.pos, self.ry)
        return print_str
    
    def __repr__(self):
        return self.__str__()


    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.trucation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.pos[0], self.pos[1], self.pos[2],
                       self.ry)
        return kitti_str


##### Calibration #####

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

class Calibration(object):
    def __init__(self, P2, R0, V2C, C2V):
        self.P2 = P2
        self.R0 = R0  # 3 x 3
        self.V2C = V2C  # 3 x 4
        # self.C2V = self.inverse_rigid_trans(self.V2C)
        self.C2V = C2V

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)
    
    @classmethod
    def from_kitti_calib_file(cls, calib_file):
        if isinstance(calib_file, str):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        P2 = calib['P2']  # 3 x 4
        R0 = calib['R0']  # 3 x 3
        V2C = calib['Tr_velo2cam']  # 3 x 4
        C2V = cls.inverse_rigid_trans(cls, V2C)

        return cls(P2, R0, V2C, C2V)
    
    @classmethod
    def from_dair_calib_file(cls, camera_calib_file, lidar_calib_file):
        import numpy as np
        import json
        
        # Load camera calibration
        with open(camera_calib_file, 'r') as f:
            camera_calib = json.load(f)
        
        # Extract camera matrices
        cam_K = camera_calib['cam_K']
        K = np.array(cam_K, dtype=np.float32).reshape(3, 3)  # Ensure float32
        P2 = np.hstack((K, np.zeros((3, 1), dtype=np.float32)))  # Ensure float32
        R0_rect = np.eye(3, dtype=np.float32)  # Assuming images are rectified
        
        # Load LiDAR calibration
        with open(lidar_calib_file, 'r') as f:
            lidar_calib = json.load(f)
        
        # Extract LiDAR to camera transformation
        R_lidar_to_cam = np.array(lidar_calib['rotation'], dtype=np.float32)  # Ensure float32
        T_lidar_to_cam = np.array(lidar_calib['translation'], dtype=np.float32).reshape(3, 1)  # Ensure float32
        
        # Construct V2C
        V2C = np.hstack((R_lidar_to_cam, T_lidar_to_cam))  # 3×4 in float32
        
        # Compute C2V
        R_C2V = R_lidar_to_cam.T
        T_C2V = -R_C2V @ T_lidar_to_cam
        C2V = np.hstack((R_C2V, T_C2V))  # 3×4 in float32
        
        # Return Calibration instance with all matrices in float32
        return cls(P2.astype(np.float32), R0_rect.astype(np.float32), V2C.astype(np.float32), C2V.astype(np.float32))

    @classmethod
    def from_rope3d_calib_file(cls, calib_file):
        with open(calib_file) as f:
            line = f.readline().strip()
        
        # Extract P2 matrix values
        if line.startswith('P2:'):
            matrix_values = line.split(':')[1].strip().split()
            matrix_values = list(map(float, matrix_values))
            P2 = np.array(matrix_values, dtype=np.float32).reshape(3, 4)
        else:
            raise ValueError("P2 matrix not found in the calibration file.")

        # Initialize R0 and V2C as identity matrices in float32
        R0 = np.eye(3, dtype=np.float32)  # 3x3 identity matrix for R0
        V2C = np.eye(4, dtype=np.float32)[:3, :]  # 3x4 identity matrix for V2C

        # Compute C2V as the inverse of V2C
        C2V = cls.inverse_rigid_trans(cls, V2C)

        # Ensure all matrices are in float32
        return cls(P2.astype(np.float32), R0.astype(np.float32), V2C.astype(np.float32), C2V.astype(np.float32))
    
    @classmethod
    def from_rcooper_calib_file(cls, calib_file, cam_id="cam_0"):
        # Load the calibration file
        with open(calib_file, 'r') as f:
            calib_data = json.load(f)
        
        # Extract camera calibration data for the specified camera
        camera_data = calib_data[cam_id]
        
        # Camera intrinsic matrix
        K = np.array(camera_data['intrinsic'], dtype=np.float32).reshape(3, 3)
        P2 = np.hstack((K, np.zeros((3, 1), dtype=np.float32)))  # 3×4 matrix in float32
        
        # Assume images are rectified, so R0_rect is an identity matrix
        R0_rect = np.eye(3, dtype=np.float32)
        
        # Extract the extrinsic matrix (lidar-to-camera transformation)
        extrinsic = np.array(camera_data['extrinsic'], dtype=np.float32)
        
        # The rotation and translation components from lidar to camera
        R_lidar_to_cam = extrinsic[:3, :3]
        T_lidar_to_cam = extrinsic[:3, 3].reshape(3, 1)
        
        # Construct V2C as a 3x4 matrix
        V2C = np.hstack((R_lidar_to_cam, T_lidar_to_cam))  # 3×4 in float32
        
        # Compute the inverse transformation C2V (camera to lidar)
        R_C2V = R_lidar_to_cam.T
        T_C2V = -R_C2V @ T_lidar_to_cam
        C2V = np.hstack((R_C2V, T_C2V))  # 3×4 in float32
        
        # Return an instance of Calibration with all matrices
        return cls(P2.astype(np.float32), R0_rect.astype(np.float32), V2C.astype(np.float32), C2V.astype(np.float32))


    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_lidar(self, pts_rect):
        pts_ref = np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_rect)))
        pts_ref = self.cart_to_hom(pts_ref)  # nx4
        return np.dot(pts_ref, np.transpose(self.C2V))

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def depthmap_to_rect(self, depth_map):
        """
        :param depth_map: (H, W), depth_map
        :return:
        """
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        fd = np.sqrt((u - self.cu) ** 2 + (v - self.cv) ** 2 + self.fu ** 2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d ** 2 - x ** 2 - y ** 2)
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        return pts_rect

    def inverse_rigid_trans(self, Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def alpha2ry(self, alpha, u):
        """
        Get rotation_y by alpha + theta - 180
        alpha : Observation angle of object, ranging [-pi..pi]
        x : Object center x to the camera center (x-W/2), in pixels
        rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """
        ry = alpha + np.arctan2(u - self.cu, self.fu)

        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi

        return ry

    def ry2alpha(self, ry, u):
        alpha = ry - np.arctan2(u - self.cu, self.fu)

        if alpha > np.pi:
            alpha -= 2 * np.pi
        if alpha < -np.pi:
            alpha += 2 * np.pi

        return alpha
    def flip(self,img_size):      
        wsize = 4
        hsize = 2
        p2ds = (np.concatenate([np.expand_dims(np.tile(np.expand_dims(np.linspace(0,img_size[0],wsize),0),[hsize,1]),-1),\
                                np.expand_dims(np.tile(np.expand_dims(np.linspace(0,img_size[1],hsize),1),[1,wsize]),-1),
                                np.linspace(2,78,wsize*hsize).reshape(hsize,wsize,1)],-1)).reshape(-1,3)
        p3ds = self.img_to_rect(p2ds[:,0:1],p2ds[:,1:2],p2ds[:,2:3])
        p3ds[:,0]*=-1
        p2ds[:,0] = img_size[0] - p2ds[:,0]
        
        #self.P2[0,3] *= -1
        cos_matrix = np.zeros([wsize*hsize,2,7])
        cos_matrix[:,0,0] = p3ds[:,0]
        cos_matrix[:,0,1] = cos_matrix[:,1,2] = p3ds[:,2]
        cos_matrix[:,1,0] = p3ds[:,1]
        cos_matrix[:,0,3] = cos_matrix[:,1,4] = 1
        cos_matrix[:,:,-2] = -p2ds[:,:2]
        cos_matrix[:,:,-1] = (-p2ds[:,:2]*p3ds[:,2:3])
        new_calib = np.linalg.svd(cos_matrix.reshape(-1,7))[-1][-1]
        new_calib /= new_calib[-1]
        
        new_calib_matrix = np.zeros([4,3]).astype(np.float32)
        new_calib_matrix[0,0] = new_calib_matrix[1,1] = new_calib[0]
        new_calib_matrix[2,0:2] = new_calib[1:3]
        new_calib_matrix[3,:] = new_calib[3:6]
        new_calib_matrix[-1,-1] = self.P2[-1,-1]
        Tz = self.P2[2,3]
        # new_calib_matrix[2, :] = [0.0, 0.0, 1.0, self.P2[2, 3]] 
        self.P2 = new_calib_matrix.T
        self.P2[2, :] = [0.0, 0.0, 1.0, Tz]  # Correctly assign the third row
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv) 
    def affine_transform(self,img_size,trans):
        wsize = 4
        hsize = 2
        random_depth = np.linspace(2,78,wsize*hsize).reshape(hsize,wsize,1)
        p2ds = (np.concatenate([np.expand_dims(np.tile(np.expand_dims(np.linspace(0,img_size[0],wsize),0),[hsize,1]),-1),np.expand_dims(np.tile(np.expand_dims(np.linspace(0,img_size[1],hsize),1),[1,wsize]),-1),random_depth],-1)).reshape(-1,3)
        p3ds = self.img_to_rect(p2ds[:,0:1],p2ds[:,1:2],p2ds[:,2:3])
        p2ds[:,:2] = np.dot(np.concatenate([p2ds[:,:2],np.ones([wsize*hsize,1])],-1),trans.T)

        cos_matrix = np.zeros([wsize*hsize,2,7])
        cos_matrix[:,0,0] = p3ds[:,0]
        cos_matrix[:,0,1] = cos_matrix[:,1,2] = p3ds[:,2]
        cos_matrix[:,1,0] = p3ds[:,1]
        cos_matrix[:,0,3] = cos_matrix[:,1,4] = 1
        cos_matrix[:,:,-2] = -p2ds[:,:2]
        cos_matrix[:,:,-1] = (-p2ds[:,:2]*p3ds[:,2:3])
        new_calib = np.linalg.svd(cos_matrix.reshape(-1,7))[-1][-1]
        new_calib /= new_calib[-1]
        
        new_calib_matrix = np.zeros([4,3]).astype(np.float32)
        new_calib_matrix[0,0] = new_calib_matrix[1,1] = new_calib[0]
        new_calib_matrix[2,0:2] = new_calib[1:3]
        new_calib_matrix[3,:] = new_calib[3:6]
        new_calib_matrix[-1,-1] = self.P2[-1,-1]
        return new_calib_matrix.T
        #return new_calib_matrix.T       
        #print('{}-->{}'.format(ori_size,tar_size))
        #print(new_calib_matrix.T)
        #print(np.abs(p3ds[:,:2] - self.img_to_rect(p2ds[:,0:1],p2ds[:,1:2],p2ds[:,2:3])[:,:2]).max())
        #assert(np.abs(p3ds[:,:2] - self.img_to_rect(p2ds[:,0:1],p2ds[:,1:2],p2ds[:,2:3])[:,:2]).max()<1e-10)


##### affine transform #####

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    #scale all area
    # src_dir = get_dir([0, scale_tmp[1] * -0.5], rot_rad)
    # dst_dir = np.array([0, dst_h * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    #scale all area
    # src[2, :] = np.array([center[0] - 0.5 * scale_tmp[0], center[1] - 0.5 * scale_tmp[1]])
    # dst[2, :] = np.array([0, 0])

    if inv:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return trans, trans_inv
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])
def compute_box_3d(obj, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)    

    # 3d bounding box dimensions
    l = obj.l;
    w = obj.w;
    h = obj.h;
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    #y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + obj.pos[0];
    corners_3d[1,:] = corners_3d[1,:] + obj.pos[1];
    corners_3d[2,:] = corners_3d[2,:] + obj.pos[2];
    
    return np.transpose(corners_3d)