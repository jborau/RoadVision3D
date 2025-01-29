import numpy as np


import numpy as np
from roadvision3d.src.datasets.object_3d import affine_transform

def encode_targets(objects, calib, trans, features_size,
                   num_classes, max_objs, use_3d_center,
                   downsample, cls_mean_size, cls2id, writelist
                    ):
    """
    Encode 2D and 3D bounding box information into heatmap and regression targets.
    
    Args:
        objects       (List[Object3d]): List of annotated 3D objects for this sample.
        calib         (Calibration):    Calibration object (for 3D-to-2D projection).
        trans         (np.ndarray):     Affine transform matrix (2x3) for image coords.
        features_size (Tuple[int,int]): (W', H') final feature map size after downsample.
        num_classes   (int):            Number of classes to detect.
        max_objs      (int):            Maximum number of objects to detect.
        use_3d_center (bool):           Whether to use 3D center for heatmap placement.
        downsample    (int):            Downsample factor for final feature map.
        cls_mean_size (np.ndarray):     Mean 3D size for each class (cls_num, 3).
        cls2id        (Dict[str,int]):   Mapping from class name to class ID.
        writelist     (List[str]):       List of classes to detect.

    Returns:
        targets (dict): {
            'depth':         np.ndarray,  (max_objs, 1)
            'size_2d':       np.ndarray,  (max_objs, 2)
            'heatmap':       np.ndarray,  (cls_num, H', W')
            'offset_2d':     np.ndarray,  (max_objs, 2)
            'indices':       np.ndarray,  (max_objs,)        flattened indices in heatmap
            'size_3d':       np.ndarray,  (max_objs, 3)
            'offset_3d':     np.ndarray,  (max_objs, 2)
            'heading_bin':   np.ndarray,  (max_objs, 1)
            'heading_res':   np.ndarray,  (max_objs, 1)
            'cls_ids':       np.ndarray,  (max_objs,)
            'mask_2d':       np.ndarray,  (max_objs,)        indicates which objects are valid
            'vis_depth':     np.ndarray,  (max_objs, 7, 7)
            'rotation_y':    np.ndarray,  (max_objs,)
            'position':      np.ndarray,  (max_objs, 3)
            'size_3d_smoke': np.ndarray,  (max_objs, 3)
        }
    """

    # Prepare arrays for all targets
    heatmap = np.zeros((num_classes, features_size[1], features_size[0]), dtype=np.float32)
    size_2d = np.zeros((max_objs, 2), dtype=np.float32)
    offset_2d = np.zeros((max_objs, 2), dtype=np.float32)
    depth = np.zeros((max_objs, 1), dtype=np.float32)
    heading_bin = np.zeros((max_objs, 1), dtype=np.int64)
    heading_res = np.zeros((max_objs, 1), dtype=np.float32)
    src_size_3d = np.zeros((max_objs, 3), dtype=np.float32)
    size_3d = np.zeros((max_objs, 3), dtype=np.float32)
    size_3d_smoke = np.zeros((max_objs, 3), dtype=np.float32)
    offset_3d = np.zeros((max_objs, 2), dtype=np.float32)
    height2d = np.zeros((max_objs, 1), dtype=np.float32)
    cls_ids = np.zeros((max_objs), dtype=np.int64)
    indices = np.zeros((max_objs), dtype=np.int64)
    rotation_y = np.zeros((max_objs), dtype=np.float32)
    position = np.zeros((max_objs, 3), dtype=np.float32)

    # For mask, we can just use uint8 to indicate validity
    mask_2d = np.zeros((max_objs), dtype=np.uint8)
    vis_depth = np.zeros((max_objs, 7, 7), dtype=np.float32)

    # Process each object, up to max_objs
    object_num = min(len(objects), max_objs)
    for i in range(object_num):
        obj = objects[i]

        # Filter out classes not in writelist or invalid samples
        if obj.cls_type not in writelist:
            continue
        if obj.level_str == 'UnKnown' or obj.pos[-1] < 2:
            continue

        # Transform the 2D bounding box
        bbox_2d = obj.box2d.copy()
        bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
        bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
        bbox_2d[:] /= downsample

        w = bbox_2d[2] - bbox_2d[0]
        h = bbox_2d[3] - bbox_2d[1]
        center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2.0,
                              (bbox_2d[1] + bbox_2d[3]) / 2.0], dtype=np.float32)

        # Compute 3D center: (x, y - h/2, z)
        center_3d = obj.pos + [0, -obj.h / 2, 0]
        center_3d = center_3d.reshape(-1, 3)
        # Project to image plane
        center_3d, _ = calib.rect_to_img(center_3d)
        center_3d = affine_transform(center_3d[0], trans)
        center_3d /= downsample

        # Determine which center to place on heatmap
        if use_3d_center:
            c_hm = center_3d.astype(np.int32)
        else:
            c_hm = center_2d.astype(np.int32)

        # Check if center is valid in final feature map
        if (c_hm[0] < 0 or c_hm[0] >= features_size[0] or
            c_hm[1] < 0 or c_hm[1] >= features_size[1]):
            continue

        # Heatmap radius
        radius = gaussian_radius((w, h))
        radius = max(0, int(radius))

        # Possibly skip or merge certain classes
        if obj.cls_type in ['Van', 'Truck', 'DontCare']:
            # Example: place them in heatmap channel 1
            draw_umich_gaussian(heatmap[1], c_hm, radius)
            continue

        # Class ID
        cls_id = cls2id[obj.cls_type]
        cls_ids[i] = cls_id
        draw_umich_gaussian(heatmap[cls_id], c_hm, radius)

        # Fill in the regression targets
        indices[i] = c_hm[1] * features_size[0] + c_hm[0]
        offset_2d[i] = center_2d - c_hm
        size_2d[i] = [w, h]
        depth[i] = obj.pos[-1]

        # heading angle (alpha)
        heading_angle = calib.ry2alpha(obj.ry, 0.5*(obj.box2d[0] + obj.box2d[2]))
        # keep alpha in [-pi, pi]
        if heading_angle > np.pi:
            heading_angle -= 2 * np.pi
        if heading_angle < -np.pi:
            heading_angle += 2 * np.pi

        bin_idx, res_val = angle2class(heading_angle)
        heading_bin[i] = bin_idx
        heading_res[i] = res_val

        rotation_y[i] = obj.ry
        position[i] = obj.pos

        offset_3d[i] = center_3d - c_hm

        # 3D size
        src_size_3d[i] = np.array([obj.h, obj.w, obj.l], dtype=np.float32)
        mean_size = cls_mean_size[cls_id]
        size_3d[i] = src_size_3d[i] - mean_size
        size_3d_smoke[i] = src_size_3d[i] / mean_size

        # Visibility mask
        if obj.trucation <= 0.5 and obj.occlusion <= 2:
            mask_2d[i] = 1

        vis_depth[i] = depth[i]

    # Pack into a dictionary
    targets = {
        'depth': depth,
        'size_2d': size_2d,
        'heatmap': heatmap,
        'offset_2d': offset_2d,
        'indices': indices,
        'size_3d': size_3d,
        'offset_3d': offset_3d,
        'heading_bin': heading_bin,
        'heading_res': heading_res,
        'cls_ids': cls_ids,
        'mask_2d': mask_2d,
        'vis_depth': vis_depth,
        'rotation_y': rotation_y,
        'position': position,
        'size_3d_smoke': size_3d_smoke
    }
    return targets




num_heading_bin = 12  # hyper param
def check_range(angle):
    if angle > np.pi:  angle -= 2 * np.pi
    if angle < -np.pi: angle += 2 * np.pi
    return angle
    
def get_angle_from_box3d(box3d_pts_3d):
    direct_vec = (box3d_pts_3d[0]+box3d_pts_3d[1])/2-(box3d_pts_3d[2]+box3d_pts_3d[3])/2
    if direct_vec[0]>=0 and direct_vec[-1]>=0:
        angle = -np.arctan(direct_vec[-1]/direct_vec[0])
    elif direct_vec[0]<0 and direct_vec[-1]>=0:   
        angle = -(np.pi-np.arctan(np.abs(direct_vec[-1]/direct_vec[0])))
    elif direct_vec[0]<0 and direct_vec[-1]<0: 
        angle = np.pi-np.arctan(np.abs(direct_vec[-1]/direct_vec[0]))
    elif direct_vec[0]>=0 and direct_vec[-1]<0:
        angle = np.arctan(np.abs(direct_vec[-1]/direct_vec[0]))
    return angle  
def angle2class(angle):
    ''' Convert continuous angle to discrete class and residual. '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(cls, residual, to_label_format=False):
    ''' Inverse function to angle2class. '''
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def gaussian_radius(bbox_size, min_overlap=0.7):
    height, width = bbox_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap