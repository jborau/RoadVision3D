import numpy as np
from PIL import Image

from roadvision3d.src.datasets.object_3d import get_affine_transform

class DataAugmention:
    """
    A class-based data augmentation module that currently supports:
      - Random horizontal flip
      - Random cropping (scaling & shifting the crop region)
    """
    def __init__(self, cfg, dataset=None):
        """
        Args:
            random_flip_prob  (float): Probability to apply random horizontal flip
            random_crop_prob  (float): Probability to apply random crop
            scale             (float): Max ratio for random scaling factor 
                                       (example: 0.4 means scale can be from 0.6 to 1.4)
            shift             (float): Max ratio for random shifting 
                                       (example: 0.1 means we shift up to 10% of image width/height)
        """
        self.random_flip_prob = cfg.get('random_flip', 0.5)
        self.random_crop_prob = cfg.get('random_crop', 0.5)
        self.random_mix_prob = cfg.get('random_mix', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

        # Dataset reference for random mix
        self.dataset = dataset
        self.max_objs = cfg.get('max_objs', 50)

    def __call__(self, img, calib, objects):
        """
        Perform random flip and random crop (if data_augmentation is True).
        
        Args:
            img               (PIL.Image): The original image
            calib             (Calibration): The Calibration object for current sample
            data_augmentation (bool): Whether to perform augmentation at all

        Returns:
            img          (PIL.Image): Possibly flipped or cropped image
            calib        (Calibration): Possibly flipped calibration
            center       (np.array): The final center used by the affine transform
            crop_size    (np.array): The final crop size used by the affine transform
            random_flip_flag (bool): Whether flip was performed
            random_crop_flag (bool): Whether crop was performed
        """
        img_size = np.array(img.size, dtype=np.float32)  # (W, H)
        center = img_size / 2.0
        crop_size = img_size.copy()


        # random flip
        flipped = False
        if np.random.rand() < self.random_flip_prob:
            flipped = True
            # Flip the image
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Flip the calibration
            calib.flip(img_size)
            for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:  object.ry -= 2 * np.pi
                    if object.ry < -np.pi: object.ry += 2 * np.pi

        # random crop
        if np.random.rand() < self.random_crop_prob:
            # random scaling
            scale_factor = np.clip(np.random.randn() * self.scale + 1,
                                    1 - self.scale, 
                                    1 + self.scale)
            crop_size = img_size * scale_factor

            # random shift
            shift_x = img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
            shift_y = img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

            center[0] += shift_x
            center[1] += shift_y

        if np.random.rand() < self.random_mix_prob:
            img, objects = self._try_random_mix(img, calib, objects, flipped, img_size)

        return img, calib, center, crop_size, objects

    def _try_random_mix(self, img, calib, objects, flipped, img_size):
        """
        Attempt random mix:
        - pick up to 50 random indices
        - check if second image's intrinsics match
        - check combined objects < self.max_objs
        - if success, alpha-blend & merge objects.
        Returns:
            img: The updated image after blending.
            objects: The updated list of objects.
        """
        for _ in range(50):
            random_index = np.random.randint(len(self.dataset.idx_list))
            random_index = int(self.dataset.idx_list[random_index])
            calib_temp = self.dataset.get_calib(random_index)

            # Check calibration compatibility
            if (calib_temp.cu == calib.cu and calib_temp.cv == calib.cv and
                calib_temp.fu == calib.fu and calib_temp.fv == calib.fv):
                
                img_temp = self.dataset.get_image(random_index)
                if img_temp.size == img.size:
                    objects_2 = self.dataset.get_label(random_index)

                    # Check if combined object count is within limits
                    if (len(objects) + len(objects_2)) < self.max_objs:
                        if flipped:
                            img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                            for obj in objects_2:
                                [x1, _, x2, _] = obj.box2d
                                obj.box2d[0], obj.box2d[2] = img_size[0] - x2, img_size[0] - x1
                                obj.ry = np.pi - obj.ry
                                obj.pos[0] *= -1
                                if obj.ry > np.pi:
                                    obj.ry -= 2 * np.pi
                                if obj.ry < -np.pi:
                                    obj.ry += 2 * np.pi

                        # Blend images and combine objects
                        img = Image.blend(img, img_temp, alpha=0.5)
                        objects += objects_2
                        return img, objects

        # If no mix was performed, return the original image and objects
        return img, objects