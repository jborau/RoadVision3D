from .smoke_coder import SMOKECoder
import torch
import torch.nn.functional as F

class PostProcessor:
    def __init__(self, smoke_coder, reg_head, det_threshold, max_detection, pred_2d):
        self.smoke_coder = smoke_coder
        self.reg_head = reg_head
        self.det_threshold = det_threshold
        self.max_detection = max_detection
        self.pred_2d = pred_2d

    def __call__(self, predictions, calibs, info, cls_mean_size):
        pred_heatmap, pred_regression = predictions[0], predictions[1]
        batch_size = pred_heatmap.shape[0]

        # Apply Non-Maximum Suppression (NMS) to the heatmap
        heatmap = self.nms_hm(pred_heatmap)

        # Select top K predictions from the heatmap
        scores, indices, clses, ys, xs = self.select_topk(heatmap, K=self.max_detection)

        # Select regression outputs at points of interest
        pred_regression_pois = self.select_point_of_interest(batch_size, indices, pred_regression)
        pred_regression_pois = pred_regression_pois.view(-1, self.reg_head)

        # Decode the regression outputs
        pred_proj_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)
        pred_depths_offset = pred_regression_pois[:, 0]
        pred_proj_offsets = pred_regression_pois[:, 1:3]
        pred_dimensions_offsets = pred_regression_pois[:, 3:6]
        pred_orientation = pred_regression_pois[:, 6:]

        pred_depths = self.smoke_coder.decode_depth(pred_depths_offset)

        pred_locations = self.smoke_coder.decode_location(
            pred_proj_points,
            pred_proj_offsets,
            pred_depths,
            calibs,
            downsample_ratio=4  # Adjust based on your model
        )

        pred_dimensions = self.smoke_coder.decode_dimension(
            clses,
            pred_dimensions_offsets
        )

        # Adjust Y-coordinate of the locations
        # pred_locations[:, 1] += pred_dimensions[:, 1] / 2

        pred_rotys, pred_alphas = self.smoke_coder.decode_orientation(
            pred_orientation,
            pred_locations
        )

        # # Prepare the final result tensor
        # clses = clses.view(-1, 1).float()
        # pred_alphas = pred_alphas.view(-1, 1)
        # pred_rotys = pred_rotys.view(-1, 1)
        # scores = scores.view(-1, 1)
        # pred_dimensions = pred_dimensions.roll(shifts=-1, dims=1)

        # result = torch.cat([
        #     clses, pred_alphas, pred_dimensions, pred_locations, pred_rotys, scores
        # ], dim=1)

        # # Apply detection threshold
        # keep_idx = result[:, -1] > self.det_threshold
        # result = result[keep_idx]

        # Now, convert the result tensor into the desired structure
        results = {}
        # Iterate over the batch to construct the final results
        for i in range(batch_size):
            preds = []
            for j in range(self.max_detection):
                score = scores[i * self.max_detection + j]
                if score < self.det_threshold:
                    continue

                cls_id = clses[i * self.max_detection + j].item()

                # 2D bounding box decoding
                x, y = xs[i * self.max_detection + j], ys[i * self.max_detection + j]
                w, h = pred_proj_offsets[i * self.max_detection + j]

                bbox = [
                    (x - w / 2).item() * info['bbox_downsample_ratio'][i][0],
                    (y - h / 2).item() * info['bbox_downsample_ratio'][i][1],
                    (x + w / 2).item() * info['bbox_downsample_ratio'][i][0],
                    (y + h / 2).item() * info['bbox_downsample_ratio'][i][1],
                ]

                # Decode depth, dimensions, and positions
                depth = pred_depths[i * self.max_detection + j].item()
                dimensions = pred_dimensions[i * self.max_detection + j].tolist()
                alpha = 0.00
                ry = pred_rotys[i * self.max_detection + j].tolist()

                locations = pred_locations[i * self.max_detection + j].tolist()

                preds.append([cls_id, alpha] + bbox + dimensions + locations + [ry, score.item()])  # Append the prediction
            results[info['img_id'][i]] = preds

        return results
    
    def nms_hm(self, heatmap, pool_size=3):
        """Non-Maximum Suppression for heatmaps"""
        pad = (pool_size - 1) // 2
        hmax = F.max_pool2d(heatmap, (pool_size, pool_size), stride=1, padding=pad)
        keep = (hmax == heatmap).float()
        return heatmap * keep

    def select_topk(self, heatmap, K=100):
        """Select top K scores and corresponding indices from the heatmap"""
        batch, cat, height, width = heatmap.size()
        heatmap = heatmap.view(batch, cat, -1)
        scores, indices = torch.topk(heatmap, K)
        clses = (indices / (height * width)).int()
        indices = indices % (height * width)
        ys = (indices / width).int().float()
        xs = (indices % width).int().float()
        # Adjust indices to be absolute indices in the batch
        indices = indices + (torch.arange(batch).view(batch, 1, 1) * height * width).type_as(indices)
        return scores.view(-1), indices.view(-1), clses.view(-1), ys.view(-1), xs.view(-1)

    def select_point_of_interest(self, batch_size, indices, pred_regression):
        """Select regression outputs corresponding to the top K points of interest"""
        # Flatten the regression outputs
        pred_regression_flat = pred_regression.view(batch_size, pred_regression.shape[1], -1)
        # Initialize a list to collect regression outputs
        pred_regression_pois = []
        for b in range(batch_size):
            # Get indices for the current batch
            batch_indices = indices[(indices // (pred_regression.shape[2] * pred_regression.shape[3])) == b]
            point_indices = batch_indices % (pred_regression.shape[2] * pred_regression.shape[3])
            # Get regression outputs at the selected points
            reg_pois = pred_regression_flat[b, :, point_indices].transpose(0, 1)
            pred_regression_pois.append(reg_pois)
        # Concatenate all regression outputs
        pred_regression_pois = torch.cat(pred_regression_pois, dim=0)
        return pred_regression_pois




def build_smoke_postprocessor(cfg, device):
    smoke_coder_depth_reference = (28.01, 16.32)
    smoke_coder_dimension_reference = cfg['dataset']['cls_mean_size']
    smoke_device = device

    smoke_coder = SMOKECoder(
        smoke_coder_depth_reference,
        smoke_coder_dimension_reference,
        smoke_device,
    )

    regression_heads = 8
    detections_threshold = cfg['tester']['threshold']
    detections_per_img = 50
    pred_2d = True

    postprocessor = PostProcessor(
        smoke_coder,
        regression_heads,
        detections_threshold,
        detections_per_img,
        pred_2d,
    )

    return postprocessor
