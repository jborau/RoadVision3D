from .smoke_coder import SMOKECoder
import torch
import torch.nn.functional as F
import torch.nn as nn

class PostProcessor:
    def __init__(self, smoke_coder, reg_head, det_threshold, max_detection, pred_2d):
        self.smoke_coder = smoke_coder
        self.reg_head = reg_head
        self.det_threshold = det_threshold
        self.max_detection = max_detection
        self.pred_2d = pred_2d

    def __call__(self, outputs, calibs, info, cls_mean_size):
        
        dets = self.extract_dets_from_outputs(outputs)
        dets = dets.detach().cpu().numpy()

        dets = self.decode_detections(dets = dets,
                        info = info,
                        calibs = calibs,
                        cls_mean_size=cls_mean_size,
                        threshold = 0.2)

        return dets

    def extract_dets_from_outputs(self, outputs):
        # get src outputs
        heatmap = outputs['heatmap']
        size_2d = outputs['size_2d']
        offset_2d = outputs['offset_2d']
        batch, channel, height, width = heatmap.size() # get shape

        heatmap = torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)
        # perform nms on heatmaps
        heatmap = self._nms(heatmap)
        scores, inds, cls_ids, xs, ys = self._topk(heatmap)

        offset_2d = _transpose_and_gather_feat(offset_2d, inds)
        offset_2d = offset_2d.view(batch, self.max_detection, 2)
        xs2d = xs.view(batch, self.max_detection, 1) + offset_2d[:, :, 0:1]
        ys2d = ys.view(batch, self.max_detection, 1) + offset_2d[:, :, 1:2]

        cls_ids = cls_ids.view(batch, self.max_detection, 1).float()
        scores = scores.view(batch, self.max_detection, 1)

        # check shape
        xs2d = xs2d.view(batch, self.max_detection, 1)
        ys2d = ys2d.view(batch, self.max_detection, 1)

        size_2d = _transpose_and_gather_feat(size_2d, inds)
        size_2d = size_2d.view(batch, self.max_detection, 2)

        # size 3d
        # Assuming 'size_3d_offset' is in outputs and has shape [batch, 3, height, width]
        size_3d = outputs['size_3d_offset']
        size_3d = _transpose_and_gather_feat(size_3d, inds)  # inds obtained from top-k
        size_3d = size_3d.view(batch, self.max_detection, 3).exp()

        # offset 3d
        offset_3d = outputs['offset_3d']
        offset_3d = _transpose_and_gather_feat(offset_3d, inds)  # inds obtained from top-k
        offset_3d = offset_3d.view(batch, self.max_detection, 2)
        xs3d = xs.view(batch, self.max_detection, 1) + offset_3d[:, :, 0:1]
        ys3d = ys.view(batch, self.max_detection, 1) + offset_3d[:, :, 1:2]


        # Depth
        depth_offsets = _transpose_and_gather_feat(outputs['depth'], inds)
        depth_offsets = depth_offsets.view(batch, self.max_detection, 1)
        # Decode depths using the same coder you used in training
        pred_depths = self.smoke_coder.decode_depth(depth_offsets).view(batch, self.max_detection, 1)

        # angle
        heading = _transpose_and_gather_feat(outputs['ori'], inds)
        heading = heading.view(batch, self.max_detection, 2)

        # detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, heading, size_3d, xs3d, ys3d, merge_depth, merge_conf], dim=2)
        detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, size_3d, xs3d, ys3d, pred_depths, heading], dim=2)
        return detections
    
    def decode_detections(self, dets, info, calibs, cls_mean_size, threshold, problist=None):
        '''
        NOTE: THIS IS A NUMPY FUNCTION
        input: dets, numpy array, shape in [batch x max_dets x dim]
        input: img_info, dict, necessary information of input images
        input: calibs, corresponding calibs for the input batch
        output:
        '''
        calibs = info['calibs']

        results = {}
        for i in range(dets.shape[0]):  # batch
            preds = []
            for j in range(dets.shape[1]):  # max_dets
                cls_id = int(dets[i, j, 0])
                score = dets[i, j, 1]
                if score < threshold: continue

                # 2d bboxs decoding
                x = dets[i, j, 2] * info['bbox_downsample_ratio'][i][0]
                y = dets[i, j, 3] * info['bbox_downsample_ratio'][i][1]
                w = dets[i, j, 4] * info['bbox_downsample_ratio'][i][0]
                h = dets[i, j, 5] * info['bbox_downsample_ratio'][i][1]
                bbox = [x-w/2, y-h/2, x+w/2, y+h/2]

                dimensions = dets[i, j, 6:9]
                dimensions = dimensions * cls_mean_size[int(cls_id)]
                if True in (dimensions<0.0): continue

                x3d = dets[i, j, 9] * info['bbox_downsample_ratio'][i][0]
                y3d = dets[i, j, 10]  * info['bbox_downsample_ratio'][i][1]
                depth = dets[i, j, 11]

                # Substitute this to training form
                locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
                locations[1] += dimensions[0] / 2

                pred_ori = dets[i, j, 12:14]
                pred_ori = torch.from_numpy(pred_ori).float()
                if pred_ori.dim() == 1:
                    pred_ori = pred_ori.unsqueeze(0)  # now shape is [1, 2]
                # Decode predicted orientation into rotys (and alphas, if needed)
                ry, alpha = self.smoke_coder.decode_orientation(pred_ori, torch.from_numpy(locations))

                preds.append(
                    [cls_id, alpha.item()] 
                    + bbox 
                    + dimensions.tolist() 
                    + locations.tolist() 
                    + [ry.item(), score]
                )

            results[info['img_id'][i]] = preds
        return results


    def _nms(self, heatmap, kernel=3):
        padding = (kernel - 1) // 2
        heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
        keep = (heatmapmax == heatmap).float()
        return heatmap * keep

    def _topk(self, heatmap):
        batch, cat, height, width = heatmap.size()

        # batch * cls_ids * 50
        topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), self.max_detection)

        topk_inds = topk_inds % (height * width)
        if torch.__version__ == '1.6.0':
            topk_ys = (topk_inds // width).int().float()
        else:
            topk_ys = (topk_inds / width).int().float()
        # topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # batch * cls_ids * 50
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.max_detection)
        if torch.__version__ == '1.6.0':
            topk_cls_ids = (topk_ind // self.max_detection).int()
        else:
            topk_cls_ids = (topk_ind / self.max_detection).int()
        # topk_cls_ids = (topk_ind / K).int()
        topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.max_detection)
        topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.max_detection)
        topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.max_detection)

        return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys

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

def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat