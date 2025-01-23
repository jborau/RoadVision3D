import torch
import torch.nn.functional as F
import torch.nn as nn


class PostProcessor:
    def __init__(self, det_threshold, max_detection):
        self.det_threshold = det_threshold
        self.max_detection = max_detection

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

        detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d], dim=2)
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

                preds.append(
                    [cls_id, 0.0] 
                    + bbox 
                    + [1.0, 1.0, 1.0]
                    + [0.0, 0.0, 0.0]
                    + [0.0, score]
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
    

def build_GUPNet_postprocessor(cfg):
    detections_threshold = 0.2
    detections_per_img = 50
    postprocessor = PostProcessor(
        detections_threshold,
        detections_per_img,
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