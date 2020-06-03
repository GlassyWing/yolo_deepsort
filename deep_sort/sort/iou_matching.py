import torch
from .linear_assignment import INFTY_COST


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """

    bbox = bbox.unsqueeze(1)  # (n, 1, 4)
    candidates = candidates.unsqueeze(0)  # (1, m, 4)
    bbox_mins = bbox[..., :2]
    bbox_maxes = bbox[..., :2] + bbox[..., 2:]

    candidates_mins = candidates[..., :2]
    candidates_maxes = candidates[..., 2:] + candidates[..., :2]

    inter_mins = torch.max(bbox_mins, candidates_mins)  # (n, m, 2)
    inter_maxes = torch.min(bbox_maxes, candidates_maxes)  # (n, m, 2)

    inter_wh = torch.clamp(inter_maxes - inter_mins + 1, min=0)  # (n, m, 2)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]  # (n, m)
    bbox_area = bbox[..., 2] * bbox[..., 3]  # (n, m)
    candidates_area = candidates[..., 2] * candidates[..., 3]  # (n, m)

    return inter_area / (bbox_area + candidates_area - inter_area)  # (n, m)


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """

    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    candidates = []
    for i in detection_indices:
        candidates.append(detections[i].tlwh)
    candidates = torch.stack(candidates, dim=0)  # (m, 4)

    bboxes = []
    for track_idx in track_indices:
        bboxes.append(tracks[track_idx].to_tlwh())
    bboxes = torch.stack(bboxes, dim=0)  # (n, 4)

    cost_matrix = 1. - iou(bboxes, candidates)

    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = INFTY_COST
            continue

    return cost_matrix
