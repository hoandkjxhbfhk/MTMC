"""
Phiên bản bám sát PoseTracker gốc, chỉ bổ sung cluster refinement nội bộ bằng kế thừa.
- Không chỉnh sửa mã trong `Tracker/PoseTracker.py`.
- Ghi đè tối thiểu các điểm tạo track và cập nhật track.
"""

from typing import Any, List
import numpy as np
from collections import deque
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import procrustes

from util.camera import epipolar_3d_score_norm  # chỉ để giữ tương thích nếu cần
from util.process import find_view_for_cluster  # dùng trong target_init

from .PoseTracker import (
    PoseTrack as PoseTrackBase,
    PoseTracker as PoseTrackerBase,
    TrackState,
    Track2DState,
    Detection_Sample,
)


class PoseTrackRefined(PoseTrackBase):
    """Track đơn được refine bank ReID bằng clustering nội bộ.

    Bổ sung buffer quan sát tốt và tách cụm khi phát hiện lẫn nhiều người.
    """

    def __init__(self, cameras):
        super().__init__(cameras)
        self._features: deque[np.ndarray] = deque([], maxlen=10)
        self._poses: deque[np.ndarray] = deque([], maxlen=10)
        self._path_tlbr: dict[str, np.ndarray] = {}
        self._cluster_refiner = AgglomerativeClustering(
            n_clusters=2, metric="cosine", linkage="average"
        )

    def single_view_2D_update(self, v, sample, iou, ovr, ovr_tgt, avail_idx):
        # Gọi logic gốc
        super().single_view_2D_update(v, sample, iou, ovr, ovr_tgt, avail_idx)

        # Thu thập quan sát chất lượng cao cho refinement
        try:
            if (
                np.all(sample.keypoints_2d[self.upper_body, -1] > 0.5)
                and sample.bbox[4] > 0.9
                and np.sum(iou > 0.15) < 2
                and np.sum(ovr_tgt > 0.3) < 2
            ):
                feat = self.track2ds[v].reid_feat
                pose_kpts = sample.keypoints_2d
                tlbr = sample.bbox[:4]
                # Lọc trùng pose bằng Procrustes
                is_dup = False
                p1 = pose_kpts[:, :2]
                for p in self._poses:
                    try:
                        _, _, d = procrustes(p[:, :2], p1)
                        if d < 1e-8:
                            is_dup = True
                            break
                    except Exception:
                        pass
                if not is_dup:
                    if feat is not None:
                        self._features.append(feat)
                    self._poses.append(pose_kpts)
                    # key synthetics để debug/nhật ký (không dùng về sau)
                    self._path_tlbr[f"cam:{v}"] = tlbr
        except Exception:
            pass

    def multi_view_3D_update(self, avail_tracks: List[PoseTrackBase]):
        corr_v = super().multi_view_3D_update(avail_tracks)
        # Sau khi cập nhật 3D xong, refine bank nếu thấy lẫn người
        try:
            self._cluster_refinement_bank()
        except Exception:
            pass
        return corr_v

    def _cluster_refinement_bank(self) -> None:
        feats = np.array(self._features)
        if feats.shape[0] < 2:
            return
        # Phân cụm 2 nhánh để phát hiện lẫn người
        self._cluster_refiner.fit(feats)
        labels = self._cluster_refiner.labels_
        if len(set(labels)) < 2:
            return
        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]
        if idx0.size == 0 or idx1.size == 0:
            return
        a = feats[idx0]
        b = feats[idx1]
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
        sim = a_norm @ b_norm.T
        emb_dist = 1.0 - float(np.mean(sim))
        # Ngưỡng mặc định 0.3; có thể chỉnh theo scene ngoài vào nếu cần
        if emb_dist <= 0.3:
            return
        mean0 = float(np.mean(idx0))
        mean1 = float(np.mean(idx1))
        keep_idx = idx0 if mean0 <= mean1 else idx1
        kept = feats[keep_idx]
        # Ghi đè lại feat_bank bằng cụm được giữ
        new_bank = kept[: self.bank_size]
        self.feat_bank[: len(new_bank)] = new_bank
        self.feat_count = int(len(new_bank))


class PoseTracker(PoseTrackerBase):
    """Tracker đa-camera dùng PoseTrackRefined khi khởi tạo track mới.

    Mọi logic còn lại giữ nguyên từ PoseTracker gốc.
    """

    def target_init(
        self,
        detection_sample_list_mv,
        miss_tracks,
        iou_det_mv,
        ovr_det_mv,
        ovr_tgt_mv,
    ):
        # Bản sao tối giản từ gốc, chỉ thay PoseTrack(...) -> PoseTrackRefined(...)
        cam_idx_map = []  # cam_idx_map for per det
        det_count = []  # per view det count
        det_all_count = [0]
        for v in range(self.num_cam):
            det_count.append(len(detection_sample_list_mv[v]))
            det_all_count.append(det_all_count[-1] + det_count[-1])
            cam_idx_map += [v] * det_count[-1]

        if det_all_count[-1] == 0:
            return self.tracks

        det_num = det_all_count[-1]

        aff_homo = np.ones((det_num, det_num)) * (-10000)
        aff_epi = np.ones((det_num, det_num)) * (-10000)

        mv_rays = self.CalcJointRays(detection_sample_list_mv)
        feet_idxs = [15, 16]

        for vi in range(self.num_cam):
            samples_vi = detection_sample_list_mv[vi]
            pos_i = self.cameras[vi].pos
            for vj in range(vi, self.num_cam):
                if vi == vj:
                    continue
                else:
                    pos_j = self.cameras[vj].pos
                    samples_vj = detection_sample_list_mv[vj]

                    aff_temp = np.zeros((det_count[vi], det_count[vj]))
                    reid_sim_temp = np.zeros((det_count[vi], det_count[vj]))
                    aff_homo_temp = np.zeros((det_count[vi], det_count[vj]))
                    # calculate for each det pair
                    for a in range(det_count[vi]):
                        sample_a = samples_vi[a]
                        feet_valid_a = np.all(
                            sample_a.keypoints_2d[feet_idxs, -1] > self.keypoint_thrd
                        )
                        if feet_valid_a:
                            feet_a = np.mean(
                                sample_a.keypoints_2d[feet_idxs, :-1], axis=0
                            )
                            feet_a = self.cameras[vi].homo_feet_inv @ np.array(
                                [feet_a[0], feet_a[1], 1]
                            )
                            feet_a = feet_a[:-1] / feet_a[-1]
                        else:
                            feet_a = np.array(
                                [
                                    (sample_a.bbox[0] + sample_a.bbox[0]) / 2,
                                    sample_a.bbox[3],
                                ]
                            )
                            feet_a = self.cameras[vi].homo_inv @ np.array(
                                [feet_a[0], feet_a[1], 1]
                            )
                            feet_a = feet_a[:-1] / feet_a[-1]

                        feet_valid_a = True

                        for b in range(det_count[vj]):
                            sample_b = samples_vj[b]
                            aff = np.zeros(self.num_keypoints)
                            valid_kp = (
                                sample_a.keypoints_2d[:, -1] > self.keypoint_thrd
                            ) & (sample_b.keypoints_2d[:, -1] > self.keypoint_thrd)
                            j_id = np.where(valid_kp)[0]
                            aff[j_id] = epipolar_3d_score_norm(
                                pos_i,
                                mv_rays[vi][a][j_id, :],
                                pos_j,
                                mv_rays[vj][b][j_id, :],
                                self.thred_epi,
                            )

                            if feet_valid_a and np.all(
                                sample_b.keypoints_2d[feet_idxs, -1] > self.keypoint_thrd
                            ):
                                feet_b = np.mean(
                                    sample_b.keypoints_2d[feet_idxs, :-1], axis=0
                                )
                                feet_b = (
                                    self.cameras[vj].homo_feet_inv
                                    @ np.array([feet_b[0], feet_b[1], 1])
                                )
                                feet_b = feet_b[:-1] / feet_b[-1]

                                aff_homo_temp[a, b] = 1 - np.linalg.norm(feet_b - feet_a) / self.thred_homo
                            else:
                                feet_b = np.array(
                                    [
                                        (sample_b.bbox[0] + sample_b.bbox[0]) / 2,
                                        sample_b.bbox[3],
                                    ]
                                )
                                feet_b = (
                                    self.cameras[vj].homo_feet_inv
                                    @ np.array([feet_b[0], feet_b[1], 1])
                                )
                                feet_b = feet_b[:-1] / feet_b[-1]

                                aff_homo_temp[a, b] = 1 - np.linalg.norm(feet_b - feet_a) / self.thred_homo

                            aff_temp[a, b] = np.sum(
                                aff
                                * sample_a.keypoints_2d[:, -1]
                                * sample_b.keypoints_2d[:, -1]
                            ) / (
                                np.sum(
                                    valid_kp
                                    * sample_a.keypoints_2d[:, -1]
                                    * sample_b.keypoints_2d[:, -1]
                                )
                                + 1e-5
                            )
                            reid_sim_temp[a, b] = sample_a.reid_feat @ sample_b.reid_feat

                    aff_epi[
                        det_all_count[vi] : det_all_count[vi + 1],
                        det_all_count[vj] : det_all_count[vj + 1],
                    ] = aff_temp
                    aff_homo[
                        det_all_count[vi] : det_all_count[vi + 1],
                        det_all_count[vj] : det_all_count[vj + 1],
                    ] = aff_homo_temp

        aff_final = 2 * aff_epi + aff_homo
        aff_final[aff_final < -1000] = -np.inf

        clusters, sol_matrix = self.glpk_bip.solve(aff_final, True)

        for cluster in clusters:
            if len(cluster) == 1:
                view_list, number_list = find_view_for_cluster(cluster, det_all_count)
                det = detection_sample_list_mv[view_list[0]][number_list[0]]

                if (
                    det.bbox[-1] > 0.9
                    and np.all(det.keypoints_2d[self.main_joints, -1] > 0.5)
                    and np.sum(iou_det_mv[view_list[0]][number_list[0]] > 0.15) < 1
                    and np.sum(ovr_det_mv[view_list[0]][number_list[0]] > 0.3) < 2
                ):
                    new_track = PoseTrackRefined(self.cameras)
                    new_track.single_view_init(det, id=len(self.tracks) + 1)

                    self.match_with_miss_tracks(new_track, miss_tracks)

            else:
                view_list, number_list = find_view_for_cluster(cluster, det_all_count)
                sample_list = [
                    detection_sample_list_mv[view_list[idx]][number_list[idx]]
                    for idx in range(len(view_list))
                ]
                for i, sample in enumerate(sample_list):
                    if (
                        np.all(sample.keypoints_2d[self.main_joints, -1] > 0.5)
                        and sample.bbox[-1] > 0.9
                        and np.sum(iou_det_mv[view_list[i]][number_list[i]] > 0.15) < 1
                        and np.sum(ovr_det_mv[view_list[i]][number_list[i]] > 0.3) < 2
                    ):
                        new_track = PoseTrackRefined(self.cameras)
                        for j in range(len(view_list)):
                            new_track.iou_mv[view_list[j]] = iou_det_mv[view_list[j]][number_list[j]]
                            new_track.ovr_mv[view_list[j]] = ovr_det_mv[view_list[j]][number_list[j]]
                            new_track.ovr_tgt_mv[view_list[j]] = ovr_tgt_mv[view_list[j]][number_list[j]]

                        new_track.multi_view_init(sample_list, id=len(self.tracks) + 1)
                        self.match_with_miss_tracks(new_track, miss_tracks)
                        break


__all__ = ["PoseTracker", "PoseTrackRefined"]


