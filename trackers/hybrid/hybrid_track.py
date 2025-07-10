from typing import List, Optional
import numpy as np

# Sử dụng lại các thành phần từ multicam_tracker
from trackers.multicam_tracker.cluster_track import MCTracker as _BaseMCTracker, MTrack
from trackers.multicam_tracker import matching
from trackers.multicam_tracker.basetrack import TrackState
from trackers.multicam_tracker.cluster_track import grouping_rerank
from .kalman_utils import KalmanFilter_box

from .geometry_utils import triangulate, p2l_distance
from .config import HybridConfig


class HybridTrack(MTrack):
    """MTrack mở rộng – thêm dữ liệu hình học 3D và quản lý tuổi."""

    def __init__(self, *args, cameras: Optional[List[object]] = None, config: Optional[HybridConfig] = None, **kwargs):
        # --- basic geometry & reid bank ---
        self.cameras = cameras  # reference tới list Camera của hệ thống
        self.num_cam = len(cameras) if cameras is not None else 0
        self.num_kp = 17  # default COCO
        self.keypoints_mv = np.zeros((self.num_cam, self.num_kp, 3))
        self.keypoints_3d = np.zeros((self.num_kp, 4))
        # ReID feature bank
        self.bank_size = 100
        self.feat_bank = np.zeros((self.bank_size, 2048), dtype=np.float32)
        self.feat_count = 0
        self.thred_reid = 0.5

        # Ageing
        self.cfg = config if config is not None else HybridConfig()
        self.age_2D = np.full((self.num_cam, self.num_kp), np.inf)
        self.age_3D = np.full(self.num_kp, np.inf)
        self.age_bbox = np.full(self.num_cam, np.inf)
        self.dura_bbox = np.zeros(self.num_cam)
        # track age (frames since last successful update)
        self.update_age = 0

        # Kalman filter cho bbox mỗi camera
        self.bbox_kalman = [KalmanFilter_box() for _ in range(self.num_cam)]
        # Occlusion state theo từng camera
        self.oc_state = [False for _ in range(self.num_cam)]

        super().__init__(*args, **kwargs)

        # re-init again in case super changed cams
        self.num_cam = len(cameras) if cameras is not None else 0
        # ensure arrays sizes ok (may be 0 before cameras set)
        if self.age_2D.shape[0] != self.num_cam:
            self.age_2D = np.full((self.num_cam, self.num_kp), np.inf)
            self.age_bbox = np.full(self.num_cam, np.inf)

        # NOTE: update_features may have filled keypoints already in super().__init__

    def update_geometry(self):
        if self.cameras is None or self.num_cam < 2:
            return
        self.keypoints_3d, _ = triangulate(self.keypoints_mv, self.cameras)
        # reset age_3D for valid 3D points
        valid3d = self.keypoints_3d[:, -1] != 0
        self.age_3D[valid3d] = 0

    # override nếu cần update features để lưu keypoints_mv
    def update_features(self, features, poses, img_paths, tlbrs, coords):
        # --- ReID bank update với điều kiện lọc ---------------------
        if features is not None:
            upper_idx = [5,6,11,12]
            for idx, feat in enumerate(features):
                feat_vec = np.asarray(feat).flatten()
                if feat_vec.size != 2048:
                    continue

                # --- Các điều kiện lọc bổ sung ---
                # 1) upper-body joints đủ tin cậy
                pose_i = poses[idx] if poses is not None and idx < len(poses) else None
                if pose_i is None:
                    continue
                kps = pose_i["keypoints"]
                if not np.all(kps[upper_idx,2] > 0.5):
                    continue

                # 2) bbox score > 0.9
                tlbr_i = tlbrs[idx] if tlbrs is not None and idx < len(tlbrs) else None
                score = tlbr_i[4] if tlbr_i is not None and len(tlbr_i) >=5 else 1.0
                if tlbr_i is None or score <= 0.9:
                    continue

                # 3) không occlusion tại camera đó
                cam_id = pose_i.get("cam_id", 0)
                not_occluded = (cam_id >= len(self.oc_state)) or (not self.oc_state[cam_id])
                if not not_occluded:
                    continue

                # --- Lọc trùng lặp theo cosine similarity như cũ ---
                if self.feat_count:
                    bank = self.feat_bank[: self.feat_count % self.bank_size]
                    norms = np.linalg.norm(bank, axis=1) * (np.linalg.norm(feat_vec) + 1e-6)
                    sims = (bank @ feat_vec) / (norms + 1e-6)
                    sim = np.max(sims) if sims.size else 0.0
                    if sim >= self.thred_reid:
                        continue  # quá giống, bỏ qua

                # add to bank
                self.feat_bank[self.feat_count % self.bank_size] = feat_vec
                self.feat_count += 1

        # cập nhật deque features để tương thích cost cũ
        self.features.clear()
        valid_num = min(self.feat_count, 10)
        if valid_num:
            idxs = (np.arange(self.feat_count - valid_num, self.feat_count)) % self.bank_size
            for idx in idxs:
                self.features.append(self.feat_bank[idx])

        # Gọi logic gốc để lưu path_tlbr, poses ...
        super().update_features(features, poses, img_paths, tlbrs, coords)

        # --- Update Kalman bbox regardless of camera availability ----
        if tlbrs is not None and len(tlbrs):
            # giả định cam_id lấy từ pose hoặc 0
            cam_id = 0
            if poses is not None and len(poses):
                p0 = poses[0]
                if p0 is not None:
                    cam_id = p0.get("cam_id", 0)
            cam_id = int(cam_id) if cam_id < self.num_cam else 0
            tlbr0 = tlbrs[0]
            if tlbr0 is not None and cam_id < len(self.bbox_kalman):
                self.bbox_kalman[cam_id].update(tlbr0[:4])

        # cập nhật keypoints_mv + reset age nếu cameras được cung cấp
        if poses is None or self.cameras is None:
            return
        for idx, p in enumerate(poses):
            if p is None:
                continue
            cam_id = p.get('cam_id', None)
            if cam_id is None:
                continue
            if cam_id < self.num_cam:
                kp = p['keypoints']
                self.keypoints_mv[cam_id] = kp
                # reset age for valid joints & bbox
                valid = kp[:,2] > 0.5
                self.age_2D[cam_id][valid] = 0
                self.age_bbox[cam_id] = 0
                self.dura_bbox[cam_id] += 1
                # cập nhật KalmanFilter với bbox đo được (nếu tlbrs có)
                if idx < len(tlbrs):
                    tlbr = tlbrs[idx]
                    if tlbr is not None:
                        self.bbox_kalman[cam_id].update(tlbr[:4])
        # reset age because just updated
        self.update_age = 0

    # -----------------------------------------------------------------
    def increment_age(self):
        """Tăng tuổi mỗi frame và làm sạch view/bbox khi vượt ngưỡng."""
        self.age_2D += 1
        self.age_3D += 1
        self.age_bbox += 1
        self.update_age += 1

        # drop joints quá cũ
        self.age_2D[self.age_2D >= self.cfg.max_age_2d] = np.inf

        # drop view quá cũ
        for v in range(self.num_cam):
            if self.age_bbox[v] >= self.cfg.max_age_bbox:
                self.keypoints_mv[v] = 0
                self.age_2D[v] = np.inf
                self.age_bbox[v] = np.inf
                self.dura_bbox[v] = 0
                self.oc_state[v] = False

    # -------------------------------------------------------------
    def switch_view(self, other: "HybridTrack", cam_id: int):
        """Trao đổi dữ liệu của view cam_id giữa hai track (giống PoseTracker)."""
        if cam_id >= self.num_cam or cam_id >= other.num_cam:
            return
        # Swap per-view arrays
        self.keypoints_mv[cam_id], other.keypoints_mv[cam_id] = (other.keypoints_mv[cam_id].copy(), self.keypoints_mv[cam_id].copy())
        self.age_2D[cam_id], other.age_2D[cam_id] = (other.age_2D[cam_id].copy(), self.age_2D[cam_id].copy())
        self.age_bbox[cam_id], other.age_bbox[cam_id] = other.age_bbox[cam_id], self.age_bbox[cam_id]
        self.dura_bbox[cam_id], other.dura_bbox[cam_id] = other.dura_bbox[cam_id], self.dura_bbox[cam_id]
        # Swap Kalman filter object
        self.bbox_kalman[cam_id], other.bbox_kalman[cam_id] = other.bbox_kalman[cam_id], self.bbox_kalman[cam_id]
        # Occlusion flags
        self.oc_state[cam_id], other.oc_state[cam_id] = other.oc_state[cam_id], self.oc_state[cam_id]


class HybridMCTracker(_BaseMCTracker):
    """Phiên bản kết hợp: thêm chi phí hình học khi matching."""

    def __init__(self, cameras: Optional[List[object]] = None, config: Optional[HybridConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.cameras = cameras
        self.cfg = config if config is not None else HybridConfig()

    # ---------------------------------------------------------------
    # Helper methods làm gọn hàm update
    # ---------------------------------------------------------------

    # 1) Tạo danh sách HybridTrack từ input groups
    def _build_new_groups(self, groups, scene):
        if len(groups):
            global_ids, features, centroids, poses, paths, tlbrs, coords = (
                groups[:, 0], groups[:, 1], groups[:, 2], groups[:, 3], groups[:, 4], groups[:, 5], groups[:, 6]
            )
        else:
            return []

        return [
            HybridTrack(g, c, f, p, ph, t, cd, self.min_hits, scene, cameras=self.cameras, config=self.cfg)
            for (g, c, f, p, ph, t, cd) in zip(global_ids, centroids, features, poses, paths, tlbrs, coords)
        ]

    # 2) Chia track thành tracked & unconfirmed
    def _split_tracks(self):
        unconfirmed, tracked = [], []
        for trk in self.tracked_mtracks:
            (unconfirmed if not trk.is_activated else tracked).append(trk)
        return tracked, unconfirmed

    # 3) Quick match bằng global_id – trả về list mới đã loại matched
    def _quick_match_gid(self, tracked_mtracks, new_groups, frame_id):
        activated, matched_new_idxs = [], []
        gid2track = {t.global_id: t for t in tracked_mtracks}
        for idx, g in enumerate(new_groups):
            if g.global_id in gid2track:
                trk = gid2track[g.global_id]
                trk.update(g, frame_id)
                activated.append(trk)
                matched_new_idxs.append(idx)

        new_groups_remain = [g for i, g in enumerate(new_groups) if i not in matched_new_idxs]
        tracked_remain = [t for t in tracked_mtracks if t not in activated]
        return activated, tracked_remain, new_groups_remain

    # 4) Tính ma trận distance & ánh xạ – dùng cho 2 phase (tracked/unconfirmed)
    def _associate(self, src_tracks, dst_groups, full_cost=True, thresh=None):
        # Điều chỉnh trọng số động trước khi tính distance
        if full_cost:
            self._adjust_weights(src_tracks)

        if thresh is None:
            thresh = self.cfg.thresh_match
        shape = (len(src_tracks), len(dst_groups))
        if 0 in shape:
            dists = np.empty(shape)
        else:
            # features & centroid lists
            src_feats = [feat for m in src_tracks for feat in list(m.features)]
            len_src = [len(m.features) for m in src_tracks]
            dst_feats = [feat for g in dst_groups for feat in list(g.features)]
            len_dst = [len(g.features) for g in dst_groups]
            # Tính khoảng cách embedding với grouping_rerank đã import sẵn
            rerank_dists = matching.embedding_distance(src_feats, dst_feats) / 2.0
            emb_dists = grouping_rerank(rerank_dists, len_src, len_dst, shape, normalize=False)

            if full_cost:
                src_centroids = [m.centroid for m in src_tracks]
                dst_centroids = [g.centroid for g in dst_groups]
                euc_dists = matching.euclidean_distance(src_centroids, dst_centroids)

                geo_d = self._geometry_distance(src_tracks, dst_groups)
                p2l_d = self._p2l_distance_matrix(src_tracks, dst_groups)
                kp2d_d = self._kp2d_distance_matrix(src_tracks, dst_groups)
                iou_d = self._iou_distance_matrix(src_tracks, dst_groups)
                feet_d = self._feet_distance_matrix(src_tracks, dst_groups)
                epi_d = self._epi_distance_matrix(src_tracks, dst_groups)

                # normalize (dùng helper chia sẻ)
                norm_emb = self._norm_matrix(emb_dists)
                norm_euc = self._norm_matrix(euc_dists)
                norm_geo = self._norm_matrix(geo_d) if geo_d.size else np.zeros_like(norm_emb)

                c = self.cfg
                dists = (
                    c.w_reid * norm_emb
                    + c.w_euc * norm_euc
                    + c.w_geo * norm_geo
                    + c.w_iou * iou_d
                    + c.w_p2l * p2l_d
                    + c.w_kp2d * kp2d_d
                    + c.w_feet * feet_d
                    + c.w_epi * epi_d
                )
            else:
                dists = emb_dists

        matches, u_src, u_dst = matching.linear_assignment(dists, thresh=thresh)
        return matches, u_src, u_dst

    # -----------------------------------------------------------------
    def _adjust_weights(self, tracks: List["HybridTrack"], border_thresh: float = 50, age_thresh: int = 10):
        """Điều chỉnh trọng số cfg dựa trên tình trạng track.

        - Giảm w_reid khi track quá cũ hoặc đang occlusion.
        - Tăng w_feet và w_geo nếu bbox gần biên ảnh.
        Copy cfg gốc sang biến local để không phá vỡ cấu hình toàn cục.
        """
        c = self.cfg
        # lưu baseline nếu chưa có
        if not hasattr(self, "_base_weights"):
            self._base_weights = dict(w_reid=c.w_reid, w_feet=c.w_feet, w_geo=c.w_geo)

        # reset về baseline
        c.w_reid = self._base_weights["w_reid"]
        c.w_feet = self._base_weights["w_feet"]
        c.w_geo = self._base_weights["w_geo"]

        # heuristic: nếu bất kỳ track thoả điều kiện, điều chỉnh weight toàn batch
        need_reduce_reid = any(t.update_age > age_thresh or any(t.oc_state) for t in tracks if isinstance(t, HybridTrack))
        if need_reduce_reid:
            c.w_reid *= 0.5  # giảm một nửa

        # tăng feet/geo khi có track sát biên ảnh (giả sử tlbr trong path_tlbr lớn nhất)
        near_border = False
        for t in tracks:
            if not t.path_tlbr:
                continue
            # bbox lớn nhất
            bbox = list(t.path_tlbr.values())[0]
            x1, y1, x2, y2 = bbox[:4]
            if x1 < border_thresh or y1 < border_thresh:
                near_border = True; break
        if near_border:
            c.w_feet *= 1.5
            c.w_geo *= 1.5

    # 5) Cập nhật các list tracked/lost sau mỗi vòng
    def _merge_lists(self, activated, refind, lost, removed):
        from trackers.multicam_tracker.cluster_track import joint_mtracks, sub_mtracks

        self.tracked_mtracks = [t for t in self.tracked_mtracks if t.state == TrackState.Tracked]
        self.tracked_mtracks = joint_mtracks(self.tracked_mtracks, activated)
        self.tracked_mtracks = joint_mtracks(self.tracked_mtracks, refind)

        self.lost_mtracks = sub_mtracks(self.lost_mtracks, self.tracked_mtracks)
        self.lost_mtracks.extend(lost)
        self.lost_mtracks = sub_mtracks(self.lost_mtracks, removed)

        return [t for t in self.tracked_mtracks if t.is_activated]

    # --- Hàm tiện ích -----------------------------------------------------
    def _geometry_distance(self, tracks: List[MTrack], groups: List[MTrack]) -> np.ndarray:
        """Tính ma trận khoảng cách hình học giữa danh sách track và detection groups."""
        if self.cameras is None:
            # nếu không có calib, trả về ma trận 0
            return np.zeros((len(tracks), len(groups)))
        dists = np.zeros((len(tracks), len(groups)))
        for i, trk in enumerate(tracks):
            for j, grp in enumerate(groups):
                if grp.centroid is None or trk.centroid is None:
                    dists[i, j] = 1e3
                    continue
                dists[i, j] = np.linalg.norm(np.array(trk.centroid) - np.array(grp.centroid))
        if dists.size:
            maxv, minv = np.max(dists), np.min(dists)
            if maxv > minv:
                dists = (dists - minv) / (maxv - minv)
        # áp dụng decay theo tuổi track (số frame kể từ lần update cuối)
        for i, trk in enumerate(tracks):
            decay = 1.0 + trk.update_age / (self.cfg.max_age_bbox + 1e-6)
            dists[i] *= decay
        return dists

    # ----------------- New affinity helpers ------------------------------
    def _p2l_distance_matrix(self, tracks: List[MTrack], groups: List[MTrack]) -> np.ndarray:
        if self.cameras is None:
            return np.ones((len(tracks), len(groups)))
        from .geometry_utils import p2l_distance_set
        dmat = np.ones((len(tracks), len(groups))) * 1e6
        for i, trk in enumerate(tracks):
            for j, grp in enumerate(groups):
                pose_list = grp.curr_pose if grp.curr_pose is not None else []
                if not pose_list:
                    continue
                pose = pose_list[0]
                cam_id = pose.get('cam_id', 0)
                if cam_id >= len(self.cameras):
                    continue
                d = p2l_distance_set(trk.keypoints_3d, pose, self.cameras[cam_id])
                dmat[i, j] = d
        if dmat.size:
            maxv, minv = np.max(dmat), np.min(dmat)
            if maxv > minv:
                dmat = (dmat - minv) / (maxv - minv)
        # decay theo tuổi 3D (khớp 3D càng cũ khoảng cách càng lớn)
        for i, trk in enumerate(tracks):
            age3d = np.nanmin(trk.age_3D) if np.any(np.isfinite(trk.age_3D)) else self.cfg.max_age_2d
            decay = 1.0 + age3d / (self.cfg.max_age_2d + 1e-6)
            dmat[i] *= decay
        return dmat

    def _kp2d_distance_matrix(self, tracks: List[MTrack], groups: List[MTrack]) -> np.ndarray:
        if self.cameras is None:
            return np.ones((len(tracks), len(groups)))
        dmat = np.ones((len(tracks), len(groups))) * 1.0
        for i, trk in enumerate(tracks):
            for j, grp in enumerate(groups):
                pose_list = grp.curr_pose if grp.curr_pose is not None else []
                if not pose_list:
                    continue
                pose = pose_list[0]
                cam_id = pose.get('cam_id', 0)
                if cam_id >= trk.keypoints_mv.shape[0]:
                    continue
                kp_t = trk.keypoints_mv[cam_id]
                kp_s = pose['keypoints']
                valid = (kp_t[:,2] > 0.5) & (kp_s[:,2] > 0.5) & (trk.age_2D[cam_id] < np.inf)
                if not np.any(valid):
                    continue
                # Áp dụng decay theo tuổi khớp 2D để giảm ảnh hưởng của điểm cũ
                dif = kp_t[valid, :2] - kp_s[valid, :2]
                # hệ số decay: exp(-age)
                decay = np.exp(-trk.age_2D[cam_id][valid])
                w = np.linalg.norm(dif, axis=1)
                dist = np.sum(w * decay) / (np.sum(decay) + 1e-6)
                dmat[i, j] = dist
        if dmat.size:
            maxv, minv = np.max(dmat), np.min(dmat)
            if maxv > minv:
                dmat = (dmat - minv) / (maxv - minv)
        # decay theo tuổi 2D (các khớp trong views còn mới thì đáng tin hơn)
        for i, trk in enumerate(tracks):
            age2d_min = np.nanmin(trk.age_2D) if np.any(np.isfinite(trk.age_2D)) else self.cfg.max_age_2d
            decay = 1.0 + age2d_min / (self.cfg.max_age_2d + 1e-6)
            dmat[i] *= decay
        return dmat

    def _iou_distance_matrix(self, tracks: List[MTrack], groups: List[MTrack]) -> np.ndarray:
        """Tính IoU distance giữa bbox đại diện (lớn nhất) của track và detection."""
        from trackers.multicam_tracker.matching import ious
        track_boxes = []
        for t in tracks:
            if t.path_tlbr:
                # lấy bbox có diện tích max
                max_key = max(t.path_tlbr, key=lambda k: (t.path_tlbr[k][2]-t.path_tlbr[k][0])*(t.path_tlbr[k][3]-t.path_tlbr[k][1]))
                track_boxes.append(t.path_tlbr[max_key])
            elif hasattr(t, 'tlbrs') and t.tlbrs:
                track_boxes.append(t.tlbrs[0])
            else:
                track_boxes.append(np.array([0,0,0,0]))
        det_boxes = []
        for g in groups:
            if g.tlbrs:
                det_boxes.append(g.tlbrs[0])
            else:
                det_boxes.append(np.array([0,0,0,0]))
        if not track_boxes or not det_boxes:
            return np.ones((len(tracks), len(groups)))
        track_boxes = np.stack(track_boxes)
        det_boxes = np.stack(det_boxes)
        iou_mat = ious(track_boxes, det_boxes)
        dist_mat = 1 - iou_mat  # IoU similarity → distance
        # decay theo tuổi bbox (view lâu không update sẽ ít đáng tin → distance ↑)
        for i, trk in enumerate(tracks):
            age_bbox_min = np.nanmin(trk.age_bbox) if np.any(np.isfinite(trk.age_bbox)) else self.cfg.max_age_bbox
            decay = 1.0 + age_bbox_min / (self.cfg.max_age_bbox + 1e-6)
            dist_mat[i] *= decay
        return dist_mat

    def _feet_distance_matrix(self, tracks: List[MTrack], groups: List[MTrack]) -> np.ndarray:
        """Khoảng cách bàn chân trên mặt phẳng ground."""
        if self.cameras is None:
            return np.ones((len(tracks), len(groups)))
        from .geometry_utils import feet_distance
        feet_mat = np.ones((len(tracks), len(groups))) * 1e3
        for i, trk in enumerate(tracks):
            # estimate track feet world: dùng centroid (x,y) nếu có
            if trk.centroid is None:
                continue
            feet_world = np.array(trk.centroid)
            for j, grp in enumerate(groups):
                pose_list = grp.curr_pose if grp.curr_pose else []
                if not pose_list:
                    continue
                pose = pose_list[0]
                cam_id = pose.get('cam_id',0)
                if cam_id>=len(self.cameras):
                    continue
                kp2d = pose['keypoints']
                feet_idxs = [15,16]
                if np.all(kp2d[feet_idxs,2]>0.5):
                    feet_xy = kp2d[feet_idxs,:2].mean(axis=0)
                    d = feet_distance(feet_world, feet_xy, self.cameras[cam_id])
                    feet_mat[i,j]=d
        if feet_mat.size:
            maxv,minv=np.max(feet_mat),np.min(feet_mat)
            if maxv>minv:
                feet_mat=(feet_mat-minv)/(maxv-minv)
        # decay theo tuổi bbox
        for i, trk in enumerate(tracks):
            age_bbox_min = np.nanmin(trk.age_bbox) if np.any(np.isfinite(trk.age_bbox)) else self.cfg.max_age_bbox
            decay = 1.0 + age_bbox_min / (self.cfg.max_age_bbox + 1e-6)
            feet_mat[i] *= decay
        return feet_mat

    def _epi_distance_matrix(self, tracks: List[MTrack], groups: List[MTrack]) -> np.ndarray:
        """Epipolar consistency distance using main joints rays."""
        if self.cameras is None or len(self.cameras) < 2:
            return np.ones((len(tracks), len(groups)))
        from .geometry_utils import compute_ray_from_pixel, epipolar_score
        num_t = len(tracks)
        num_g = len(groups)
        dmat = np.ones((num_t, num_g))
        for i, trk in enumerate(tracks):
            for j, grp in enumerate(groups):
                pose_list = grp.curr_pose if grp.curr_pose else []
                if not pose_list:
                    continue
                pose_j = pose_list[0]
                cam_j = pose_j.get('cam_id',0)
                if cam_j>=len(self.cameras):
                    continue
                # tìm một view khác của track (tuổi bbox 0) khác cam_j
                cam_i = None
                for v in range(len(self.cameras)):
                    if v!=cam_j and np.any(trk.keypoints_mv[v,:,2]>0):
                        cam_i=v;break
                if cam_i is None:
                    continue
                # dùng khớp vai (5) làm ví dụ
                j_idx=5
                kp_i=trk.keypoints_mv[cam_i][j_idx]
                kp_j=pose_j['keypoints'][j_idx]
                if kp_i[2]<0.5 or kp_j[2]<0.5:
                    continue
                pos_i=self.cameras[cam_i].pos
                origin_i, dir_i=compute_ray_from_pixel(kp_i[:2], self.cameras[cam_i])
                pos_j=self.cameras[cam_j].pos
                origin_j, dir_j=compute_ray_from_pixel(kp_j[:2], self.cameras[cam_j])
                score=epipolar_score(pos_i, dir_i, pos_j, dir_j)
                dmat[i,j]=1-score  # distance
        if dmat.size:
            maxv,minv=np.max(dmat),np.min(dmat)
            if maxv>minv:
                dmat=(dmat-minv)/(maxv-minv)
        # decay theo tuổi 2D (các khớp trong views còn mới thì đáng tin hơn)
        for i, trk in enumerate(tracks):
            age2d_min = np.nanmin(trk.age_2D) if np.any(np.isfinite(trk.age_2D)) else self.cfg.max_age_2d
            decay = 1.0 + age2d_min / (self.cfg.max_age_2d + 1e-6)
            dmat[i] *= decay
        return dmat

    # -- Tiện ích chuẩn hoá ma trận (min-max) dùng chung ----------------
    @staticmethod
    def _norm_matrix(mat: np.ndarray) -> np.ndarray:
        """Chuẩn hoá ma trận về [0,1] với bảo vệ chia 0."""
        if mat.size == 0:
            return mat
        maxv = np.max(mat)
        minv = np.min(mat)
        if maxv > minv:
            return (mat - minv) / (maxv - minv + 1e-6)
        return np.zeros_like(mat)

    # --- Override update --------------------------------------------------
    def update(self, trackers, groups, scene=None):
        """Refactored update gồm 5 bước gọi các helper method tương ứng."""
        from trackers.multicam_tracker.cluster_track import sub_mtracks

        self.frame_id += 1

        # --- Dự đoán bbox & phát hiện occlusion ---
        self._detect_occlusions()
        self._resolve_occlusions()

        # -------------------------------- 1. Parse groups
        new_groups = self._build_new_groups(groups, scene)

        # -------------------------------- 2. Split current tracks
        tracked_mtracks, unconfirmed = self._split_tracks()

        # -------------------------------- 3. Quick match theo global_id
        activated, tracked_mtracks, new_groups = self._quick_match_gid(tracked_mtracks, new_groups, self.frame_id)

        # -------------------------------- 4. Assoc với tracked
        matches, u_trk, u_new = self._associate(tracked_mtracks, new_groups, full_cost=True, thresh=self.cfg.thresh_match)
        refind, lost = [], []
        for it_src, it_dst in matches:
            src = tracked_mtracks[it_src]
            dst = new_groups[it_dst]
            if src.state == TrackState.Tracked:
                src.update(dst, self.frame_id)
                activated.append(src)
            else:
                src.re_activate(dst, self.frame_id, new_id=False)
                refind.append(src)
        for it in u_trk:
            trk = tracked_mtracks[it]
            if trk.state not in (TrackState.Lost, TrackState.Removed):
                trk.mark_lost()
                lost.append(trk)

        # danh sách còn lại sau bước trên
        new_groups = [new_groups[i] for i in u_new]

        # -------------------------------- 5. Assoc lost & unconfirmed + init
        # Lost ↔ new (emb only)
        matches, u_lost, u_new = self._associate(self.lost_mtracks, new_groups, full_cost=False, thresh=0.35)
        for ilost, inew in matches:
            losttrk = self.lost_mtracks[ilost]
            newtrk = new_groups[inew]
            losttrk.re_activate(newtrk, self.frame_id, new_id=False)
            refind.append(losttrk)
        new_groups = [new_groups[i] for i in u_new]

        # Unconfirmed ↔ new (full cost)
        matches, u_unconf, u_new = self._associate(unconfirmed, new_groups, full_cost=True, thresh=self.cfg.thresh_match)
        removed = []
        for iu, inew in matches:
            unconfirmed[iu].update(new_groups[inew], self.frame_id)
            activated.append(unconfirmed[iu])
        for iu in u_unconf:
            trk = unconfirmed[iu]
            trk.mark_removed()
            removed.append(trk)

        # Initialize mới
        for inew in u_new:
            trk = new_groups[inew]
            trk.activate(self.frame_id)
            activated.append(trk)

        # Remove quá hạn
        for trk in self.lost_mtracks:
            if self.frame_id - trk.end_frame > self.max_time_lost:
                trk.mark_removed()
                removed.append(trk)

        # Merge lists
        outputs = self._merge_lists(activated, refind, lost, removed)

        # ---- Ageing update for all tracks ----
        for trk in (self.tracked_mtracks + self.lost_mtracks):
            if isinstance(trk, HybridTrack):
                trk.increment_age()

        # ---- Periodic triangulation optimisation ----
        if self.cameras is not None and (self.frame_id % self.cfg.triang_interval == 0):
            for trk in self.tracked_mtracks:
                if isinstance(trk, HybridTrack):
                    # valid views count (gần đây, chưa quá hạn)
                    if np.sum(trk.age_bbox < self.cfg.max_age_bbox) >= 2:
                        trk.update_geometry()

        return outputs

    # -----------------------------------------------------------------
    def _detect_occlusions(self, iou_thresh: float = 0.5, reid_thresh: float = 0.5):
        """Cập nhật oc_state cho các track dựa trên IoU bbox_kalman & độ tương đồng Re-ID."""
        from trackers.multicam_tracker.matching import ious

        tracks = [t for t in self.tracked_mtracks if isinstance(t, HybridTrack)]
        n = len(tracks)
        if n < 2:
            return
        # reset oc_state trước khi tính
        for t in tracks:
            t.oc_state = [False for _ in range(t.num_cam)]

        for i in range(n):
            t1 = tracks[i]
            for j in range(i + 1, n):
                t2 = tracks[j]
                # tính reid similarity đơn giản
                sim = self._reid_similarity(t1, t2)
                if sim >= reid_thresh:
                    continue  # giống cùng người, bỏ qua occlusion
                # xét từng camera
                for v in range(min(t1.num_cam, t2.num_cam)):
                    b1 = t1.bbox_kalman[v].project() if v < len(t1.bbox_kalman) else None
                    b2 = t2.bbox_kalman[v].project() if v < len(t2.bbox_kalman) else None
                    if b1 is None or b2 is None:
                        continue
                    iou = ious(np.array([b1[:4]]), np.array([b2[:4]]))[0, 0]
                    if iou > iou_thresh:
                        t1.oc_state[v] = True
                        t2.oc_state[v] = True

    # -----------------------------------------------------------------
    def _resolve_occlusions(self, iou_thresh: float = 0.3, reid_thresh: float = 0.5):
        """Thử giải quyết occlusion: khi IoU nhỏ hơn ngưỡng, quyết định switch_view nếu cần."""
        from trackers.multicam_tracker.matching import ious

        tracks = [t for t in self.tracked_mtracks if isinstance(t, HybridTrack)]
        if len(tracks) < 2 or self.cameras is None:
            return

        num_cam = len(self.cameras)
        for v in range(num_cam):
            # các track đang occluded ở camera v
            occ_tracks = [t for t in tracks if v < t.num_cam and t.oc_state[v]]
            for t in occ_tracks:
                b1 = t.bbox_kalman[v].project()
                if b1 is None:
                    continue
                cleared = True  # giả định hết occlusion
                best_sim = 0
                best_track = None
                for other in tracks:
                    if other.track_id == t.track_id or v >= other.num_cam:
                        continue
                    b2 = other.bbox_kalman[v].project()
                    if b2 is None:
                        continue
                    iou = ious(np.array([b1[:4]]), np.array([b2[:4]]))[0, 0]
                    if iou > iou_thresh:
                        cleared = False
                        break
                    sim = self._reid_similarity(t, other)
                    if sim > best_sim:
                        best_sim = sim
                        best_track = other
                if not cleared:
                    continue
                # Nếu đã tách và sim với track khác cao hơn self-sim => switch view
                if best_track is not None and best_sim > reid_thresh:
                    t.switch_view(best_track, v)
                # Xoá cờ oc_state vì đã tách
                t.oc_state[v] = False
                if best_track is not None:
                    best_track.oc_state[v] = False

    # -----------------------------------------------------------------
    @staticmethod
    def _reid_similarity(t1: HybridTrack, t2: HybridTrack) -> float:
        """Tính độ giống Re-ID tối đa giữa hai bank."""
        if t1.feat_count == 0 or t2.feat_count == 0:
            return 0.0
        bank1 = t1.feat_bank[: min(t1.feat_count, t1.bank_size)]
        bank2 = t2.feat_bank[: min(t2.feat_count, t2.bank_size)]
        sim = np.max(bank1 @ bank2.T)
        return float(sim)