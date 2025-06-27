from typing import List, Optional
import numpy as np

# Sử dụng lại các thành phần từ multicam_tracker
from trackers.multicam_tracker.cluster_track import MCTracker as _BaseMCTracker, MTrack
from trackers.multicam_tracker import matching
from trackers.multicam_tracker.basetrack import TrackState

from .geometry_utils import triangulate, p2l_distance


class HybridTrack(MTrack):
    """MTrack mở rộng – bổ sung dữ liệu hình học 3D."""

    def __init__(self, *args, cameras: Optional[List[object]] = None, **kwargs):
        self.cameras = cameras  # reference tới list Camera của hệ thống
        self.num_cam = len(cameras) if cameras is not None else 0
        self.num_kp = 17  # default COCO
        self.keypoints_mv = np.zeros((self.num_cam, self.num_kp, 3))
        self.keypoints_3d = np.zeros((self.num_kp, 4))
        super().__init__(*args, **kwargs)
        # Geometry attributes thiết lập sau khi call super (nếu chưa có)
        self.num_cam = len(cameras) if cameras is not None else 0
        self.num_kp = 17  # default COCO
        self.keypoints_mv = np.zeros((self.num_cam, self.num_kp, 3))
        self.keypoints_3d = np.zeros((self.num_kp, 4))
        # NOTE: update_features may have filled keypoints already in super().__init__
        # bank re-id mở rộng (numpy) – kế thừa deque hiện có nhưng có thể chuyển dần

    def update_geometry(self):
        if self.cameras is None or self.num_cam < 2:
            return
        self.keypoints_3d, _ = triangulate(self.keypoints_mv, self.cameras)

    # override nếu cần update features để lưu keypoints_mv
    def update_features(self, features, poses, img_paths, tlbrs, coords):
        super().update_features(features, poses, img_paths, tlbrs, coords)
        # cập nhật keypoints_mv nếu cameras được cung cấp
        if poses is None or self.cameras is None:
            return
        for p in poses:
            if p is None:
                continue
            cam_id = p.get('cam_id', None)
            if cam_id is None:
                continue
            if cam_id < self.num_cam:
                self.keypoints_mv[cam_id] = p['keypoints']
        self.update_geometry()


class HybridMCTracker(_BaseMCTracker):
    """Phiên bản kết hợp: thêm chi phí hình học khi matching."""

    def __init__(self, cameras: Optional[List[object]] = None, w_geo: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.cameras = cameras
        self.w_geo = w_geo

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
                valid = (kp_t[:,2] > 0.5) & (kp_s[:,2] > 0.5)
                if not np.any(valid):
                    continue
                dif = kp_t[valid,:2] - kp_s[valid,:2]
                dist = np.linalg.norm(dif, axis=1).mean()
                dmat[i, j] = dist
        if dmat.size:
            maxv, minv = np.max(dmat), np.min(dmat)
            if maxv > minv:
                dmat = (dmat - minv) / (maxv - minv)
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
        return dmat

    # --- Override update --------------------------------------------------
    def update(self, trackers, groups, scene=None):
        """Bản sao logic update của MCTracker nhưng cộng thêm thành phần geometry-cost."""
        from trackers.multicam_tracker.cluster_track import grouping_rerank, joint_mtracks, sub_mtracks

        self.frame_id += 1
        activated_mtracks = []
        refind_mtracks = []
        lost_mtracks = []
        removed_mtracks = []

        # Giải nạp groups
        if len(groups):
            global_ids = groups[:, 0]
            features = groups[:, 1]
            centroids = groups[:, 2]
            poses = groups[:, 3]
            paths = groups[:, 4]
            tlbrs = groups[:, 5]
            coords = groups[:, 6]
        else:
            global_ids, features, centroids, poses, paths, tlbrs, coords = [], [], [], [], [], [], []

        if len(centroids) > 0:
            new_groups = [HybridTrack(g, c, f, p, ph, t, cd, self.min_hits, scene, cameras=self.cameras)
                          for (g, c, f, p, ph, t, cd) in zip(global_ids, centroids, features, poses, paths, tlbrs, coords)]
        else:
            new_groups = []

        # Bước 1: tách confirmed/unconfirmed
        unconfirmed = []
        tracked_mtracks = []
        for track in self.tracked_mtracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_mtracks.append(track)

        # ---------- Quick matching theo global_id ------------------------
        gid_to_track = {t.global_id: t for t in tracked_mtracks}
        matched_exist = []
        matched_new = []
        for idx, g in enumerate(new_groups):
            if g.global_id in gid_to_track:
                exist = gid_to_track[g.global_id]
                exist.update(g, self.frame_id)
                activated_mtracks.append(exist)
                matched_exist.append(exist)
                matched_new.append(idx)

        # Loại bỏ các new_groups đã matched
        if matched_new:
            new_groups = [g for i, g in enumerate(new_groups) if i not in matched_new]
            # cũng loại khỏi tracked_mtracks lists để không double match
            tracked_mtracks = [t for t in tracked_mtracks if t not in matched_exist]

        # Bước 2: Association với track đang Tracked
        exist_features = [feat for m in tracked_mtracks for feat in list(m.features)]
        lengths_exists = [len(m.features) for m in tracked_mtracks]
        new_features = [feat for g in new_groups for feat in list(g.features)]
        lengths_new = [len(m.features) for m in new_groups]
        exist_centroids = [m.centroid for m in tracked_mtracks]
        new_centroids = [g.centroid for g in new_groups]

        shape = (len(lengths_exists), len(lengths_new))
        if 0 in shape:
            dists = np.empty(shape)
        else:
            rerank_dists = matching.embedding_distance(exist_features, new_features) / 2.0
            emb_dists = grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=False)

            euc_dists = matching.euclidean_distance(exist_centroids, new_centroids) / 1.0

            geo_dists = self._geometry_distance(tracked_mtracks, new_groups)
            p2l_dists = self._p2l_distance_matrix(tracked_mtracks, new_groups)
            kp2d_dists = self._kp2d_distance_matrix(tracked_mtracks, new_groups)
            iou_dists = self._iou_distance_matrix(tracked_mtracks, new_groups)
            feet_dists = self._feet_distance_matrix(tracked_mtracks, new_groups)
            epi_dists = self._epi_distance_matrix(tracked_mtracks, new_groups)

            # normalize
            norm_emb = (emb_dists - np.min(emb_dists)) / (np.max(emb_dists) - np.min(emb_dists) + 1e-6)
            norm_euc = (euc_dists - np.min(euc_dists)) / (np.max(euc_dists) - np.min(euc_dists) + 1e-6)
            if geo_dists.size:
                norm_geo = (geo_dists - np.min(geo_dists)) / (np.max(geo_dists) - np.min(geo_dists) + 1e-6)
            else:
                norm_geo = np.zeros_like(norm_emb)

            norm_p2l = p2l_dists
            norm_kp2d = kp2d_dists

            dists = 0.5*norm_emb + 0.15*norm_euc + 0.1*norm_geo + 0.1*iou_dists + 0.05*norm_p2l + 0.05*norm_kp2d + 0.05*feet_dists + 0.05*epi_dists

        matches, u_exist, u_new = matching.linear_assignment(dists, thresh=0.7)

        for iexist, inew in matches:
            exist = tracked_mtracks[iexist]
            new = new_groups[inew]
            if exist.state == TrackState.Tracked:
                exist.update(new, self.frame_id)
                activated_mtracks.append(exist)
            else:
                exist.re_activate(new, self.frame_id, new_id=False)
                refind_mtracks.append(exist)

        for it in u_exist:
            track = tracked_mtracks[it]
            if not track.state == TrackState.Lost and (not track.state == TrackState.Removed):
                track.mark_lost()
                lost_mtracks.append(track)

        # Step 3: Lost ↔ new_groups (giữ nguyên emb_dists)
        new_groups = [new_groups[i] for i in u_new]

        lost_features = [feat for m in self.lost_mtracks for feat in list(m.features)]
        lengths_lost = [len(m.features) for m in self.lost_mtracks]
        new_features = [feat for g in new_groups for feat in list(g.features)]
        lengths_new = [len(m.features) for m in new_groups]
        shape = (len(lengths_lost), len(lengths_new))
        if 0 in shape:
            emb_dists = np.empty(shape)
        else:
            rerank_dists = matching.embedding_distance(lost_features, new_features) / 2.0
            emb_dists = grouping_rerank(rerank_dists, lengths_lost, lengths_new, shape, normalize=False)
        dists = emb_dists

        matches, u_lost, u_new = matching.linear_assignment(dists, thresh=0.35)

        for ilost, inew in matches:
            lost = self.lost_mtracks[ilost]
            new = new_groups[inew]
            lost.re_activate(new, self.frame_id, new_id=False)
            refind_mtracks.append(lost)

        # Step 4: unconfirmed ↔ new_groups
        new_groups = [new_groups[i] for i in u_new]

        exist_centroids = [m.centroid for m in unconfirmed]
        new_centroids = [g.centroid for g in new_groups]
        exist_features = [feat for m in unconfirmed for feat in list(m.features)]
        lengths_exists = [len(m.features) for m in unconfirmed]
        new_features = [feat for g in new_groups for feat in list(g.features)]
        lengths_new = [len(m.features) for m in new_groups]
        shape = (len(lengths_exists), len(lengths_new))
        if 0 in shape:
            dists = np.empty(shape)
        else:
            rerank_dists = matching.embedding_distance(exist_features, new_features) / 2.0
            emb_dists = grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=False)
            euc_dists = matching.euclidean_distance(exist_centroids, new_centroids) / 1.0
            geo_dists = self._geometry_distance(unconfirmed, new_groups)
            p2l_dists = self._p2l_distance_matrix(unconfirmed, new_groups)
            kp2d_dists = self._kp2d_distance_matrix(unconfirmed, new_groups)
            iou_dists = self._iou_distance_matrix(unconfirmed, new_groups)
            feet_dists = self._feet_distance_matrix(unconfirmed, new_groups)
            epi_dists = self._epi_distance_matrix(unconfirmed, new_groups)

            norm_emb = (emb_dists - np.min(emb_dists)) / (np.max(emb_dists) - np.min(emb_dists) + 1e-6)
            norm_euc = (euc_dists - np.min(euc_dists)) / (np.max(euc_dists) - np.min(euc_dists) + 1e-6)
            if geo_dists.size:
                norm_geo = (geo_dists - np.min(geo_dists)) / (np.max(geo_dists) - np.min(geo_dists) + 1e-6)
            else:
                norm_geo = np.zeros_like(norm_emb)

            dists = 0.5*norm_emb + 0.15*norm_euc + 0.1*norm_geo + 0.1*iou_dists + 0.05*p2l_dists + 0.05*kp2d_dists + 0.05*feet_dists + 0.05*epi_dists

        matches, u_unconfirmed, u_new = matching.linear_assignment(dists, thresh=0.7)

        for iexist, inew in matches:
            unconfirmed[iexist].update(new_groups[inew], self.frame_id)
            activated_mtracks.append(unconfirmed[iexist])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_mtracks.append(track)

        # Step 5: khởi tạo mới
        for inew in u_new:
            track = new_groups[inew]
            track.activate(self.frame_id)
            activated_mtracks.append(track)

        # Step 6: remove quá hạn
        for track in self.lost_mtracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_mtracks.append(track)

        # Merge lists
        self.tracked_mtracks = [t for t in self.tracked_mtracks if t.state == TrackState.Tracked]
        self.tracked_mtracks = joint_mtracks(self.tracked_mtracks, activated_mtracks)
        self.tracked_mtracks = joint_mtracks(self.tracked_mtracks, refind_mtracks)
        self.lost_mtracks = sub_mtracks(self.lost_mtracks, self.tracked_mtracks)
        self.lost_mtracks.extend(lost_mtracks)
        self.lost_mtracks = sub_mtracks(self.lost_mtracks, removed_mtracks)

        return [track for track in self.tracked_mtracks if track.is_activated]