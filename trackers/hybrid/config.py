from dataclasses import dataclass

@dataclass
class HybridConfig:
    """Cấu hình trọng số & ngưỡng cho HybridMCTracker."""
    w_reid: float = 0.5
    w_euc: float = 0.15
    w_geo: float = 0.1
    w_iou: float = 0.1
    w_p2l: float = 0.05
    w_kp2d: float = 0.05
    w_feet: float = 0.05
    w_epi: float = 0.05

    thresh_match: float = 0.7

    # --- Ageing parameters ---
    max_age_2d: int = 3        # số frame sau đó khớp 2D cũ bị bỏ qua
    max_age_bbox: int = 15     # số frame sau đó view/bbox bị bỏ qua

    # Tần suất triangulate (chỉ thực hiện khi frame_id % triang_interval == 0)
    triang_interval: int = 3

    @property
    def weight_sum(self):
        return self.w_reid + self.w_euc + self.w_geo + self.w_iou + self.w_p2l + self.w_kp2d + self.w_feet + self.w_epi 