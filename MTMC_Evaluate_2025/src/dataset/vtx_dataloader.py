import os
import glob
from collections import defaultdict
import json

class VTXDataset2024():
    def __init__(self, data_root, img_size):
        self.data_root = os.path.abspath(data_root) # VTX_MTMC_2024 folder path
        self.img_size = img_size  # Tuple (w, h)

        self.scenes = sorted([os.path.basename(scene) for scene in glob.glob(
            os.path.join(self.data_root, "*")) if os.path.isdir(scene)])

        self.scene_info = {}
        self.get_scenes_info()

    def get_scenes_info(self):
        """Dataset scene info structure

        scene_info = {
            "F05_01": {
                "cameras": {
                    "camera_1": {
                        "cam_id": cam_id
                        "video": video_path
                        "label": label_path
                    },
                    "camera_2": ...
                }
                "img_size": {
                    "width": video_width
                    "height": video_height
                },
                "no_cams": no_cams,
                "grid_view_video": path
            },
        }

        """
        for scene in self.scenes:
            self.scene_info[scene] = self._get_each_scene_info(scene)
            self._check_scene_valid(self.scene_info[scene])

    def _get_each_scene_info(self, scene_name):
        """Get video and label path in each scene

        Args:
            scene_name ([str]): Name of scene

        Returns:
            [dict]: {
                "cameras": {
                    "camera_1": {
                        "cam_id": cam_id
                        "video": video_path
                        "label": label_path
                    },
                    "camera_2": ...
                }
                "img_size": {
                    "width": video_width
                    "height": video_height
                },
                "no_cams": no_cams,
                "grid_view_video": path
            }
        """

        infos = defaultdict(dict)
        infos["img_size"]["width"] = self.img_size[0]
        infos["img_size"]["height"] = self.img_size[1]

        camera_views = [cam_id for cam_id in sorted(os.listdir(os.path.join(self.data_root, scene_name))) if os.path.isdir(os.path.join(self.data_root, scene_name, cam_id))]

        for cam_no, cam_id in enumerate(camera_views):
            infos["cameras"][f"camera_{cam_no+1:02d}"] = {}
            
            view_video_path = os.path.join(self.data_root, scene_name, cam_id, f"{cam_id}.mp4")
            view_label_path = os.path.join(self.data_root, scene_name, cam_id, f"{cam_id}.txt")
            infos["cameras"][f"camera_{cam_no+1:02d}"]["cam_id"] = cam_id
            infos["cameras"][f"camera_{cam_no+1:02d}"]["video"] = view_video_path
            infos["cameras"][f"camera_{cam_no+1:02d}"]["label"] = view_label_path

        infos["no_cams"] = len(camera_views)
        infos["grid_view_video"] = os.path.join(self.data_root, scene_name, f"grid_view.mp4")

        return dict(infos)
    
    def _check_scene_valid(self, scene):
        """Check every file (video and label) in this scene exists

        Args:
            scene ([dict]): Scene info 
        """

        for camera, val in scene["cameras"].items():
            if (isinstance(val, dict)):
                video = scene["cameras"][camera]["video"]
                label = scene["cameras"][camera]["label"]

                if (not os.path.isfile(video)):
                    raise Exception(
                        f"Video {video} does not exists")
                if (not os.path.isfile(label)):
                    raise Exception(
                        f"Label {label} does not exists")

    def pretty_print(self):
        """Print prettify version of dictionary"""
        print(json.dumps(self.scene_info, indent=4, sort_keys=True))