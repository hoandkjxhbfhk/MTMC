python src/viz_multi_camera_v2.py --scene F11_09 \
                                  --show_topology Yes \
                                  --topology_map assets/t11.png \
                                  --homography_matrices config/dataset_VTX2024_homo_info_calc1.json \
                                  --camera_order 111101 111105 111108 111111 111109 111110 111103 \
                                  --mode gt

# python src/viz_multi_camera_v2.py --scene VNU_315 \
#                                   --show_topology Yes \
#                                   --topology_map assets/vnu_315.png \
#                                   --homography_matrices config/vnu_homo.json \
#                                   --camera_order 315101 315102 \
#                                   --mode pred
