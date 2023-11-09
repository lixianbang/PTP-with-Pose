# Copyright (c) OpenMMLab. All rights reserved.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
from argparse import ArgumentParser

import cv2

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

from mmdet.apis import inference_detector, init_detector
import numpy as np
import json
import trans18
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def main(num_3):
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    path_in = './resources/JAAD/video_0%d.mp4' % num_3
    parser.add_argument('--det_config', default='./mmdetection_cfg/faster_rcnn_r50_fpn_coco.py', help='Config file for detection')
    parser.add_argument('--det_checkpoint', default='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', help='Checkpoint file for detection')
    parser.add_argument('--pose_config', default='../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py', help='Config file for pose')
    parser.add_argument('--pose_checkpoint', default='https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth', help='Checkpoint file for pose')
    parser.add_argument('--video-path', default=path_in, type=str, help='Video path')
    path_out = "JAAD_JSON/out2FPL_JAAD_0%d.json" % num_3
    print(num_3)

    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='./resources/JJAD',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.7,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.5, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='Using One_Euro_Filter for smoothing')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    # assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)
    for id_ in range(10):
        cap = cv2.VideoCapture(args.video_path)
        fps = None

        assert cap.isOpened(), f'Faild to load video file {args.video_path}'

        if args.out_video_root == '':
            save_out_video = False
        else:
            os.makedirs(args.out_video_root, exist_ok=True)
            save_out_video = True

        if save_out_video:
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                os.path.join(args.out_video_root,
                             f'vis_{os.path.basename(args.video_path)}'), fourcc,
                fps, size)

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        next_id = 0
        pose_results = []
        ##################################################################################################################
        # zhenshu = -1
        #################################################################################################################
        out2FPL = {}
        traj = []
        pose = []


        while (cap.isOpened()):
            #################################################################################################################
            # zhenshu += 1  # 需要输出的 帧数
            # zhenshu_str = str(zhenshu)

            # pose_18 = [np.zeros((18, 2))]
            #################################################################################################################

            pose_results_last = pose_results

            flag, img = cap.read()
            if not flag:
                break
            # test a single image, the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, img)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

            # test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                img,
                person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            # get track id for each person instance
            pose_results, next_id = get_track_id(
                pose_results,
                pose_results_last,
                next_id,
                use_oks=args.use_oks_tracking,
                tracking_thr=args.tracking_thr,
                use_one_euro=args.euro,
                fps=fps)

            try:
                traj_, pose18 = trans18.trans17to18(pose_results, id_)
            # traj_ = [(pose_results[0]['bbox'][0]+pose_results[0]['bbox'][2])/2, (pose_results[0]['bbox'][1]+pose_results[0]['bbox'][3])/2]
            # #print(traj_)
            # # print(pose_results)
            # pose_ = pose_results[0]['keypoints'][:, 0:2]
            # # print(pose_)
            # # print(pose_[0][0])
            # # print(pose_[0][1])
            # pose_18[0][0][0] = pose_[0][0]
            # pose_18[0][0][1] = pose_[0][1]
            # # print(pose_18)
            # pose_18[0][2][0], pose_18[0][2][1] = pose_[6][0], pose_[6][1]
            # # print(pose_18)
            # pose_18[0][3][0], pose_18[0][3][1] = pose_[8][0], pose_[8][1]
            # pose_18[0][4][0], pose_18[0][4][1] = pose_[10][0], pose_[10][1]
            # pose_18[0][5][0], pose_18[0][5][1] = pose_[5][0], pose_[5][1]
            # pose_18[0][6][0], pose_18[0][6][1] = pose_[7][0], pose_[7][1]
            # pose_18[0][7][0], pose_18[0][7][1] = pose_[9][0], pose_[9][1]
            # pose_18[0][8][0], pose_18[0][8][1] = pose_[12][0], pose_[12][1]
            # pose_18[0][9][0], pose_18[0][9][1] = pose_[14][0], pose_[14][1]
            # pose_18[0][10][0], pose_18[0][10][1] = pose_[16][0], pose_[16][1]
            # pose_18[0][11][0], pose_18[0][11][1] = pose_[11][0], pose_[11][1]
            # pose_18[0][12][0], pose_18[0][12][1] = pose_[13][0], pose_[13][1]
            # pose_18[0][13][0], pose_18[0][13][1] = pose_[15][0], pose_[15][1]
            # pose_18[0][14][0], pose_18[0][14][1] = pose_[2][0], pose_[2][1]
            # pose_18[0][15][0], pose_18[0][15][1] = pose_[1][0], pose_[1][1]
            # pose_18[0][16][0], pose_18[0][16][1] = pose_[4][0], pose_[4][1]
            # pose_18[0][17][0], pose_18[0][17][1] = pose_[3][0], pose_[3][1]
            #
            # pose_18[0][1][0], pose_18[0][1][1] = (pose_18[0][5][0]+pose_18[0][6][0])/2, (pose_18[0][5][1]+pose_18[0][6][1])/2
            # # print(type(pose_18))
            # # print(pose_18)
                traj.append(traj_)
            # pose18 = pose_18[0]
                pose.append(pose18)
            except:
                continue
            # if zhenshu == 1:
            #     print(pose)



            # show the results

            # vis_img = vis_pose_tracking_result(
            #     pose_model,
            #     img,
            #     pose_results,
            #     radius=args.radius,
            #     thickness=args.thickness,
            #     dataset=dataset,
            #     dataset_info=dataset_info,
            #     kpt_score_thr=args.kpt_thr,
            #     show=False)

            # if args.show:
            #     cv2.imshow('Image', vis_img)
            #
            # if save_out_video:
            #     videoWriter.write(vis_img)
            #
            # if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cap.release()

        # if save_out_video:
        #     videoWriter.release()
        # if args.show:
        #     cv2.destroyAllWindows()
        out_m = {
            'traj_sm': traj,
            'pose_sm': pose,
            'start': id_ + 1
        }
        # out2FPL['0'] = out_m
        out2FPL[id_] = out_m
        with open(path_out, 'a') as output_json:
            json.dump(out2FPL, output_json, cls=NpEncoder)


if __name__ == '__main__':
    for num_3 in range(100, 347):
        main(num_3)

