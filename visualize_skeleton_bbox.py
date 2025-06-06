import argparse
import copy
import math
from datetime import datetime

import numpy as np
import cv2
import os

import pandas as pd
import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb

# BGR
COLOURS = {(0, 1) : (255, 0, 255), 
           (0, 2) : (255, 0, 255), 
           (1, 3) : (255, 0, 255), 
           (2, 4) : (255, 0, 255),
           (5, 7) : (0, 127, 255), # left arm
           (7, 9) : (0, 255, 255), # left arm
           (6, 8) : (127, 255, 0), # right arm
           (8, 10) : (0, 255, 0), # right arm
           (11, 13) : (127, 225, 0), # left leg
           (13, 15) : (255, 225, 0), # left let
           (12, 14) : (255, 127, 0), # right leg
           (14, 16) : (255, 0, 0), # right leg
           (0, 5) : (192, 127, 192), 
           (0, 6) : (127, 127, 192),
           (5, 6) : (0, 0, 255), # chest
           (5, 11) : (0, 0, 255), # left side
           (6, 12) : (0, 0, 255), # right side
           (11, 12) : (0, 0, 255) # pelvis
           }  # Dark Green



COLOURS_POINTS = {
            0 : (255, 0, 255), 
            1 : (255, 0, 255), 
            2 : (255, 0, 255), 
            3 : (255, 0, 255), 
            4 : (255, 0, 255), 
            5 : (0, 64, 255), 
            7 : (0, 191, 255),
            9 : (0, 255, 255),
            6 : (191, 255, 0),
            8 : (64, 255, 0),
            10 : (0, 255, 0),
            11 : (127,255,127),
            13 : (192, 225, 0),
            15 : (255, 255, 0),
            12 : (255, 127, 64),
            14 : (255, 64, 0),
            16 : (255, 0, 0),
            17: (255, 0, 0)
           }  # Dark Green



parser = argparse.ArgumentParser(description='Visualize the predicted skeletons with corresponding bounding boxes.')

parser.add_argument('--elsec_db', type=bool, default=False, help='do you want to use elsec data base? y/n')
parser.add_argument('--test_data_dir', type=str, default='', help='directory of the test dir for loading masks')

parser.add_argument('--frames', type=str, help='Directory containing video frames.')
parser.add_argument('--gt_trajectories', type=str,
                              help='Directory containing the ground-truth trajectories of people in the video.')
parser.add_argument('--draw_gt_skeleton', type=bool,default=True, help='Whether to draw the ground-truth skeletons or not.')
parser.add_argument('--draw_gt_bbox',type=bool,default=False, help='Whether to draw the bounding box of the ground-truth skeletons or not.')
parser.add_argument('--trajectories', type=str,help='Directory containing the reconstructed/predicted trajectories of people in '
                                   'the video.')
parser.add_argument('--draw_pred_skeleton',type=bool,default=True,
                              help='Whether to draw the reconstructed/predicted skeleton or not.')
parser.add_argument('--draw_pred_bbox',type=bool,default=False,
                              help='Whether to draw the bounding box of the reconstructed/predicted trajectories '
                                   'or not.')
parser.add_argument('--person_id', type=int, help='Draw only a specific person in the video.')
parser.add_argument('--draw_local_skeleton', action='store_true',
                              help='If specified, draw local skeletons on a white background. It must be used '
                                   'in conjunction with --person_id, since it is only possible to visualise '
                                   'one pair (ground-truth, reconstructed/predicted) of local skeletons.')
parser.add_argument('--write_dir', default='./visualise', type=str,
                              help='Directory to write rendered frames. If the specified directory does not '
                                   'exist, it will be created.')
parser.add_argument('--generate_gif',type=bool,default=False,
                              help='Render gif from the prediction frames.')
parser.add_argument('--scale',type=int,default=1,
                              help='scale of frames.')


def prepare_keypoints(keypoints):
    keypoints = keypoints * 8
    min_x = min([k[0] for k in keypoints if k[0]!=0])
    min_y = min([k[1] for k in keypoints if k[1]!=0])
    
    max_x = max([k[0] for k in keypoints if k[0]!=0])
    max_y = max([k[1] for k in keypoints if k[1]!=0])
    
    n = 800 / (max_x-min_x)
    
    keypoints = keypoints * n
    
    new_ks = []
    for x,y in keypoints:
        if 0 in (x, y):
            new_ks.append((x,y))
        else: 
            new_ks.append((x-(min_x*n)+40.,y-(min_y*n)+40.))
    keypoints = new_ks
    frame = np.full((math.floor((max_y-min_y)*n+80.),math.floor((max_x-min_x)*n+80.)),fill_value=255, dtype=np.single)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return keypoints,frame


def draw_skeleton(frame, keypoints, colour, dotted=False, scale=4, scale_vis=False):
    connections = [(5, 6), (5, 11), (6, 12), (11, 12),
                   (0, 1), (0, 2), (1, 3), (2, 4),
                   (5, 7), (7, 9), (6, 8), (8, 10),
                   (11, 13), (13, 15), (12, 14), (14, 16),
                   (0, 5), (0, 6)]
    
    keypoints = keypoints * scale
    
    if scale_vis:
        line_thickness=4*10
        circle_thickness = -1
        radius =3*12
    else:
        line_thickness=1
        circle_thickness = 1
        radius =1


    for i,(keypoint_id1, keypoint_id2) in enumerate(connections):
        x1, y1 = keypoints[keypoint_id1]
        x2, y2 = keypoints[keypoint_id2]
        if 0 in (x1, y1, x2, y2):
            continue
        pt1 = int(round(x1)), int(round(y1))
        pt2 = int(round(x2)), int(round(y2))
        if dotted:
            draw_line(frame, pt1=pt1, pt2=pt2, color=COLOURS[connections[i]], thickness=line_thickness, gap=5)
        else:
            cv2.line(frame, pt1=pt1, pt2=pt2, color=COLOURS[connections[i]], thickness=line_thickness)
    
    for i, (x, y) in enumerate(keypoints):
        if 0 in (x, y):
            continue
        center = int(round(x)), int(round(y))
        cv2.circle(frame, center=center, radius=radius, color=COLOURS_POINTS[i], thickness=circle_thickness)

    return None

def draw_rect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    draw_poly(img, pts, color, thickness, style)

def draw_line(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1

def draw_poly(img, pts, color, thickness=1, style='dotted'):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        draw_line(img, s, e, color, thickness, style)

def compute_simple_bounding_box(skeleton):
    x = skeleton[::2]
    x = np.where(x == 0.0, np.nan, x)
    left, right = int(round(np.nanmin(x))), int(round(np.nanmax(x)))
    y = skeleton[1::2]
    y = np.where(y == 0.0, np.nan, y)
    top, bottom = int(round(np.nanmin(y))), int(round(np.nanmax(y)))

    return left, right, top, bottom

def render_trajectories_skeletons(args):
    """
    Renders visualizations of trajectories and skeletons based on specified configurations.

    This function generates images using the specified parameters, rendering trajectories and skeletons
    from provided ground truth or predicted data. The output is saved in the directory specified by the user.
    The function ensures to validate provided arguments and raises errors if required arguments are
    missing or improperly specified.

    Arguments:
        args (argparse.Namespace): Encapsulates all the required input parameters.
            write_dir: str
                Directory to save the rendered images.
            test_data_dir: str
                Directory containing test dataset.
            elsec_db: str
                Path to elsec data for rendering.
            frames: str
                Path to the folder containing frame images.
            gt_trajectories: str, optional
                Path to the ground truth trajectories file.
            draw_gt_skeleton: bool
                Whether to draw ground truth skeletons or not.
            draw_gt_bbox: bool
                Whether to draw ground truth bounding boxes or not.
            trajectories: str, optional
                Path to the predicted trajectories file.
            draw_pred_skeleton: bool
                Whether to draw predicted skeletons or not.
            draw_pred_bbox: bool
                Whether to draw predicted bounding boxes or not.
            person_id: int, optional
                ID of the specific person to visualize.
            draw_local_skeleton: bool
                Whether to draw local skeleton for a specific individual.
            scale: float, optional
                Scaling factor to adjust rendering.

    Raises:
        ValueError: If neither ground truth nor predicted trajectories are provided.
        ValueError: If none of the drawing options are specified.
        ValueError: If `draw_local_skeleton` is specified without setting a `person_id`.

    Returns:
        None
    """
    try:
        os.makedirs(args.write_dir)
    except OSError:
        print(f' \n directory for the images already exists. IMAGES WILL BE REWRITTEN!!! \n')
        pass


    test_data_dir = args.test_data_dir          # Directory containing test dataset
    elsec_data = args.elsec_db                  # boolean flag if to use elsec database
    frames_path = args.frames                    # Path to the folder with frame images
    gt_trajectories_path = args.gt_trajectories  # Path to ground truth trajectories file
    draw_gt_skeleton = args.draw_gt_skeleton     # Flag to control drawing ground truth skeletons
    draw_gt_bounding_box = args.draw_gt_bbox     # Flag to control drawing ground truth bounding boxes
    trajectories_path = args.trajectories        # Path to predicted trajectories file
    draw_trajectories_skeleton = args.draw_pred_skeleton    # Flag to control drawing predicted skeletons
    draw_trajectories_bounding_box = args.draw_pred_bbox   # Flag to control drawing predicted bounding boxes
    specific_person_id = args.person_id          # ID of specific person to visualize
    draw_local_skeleton = args.draw_local_skeleton  # Flag to control drawing local skeleton for specific person

    if gt_trajectories_path is None and trajectories_path is None:
        raise ValueError('At least one of --ground_truth_trajectories or --trajectories must be specified.')

    if not any([draw_gt_skeleton, draw_gt_bounding_box, draw_trajectories_skeleton, draw_trajectories_bounding_box]):
        raise ValueError('At least one of --draw_ground_truth_trajectories_skeleton, '
                         '--draw_ground_truth_trajectories_bounding_box, --draw_trajectories_skeleton or '
                         '--draw_trajectories_bounding_box must be specified.')

    if draw_local_skeleton and specific_person_id is None:
        raise ValueError('If --draw_local_skeleton is specified, a --person_id must be chosen as well.')
    elif draw_local_skeleton:
        draw_gt_skeleton = draw_trajectories_skeleton = True
        draw_gt_bounding_box = draw_trajectories_bounding_box = False


    _render_trajectories_skeletons(args.write_dir, frames_path, gt_trajectories_path, trajectories_path, specific_person_id, scale=args.scale,
                                   elsec_data=elsec_data, test_data_dir=test_data_dir)

    print('Visualisation successfully rendered to %s' % args.write_dir)

    return None

def fill(frames_path, frame_name,scale,ts):
    if ts is None or not ts:
        ts = [None,None]
        ts[0] = cv2.imread(os.path.join(frames_path, frame_name))
        h,w,c = ts[0].shape
        ts[0] = cv2.resize(ts[0], (w*scale,h*scale), interpolation = cv2.INTER_AREA)
        ts[1] = np.full_like(ts[0], fill_value=255)
    return ts

def fill_multi(frames_path, frame_name,scale,ts,person_ids):
    if ts is None:
        ts = {p : [] for p in person_ids}
        for p in ts.keys():
            ts1 = cv2.imread(os.path.join(frames_path, frame_name))
            h,w,c = ts1.shape
            ts1 = cv2.resize(ts1, (w*scale,h*scale), interpolation = cv2.INTER_AREA)
            ts2 = np.full_like(ts1, fill_value=255)
            ts[p] = (ts1,ts2)
    
    for p in person_ids:
        if p not in ts.keys():
            ts1 = cv2.imread(os.path.join(frames_path, frame_name))
            h,w,c = ts1.shape
            ts1 = cv2.resize(ts1, (w*scale,h*scale), interpolation = cv2.INTER_AREA)
            ts2 = np.full_like(ts1, fill_value=255)
            ts[p] = (ts1,ts2)
    return ts


def load_anomaly_masks_elsec(anomaly_masks_path):
    file_names = os.listdir(anomaly_masks_path)
    masks = {}
    for file_name in file_names:
        if file_name.split('.')[1] == 'csv':
            pandas_df = pd.read_csv(os.path.join(anomaly_masks_path, file_name))
            masks = dict(zip(pandas_df.iloc[:, 0], pandas_df.iloc[:,1]))
    return masks

def _render_trajectories_skeletons(write_dir, frames_path, gt_trajectories_path, trajectories_path,
                                   specific_person_id=None, scale=4, elsec_data=False, test_data_dir = '/home/pp/Downloads/data/HR-ShanghaiTech/testing'):
    camera_id = os.path.basename(os.path.normpath(frames_path)).split('_')[0]

    vid_id = trajectories_path.split('/')[-1]
    w_dirs = [os.path.join(write_dir,'frames',s,camera_id,vid_id) for s in ['ind_pred','ind_gt','all_pred','all_gt']]
    for d in w_dirs:
        if not os.path.isdir(d):
            os.makedirs(d)
    
    wo_dirs = [os.path.join(write_dir,'trajectories',s,camera_id,vid_id) for s in ['ind_pred','ind_gt','all_pred','all_gt']]
    for d in wo_dirs:
        if not os.path.isdir(d):
            os.makedirs(d)


    if elsec_data==True:
        frames_names = sorted(os.listdir(os.path.join(frames_path, vid_id, 'Pos','Images')))
        max_frame_id = int((datetime.now() - datetime(1975, 1, 1)).total_seconds() * 30)
    else:
        frames_names = sorted(os.listdir(frames_path))  # 000.jpg, 001.jpg, ...
        max_frame_id = len(frames_names)
    rendered_pred_frames_all = {}
    rendered_pred_frames_ind = {}
    
    rendered_gt_frames_all = {}
    rendered_gt_frames_ind = {}
    person_ids = []

    def load_anomaly_masks(anomaly_masks_path):
        file_names = os.listdir(anomaly_masks_path)
        masks = {}
        for file_name in file_names:
            full_id = file_name.split('.')[0]
            file_path = os.path.join(anomaly_masks_path, file_name)
            masks[full_id] = np.load(file_path)

        return masks

    if elsec_data == True:
        masks = load_anomaly_masks_elsec(os.path.join(test_data_dir, 'frame_level_masks', camera_id))
    else:
        masks = load_anomaly_masks(os.path.join(test_data_dir, 'frame_level_masks', camera_id))
    print(f'camera_id = {camera_id} scene id = {vid_id} ')
    if trajectories_path is not None:
        trajectories_files_names = sorted(os.listdir(trajectories_path))[0:200]  # 001.csv, 002.csv, ...
        for indx, trajectory_file_name in enumerate(trajectories_files_names):
            person_id = int(trajectory_file_name.split('.')[0])
            if specific_person_id is not None and specific_person_id != person_id or person_id<0:
                continue
            print(f'\rDrawing skeleton for person_id:{person_id} which is  {indx} out of {len(trajectories_files_names)}', end="")
            if person_id not in person_ids:
                person_ids.append(person_id)

            colour = COLOURS_POINTS[person_id % len(COLOURS)]

            if elsec_data == True:
                trajectory_df = pd.read_csv(os.path.join(trajectories_path, trajectory_file_name))
                image_file_name = trajectory_df.iloc[:, -1].tolist()
                trajectory = trajectory_df.iloc[:, :-1].to_numpy()
            else:
                trajectory = np.loadtxt(os.path.join(trajectories_path, trajectory_file_name), delimiter=',', ndmin=2)
            trajectory_frames = trajectory[:, 0].astype(np.int64)
            trajectory_coordinates = trajectory[:, 1:]

            for index, (frame_id, skeleton_coordinates) in enumerate(zip(trajectory_frames, trajectory_coordinates)):
                if frame_id >= max_frame_id:
                    break
                if elsec_data==True:
                    frames_dir_name = sorted(os.listdir(frames_path))  # 000.jpg, 001.jpg, ...
                    main_frame_name = [d for d in frames_dir_name if 'jpg' in d][0]
                    frame_ind = cv2.imread(os.path.join(frames_path, main_frame_name))
                    object_image = cv2.imread(os.path.join(frames_path, vid_id,'Pos', 'Images', image_file_name[index])+'.jpg')
                    y1, x1 = int(min(skeleton_coordinates[0::2][0:4])), int(min(skeleton_coordinates[1::2][0:4]))
                    y2, x2 = int(max(skeleton_coordinates[0::2][0:4])), int(max(skeleton_coordinates[1::2][0:4]))
                    object_image = cv2.resize(object_image, (int(y2 - y1), int(x2 - x1)), interpolation=cv2.INTER_AREA)
                    frame_ind[x1:x2, y1:y2] = object_image
                else:
                    frame_ind = cv2.imread(os.path.join(frames_path, frames_names[frame_id]))
                h,w,c = frame_ind.shape
                frame_ind = cv2.resize(frame_ind, (w*scale,h*scale), interpolation = cv2.INTER_AREA)
                blank_frame_ind = np.full_like(frame_ind, fill_value=255)
                
                coords, blank_frame_ind = prepare_keypoints(skeleton_coordinates.reshape(-1, 2))
                blank_frame_ind = blank_frame_ind.astype(np.uint8)
                el = rendered_pred_frames_all.get(frame_id)
                
                if el is not None:
                    frame = el[0]
                    blank_frame = el[1]
                else:
                    frame = frame_ind.copy()
                    blank_frame = np.full_like(frame_ind, fill_value=255).astype(np.uint8)
                
                draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2), colour=colour, dotted=False, scale=scale)
                draw_skeleton(frame_ind, keypoints=skeleton_coordinates.reshape(-1, 2), colour=colour, dotted=False, scale=scale)
                
                #height, width = blank_frame_ind.shape[:2]
                #left, right, top, bottom = compute_simple_bounding_box(skeleton_coordinates)
                #bb_center = np.array([(left + right) / 2, (top + bottom) / 2], dtype=np.float32)
                #target_center = np.array([3 * width / 4, height / 2], dtype=np.float32)
                #displacement_vector = target_center - bb_center
                
                draw_skeleton(blank_frame_ind, keypoints=coords,colour=colour, dotted=False, scale=scale, scale_vis=True)
                draw_skeleton(blank_frame, keypoints=skeleton_coordinates.reshape(-1, 2),colour=colour, dotted=False, scale=scale)

                track_id = str(int(trajectory_file_name.split('.csv')[0]))
                cv2.putText(frame, str(track_id), (int(skeleton_coordinates[2]), int(skeleton_coordinates[3]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if elsec_data == True:
                    does_the_setup_exists = True
                else:
                    mask_disc = camera_id+'_'+trajectories_path.split('/')[-1]
                    does_the_setup_exists = masks.get(mask_disc) is not None
                if does_the_setup_exists == True:
                    # if int(frame_id) < len(masks[mask_disc]):
                    if elsec_data == True:
                        is_it_anomaly_frame = masks[int(frame_id)]
                    else:
                        is_it_anomaly_frame = masks[mask_disc][int(frame_id)]
                    # pdb.set_trace()  # Pause here
                    if is_it_anomaly_frame == 1:
                        print(f'track_id {track_id} is an anomaly')
                        coordinate_y = int(np.min(int(skeleton_coordinates[3]-40),0))
                        coordinate_x = int(skeleton_coordinates[2])

                        cv2.putText(frame, 'Anomaly', (coordinate_x, coordinate_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(blank_frame, 'Anomaly', (coordinate_x, coordinate_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                rendered_pred_frames_all[frame_id] = (frame,blank_frame)
                if frame_id not in rendered_pred_frames_ind.keys():
                    rendered_pred_frames_ind[frame_id] = {}
                if person_id not in rendered_pred_frames_ind[frame_id].keys():
                    rendered_pred_frames_ind[frame_id][person_id] = []
                rendered_pred_frames_ind[frame_id][person_id] = (frame_ind,blank_frame_ind)

    if gt_trajectories_path is not None:
        gt_trajectories_files_names = sorted(os.listdir(gt_trajectories_path))[0:200]
        for gt_trajectory_file_name in gt_trajectories_files_names:
            person_id = int(gt_trajectory_file_name.split('.')[0])
            if specific_person_id is not None and specific_person_id != person_id or person_id<0:
                continue

            
            colour = COLOURS_POINTS[person_id % len(COLOURS)]

            if elsec_data == True:
                trajectory_df = pd.read_csv(os.path.join(trajectories_path, gt_trajectory_file_name))
                image_file_name = trajectory_df.iloc[:, -1].tolist()
                gt_trajectory = trajectory_df.iloc[:, :-1].to_numpy()
            else:
                gt_trajectory = np.loadtxt(os.path.join(gt_trajectories_path, gt_trajectory_file_name),
                                       delimiter=',', ndmin=2)
            gt_trajectory_frames = gt_trajectory[:, 0].astype(np.int64)
            gt_trajectory_coordinates = gt_trajectory[:, 1:]

            for indx, (frame_id, skeleton_coordinates) in enumerate(zip(gt_trajectory_frames, gt_trajectory_coordinates)):
                
                skeleton_is_null = np.any(skeleton_coordinates)
                if not skeleton_is_null:
                    continue

                if elsec_data == True:
                    frames_dir_name = sorted(os.listdir(frames_path))  # 000.jpg, 001.jpg, ...
                    main_frame_name = [d for d in frames_dir_name if 'jpg' in d][0]
                    frame_ind = cv2.imread(os.path.join(frames_path, main_frame_name))
                    object_image = cv2.imread(os.path.join(frames_path, vid_id,'Pos', 'Images', image_file_name[indx])+'.jpg')
                    y1, x1 = int(min(skeleton_coordinates[0::2][0:4])), int(min(skeleton_coordinates[1::2][0:4]))
                    y2, x2 = int(max(skeleton_coordinates[0::2][0:4])), int(max(skeleton_coordinates[1::2][0:4]))
                    object_image = cv2.resize(object_image, (int(y2 - y1), int(x2 - x1)), interpolation=cv2.INTER_AREA)
                    frame_ind[x1:x2, y1:y2] = object_image
                else:
                    frame_ind = cv2.imread(os.path.join(frames_path, frames_names[frame_id]))

                h,w,c = frame_ind.shape
                frame_ind = cv2.resize(frame_ind, (w*scale,h*scale), interpolation = cv2.INTER_AREA)
                blank_frame_ind = np.full_like(frame_ind, fill_value=255)
                
                coords, blank_frame_ind = prepare_keypoints(skeleton_coordinates.reshape(-1, 2))
                blank_frame_ind = blank_frame_ind.astype(np.uint8)

                el = rendered_gt_frames_all.get(frame_id)
                
                if el is not None:
                    frame = el[0]
                    blank_frame = el[1]
                else:
                    frame = frame_ind.copy()
                    blank_frame = np.full_like(frame_ind, fill_value=255)
                blank_frame = blank_frame.astype(np.uint8)
                draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2), colour=colour, dotted=False, scale=scale)
                draw_skeleton(frame_ind, keypoints=skeleton_coordinates.reshape(-1, 2), colour=colour, dotted=False, scale=scale)
                
                #height, width = blank_frame_ind.shape[:2]
                #left, right, top, bottom = compute_simple_bounding_box(skeleton_coordinates)
                #bb_center = np.ar  ray([(left + right) / 2, (top + bottom) / 2], dtype=np.float32)
                #target_center = np.array([3 * width / 4, height / 2], dtype=np.float32)
                #displacement_vector = target_center - bb_center
                track_id = str(int(gt_trajectory_file_name.split('.csv')[0]))
                coordinate_y = int(np.min(int(skeleton_coordinates[3] - 10), 0))
                coordinate_x = int(skeleton_coordinates[2])
                cv2.putText(frame, str(track_id), (coordinate_x, coordinate_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                draw_skeleton(blank_frame_ind, keypoints=coords,colour=colour, dotted=False, scale=scale, scale_vis=True)
                draw_skeleton(blank_frame, keypoints=skeleton_coordinates.reshape(-1, 2),colour=colour, dotted=False, scale=scale)

                # plt.imshow(frame)
                # plt.pause(0.1)
                rendered_gt_frames_all[frame_id] = (frame,blank_frame)
                if frame_id not in rendered_gt_frames_ind.keys():
                    rendered_gt_frames_ind[frame_id] = {}
                if person_id not in rendered_gt_frames_ind[frame_id].keys():
                    rendered_gt_frames_ind[frame_id][person_id] = []
                rendered_gt_frames_ind[frame_id][person_id] = (frame_ind,blank_frame_ind)
                print(
                    f'\rDrawing skeleton for person_id:{person_id} which is  {indx} out of {len(gt_trajectory_frames)}',
                    end="")


    if elsec_data == True:
        frames_path = os.path.join(frames_path, vid_id,'Pos', 'Images')
        frames_names = set(list(rendered_pred_frames_ind.keys()))
        # frames_dir_name = sorted(os.listdir(frames_path))  # 000.jpg, 001.jpg, ...
        # main_frame_name = [d for d in frames_dir_name if 'jpg' in d][0]
        # frame_ind = cv2.imread(os.path.join(frames_path, main_frame_name))
    for frame_id, frame_name in tqdm.tqdm(enumerate(frames_names),total=len(frames_names)):
        if elsec_data == True:
            frame_id = copy.deepcopy(frame_name)
            frames_dir_name = sorted(os.listdir(frames_path))  # 000.jpg, 001.jpg, ...
            frame_name = [d for d in frames_dir_name if 'jpg' in d][0]

        pred_frame_ind = rendered_pred_frames_ind.get(frame_id)
        pred_frame_all = rendered_pred_frames_all.get(frame_id)
        gt_frame_ind = rendered_gt_frames_ind.get(frame_id)
        gt_frame_all = rendered_gt_frames_all.get(frame_id)

        if elsec_data == True:
            frame_name = str(frame_id) + '.jpg'
        else:
            pred_frame_all = fill(frames_path, frame_name,scale,pred_frame_all)
            pred_frame_ind = fill_multi(frames_path, frame_name,scale,pred_frame_ind,person_ids)

            gt_frame_all = fill(frames_path, frame_name, scale, gt_frame_all)
            gt_frame_ind = fill_multi(frames_path, frame_name, scale, gt_frame_ind, person_ids)


        #cv2.imwrite(os.path.join(w_dirs[0],frame_name), pred_frame_ind[0])
        #cv2.imwrite(os.path.join(w_dirs[1],frame_name), gt_frame_ind[0])
        cv2.imwrite(os.path.join(w_dirs[2],frame_name), pred_frame_all[0])
        cv2.imwrite(os.path.join(w_dirs[3],frame_name), gt_frame_all[0])

        #cv2.imwrite(os.path.join(wo_dirs[0],frame_name), pred_frame_ind[1])
        #cv2.imwrite(os.path.join(wo_dirs[1],frame_name), gt_frame_ind[1])
        cv2.imwrite(os.path.join(wo_dirs[2],frame_name), pred_frame_all[1])
        cv2.imwrite(os.path.join(wo_dirs[3],frame_name), gt_frame_all[1])

        for person_id in pred_frame_ind.keys():
            pred_frame_ind_pid = pred_frame_ind.get(person_id)
            if not os.path.isdir(os.path.join(w_dirs[0],str(person_id))):
                os.makedirs(os.path.join(w_dirs[0],str(person_id)))
            if not os.path.isdir(os.path.join(wo_dirs[0],str(person_id))):
                os.makedirs(os.path.join(wo_dirs[0],str(person_id)))
            cv2.imwrite(os.path.join(w_dirs[0],str(person_id),frame_name), pred_frame_ind_pid[0])
            cv2.imwrite(os.path.join(wo_dirs[0],str(person_id),frame_name), pred_frame_ind_pid[1])


        for person_id in gt_frame_ind.keys():
            gt_frame_ind_pid = gt_frame_ind.get(person_id)
            gt_frame_ind_pid = fill(frames_path, frame_name,scale,gt_frame_ind_pid)
            if not os.path.isdir(os.path.join(w_dirs[1],str(person_id))):
                os.makedirs(os.path.join(w_dirs[1],str(person_id)))
            if not os.path.isdir(os.path.join(wo_dirs[1],str(person_id))):
                os.makedirs(os.path.join(wo_dirs[1],str(person_id)))
            cv2.imwrite(os.path.join(w_dirs[1],str(person_id),frame_name), gt_frame_ind_pid[0])
            cv2.imwrite(os.path.join(wo_dirs[1],str(person_id),frame_name), gt_frame_ind_pid[1])
        




def main():
    args = parser.parse_args()
    args.elsec_db = True
    if args.elsec_db:
        args.gt_trajectories = '/home/pp/Desktop/datasets/trajrec_data/elsec_data/testing/trajectories/2023_2_10'
        args.trajectories = '/home/pp/Desktop/datasets/trajrec_data/elsec_data/testing/trajectFalseories/2023_2_10'
        args.frames = '/home/pp/Desktop/datasets/elsec_dataset/frame/01'
        args.test_data_dir = '/home/pp/Desktop/datasets/trajrec_data/elsec_data/testing'
        if not os.path.exists(args.write_dir):
            os.makedirs(args.write_dir)
        if os.path.exists(args.trajectories):  # and scene_num=='0025':
            render_trajectories_skeletons(args)
    else:
        test_dir = '/home/pp/Desktop/datasets/trajrec_data/shanghaitech/testing/frames'
        scene_names = os.listdir(test_dir)
        trajectories_path_base = os.path.join(os.path.dirname(args.trajectories.rstrip('/')), '')
        frames_path_base = os.path.join(os.path.dirname(args.frames.rstrip('/')), '')
        gt_trajectories_path_base = os.path.join(os.path.dirname(os.path.dirname(args.gt_trajectories.rstrip('/'))), '')
        for scene_name in scene_names:
            scene_num  =  scene_name.split('_')[1]
            camera_number = scene_name.split('_')[0]
            # args.write_dir = 'visualization' + scene_name
            args.trajectories = os.path.join(trajectories_path_base, scene_num)
            args.frames = os.path.join(frames_path_base, scene_name)

            args.gt_trajectories = os.path.join(gt_trajectories_path_base, camera_number, scene_num)

            if not os.path.exists(args.write_dir):
                os.makedirs(args.write_dir)
            if os.path.exists(args.trajectories):# and scene_num=='0025':
                render_trajectories_skeletons(args)


if __name__ == '__main__':
    main()
