"""
Author Jianzhe Lin
May.2, 2020
"""
from mpl_toolkits import mplot3d
import argparse
import multiprocessing
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import time
from re_id import load_pv_data, match_timestamp, judge_loc, judge_list, get_key, draw_loc
from re_id import load_extrinsics, load_rig2world_transforms, get_points_in_cam_space, match
from re_id import extract_tar_file, load_lut, DEPTH_SCALING_FACTOR, project_on_depth, extract_timestamp, cam2world
if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')   
    parser = argparse.ArgumentParser(description='Save pcloud.')
    parser.add_argument("--recording_path", 
                        type=str,
                        default='C:/Users/peterlin/2022-04-15-142023',
                        help="Path to recording folder")
    args = parser.parse_args()
    sensor_name = "Depth Long Throw"
    save_in_cam_space, discard_no_rgb, depth_path_suffix, disable_project_pinhole=False, False, '', False

#######read Paths#########
    print("Creating files")
    folder = Path(args.recording_path)
    calib = r'{}_lut.bin'.format(sensor_name)
    extrinsics = r'{}_extrinsics.txt'.format(sensor_name)
    rig2world = r'{}_rig2world.txt'.format(sensor_name)
    calib_path = folder / calib
    rig2campath = folder / extrinsics
    rig2world_path = folder / rig2world if not save_in_cam_space else ''
    pv_info_path = sorted(folder.glob(r'*pv.txt'))
    has_pv = len(list(pv_info_path)) > 0
    if has_pv:
        (pv_timestamps, focal_lengths, pv2world_transforms, ox,
         oy, _, _) = load_pv_data(list(pv_info_path)[0])
        principal_point = np.array([ox, oy])
    # lookup table to extract xyz from depth
    lut = load_lut(calib_path)
    # from camera to rig space transformation (fixed)
    rig2cam = load_extrinsics(rig2campath)
    # from rig to world transformations (one per frame)
    rig2world_transforms = load_rig2world_transforms(
        rig2world_path) if rig2world_path != '' and Path(rig2world_path).exists() else None
    depth_path = Path(folder / sensor_name)
    depth_path.mkdir(exist_ok=True)
    pinhole_folder = None
    if has_pv:
        # Create folders for pinhole projection
        pinhole_folder = folder / 'pinhole_projection'
        pinhole_folder.mkdir(exist_ok=True)
        pinhole_folder_rgb = pinhole_folder / 'rgb'
        pinhole_folder_rgb.mkdir(exist_ok=True)
        pinhole_folder_depth = pinhole_folder / 'depth'
        pinhole_folder_depth.mkdir(exist_ok=True)
    # Extract tar only when calling the script directly
    if __name__ == '__main__':
        extract_tar_file(str(folder / '{}.tar'.format(sensor_name)), str(depth_path))
    # Depth path suffix used for now only if we load masked AHAT
    depth_paths = sorted(depth_path.glob('*[0-9]{}.pgm'.format(depth_path_suffix)))
    assert len(list(depth_paths)) > 0 
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()

###########get the pv location on world coordinate#########
    # load depth img
    threeD_list = []
    indx = 0
    location_dict = {}
    wh_dict = {}
    patch_dict = {}
    xyz_dict = []
    
    num = [1] * 80
#    flag_cnt = 0  
    volume = []
    max_w, min_w, max_h, min_h, max_d, min_d = 0, 0, 0, 0, 0, 0
    
    for path in depth_paths:
        indx += 1
        suffix = '_cam' if save_in_cam_space else ''
        output_path = str(path)[:-4] + f'{suffix}.ply'
        print(".", end="", flush=True)
    # extract the timestamp for this frame
        timestamp = extract_timestamp(path.name.replace(depth_path_suffix, ''))        
        img = cv2.imread(str(path), -1)
        height, width = img.shape
        assert len(lut) == width * height
    # Get xyz points in camera space
        points = get_points_in_cam_space(img, lut)
        if rig2world_transforms and (timestamp in rig2world_transforms):
            # if we have the transform from rig to world for this frame,
            # then put the point clouds in world space
            rig2world = rig2world_transforms[timestamp]
            # print('Transform found for timestamp %s' % timestamp)
            xyz, cam2world_transform = cam2world(points, rig2cam, rig2world)
            # if flag_cnt == 0: 
            #     cnt = 0
            xyz_temp = []
            width = [item[0] for item in xyz]
            height = [item[1] for item in xyz]
            depth = [item[2] for item in xyz]
            temp_max_w, temp_min_w, temp_max_h, temp_min_h, temp_max_d, temp_min_d = max(width), min(width), max(height), min(height), max(depth), min(depth)
            # for num_xyz in range(0, len(xyz)):
            #     arr = xyz[num_xyz]
            #     input_arr = np.sum(arr)
            #     if  input_arr in xyz_dict:
            #         continue
            #     xyz_dict.append(input_arr)
            #     cnt += 1
            id_list = []
            name_list = []
            if location_dict:
                for k, v in location_dict.items():
                    id_list.append(get_key(xyz, v))
                    name_list.append(k)      
                      
            rgb = None
            if has_pv:
                # if we have pv, get vertex colors
                # get the pv frame which is closest in time
                target_id = match_timestamp(timestamp, pv_timestamps)
                pv_ts = pv_timestamps[target_id]
                rgb_path = str(folder / 'PV' / f'{pv_ts}.png')
                assert Path(rgb_path).exists()
                pv_img = cv2.imread(rgb_path)
                h, w, c = pv_img.shape
                src3 = np.zeros((h, w, c), np.uint8)#use to record  
                src3.fill(255)      
                fig = plt.figure()
                ax = plt.axes(projection='3d')
#                print(cnt)
#first Step Detection###########
                if temp_max_d - max_d > 0 or temp_max_h - max_h > 0 or temp_max_w - max_w > 0 or min_d - temp_min_d > 0 or min_h - temp_min_h > 0 or min_w - temp_min_w > 0:    
                    flag_cnt = 0             
                else:
                    flag_cnt = 1
                max_d, max_h, max_w = max(temp_max_d, max_d), max(temp_max_h, max_h), max(temp_max_w, max_w)
                min_d, min_h, min_w = min(temp_min_d, min_d), min(temp_min_h, min_h), min(temp_min_w, min_w)                
                points, pv2world_transform, focal_length = xyz, pv2world_transforms[target_id], focal_lengths[target_id]

#Second step: Project from depth to pv via world space, and in return get the 3D location on world space
                homog_points = np.hstack((points, np.ones(len(points)).reshape((-1, 1))))
                world2pv_transform = np.linalg.inv(pv2world_transform)
                points_pv = (world2pv_transform @ homog_points.T).T[:, :3]
                intrinsic_matrix = np.array([[focal_length[0], 0, principal_point[0]], [
                    0, focal_length[1], principal_point[1]], [0, 0, 1]])
                rvec = np.zeros(3)
                tvec = np.zeros(3)
                xy, _ = cv2.projectPoints(points_pv, rvec, tvec, intrinsic_matrix, None)
                xy = np.squeeze(xy)
                xy[:, 0] = w - xy[:, 0]
    ########get xy for each 3D points on world space#####
                xy_0 = np.around(xy).astype(int)
                rgb = np.zeros_like(points)
    ###due to the coordinate mismatch between depth and rgb, need to remove outside ids###
                width_check = np.logical_and(0 <= xy_0[:, 0], xy_0[:, 0] < w)
                height_check = np.logical_and(0 <= xy_0[:, 1], xy_0[:, 1] < h)
                valid_ids = np.where(np.logical_and(width_check, height_check))[0]
                z = points_pv[valid_ids, 2]
                xy = xy_0[valid_ids, :] 
                res_patch_num = 0  
                id_list_adjusted = []
                for i in range(len(id_list)):
                    num_id = 0
                    if len(id_list[i][np.in1d(id_list[i], valid_ids)]) == 0:
                        id_temp = 1
                    else:
                        id_temp = id_list[i][np.in1d(id_list[i], valid_ids)][0]
                    if id_temp is not None:
                        id_list_adjusted.append(id_temp)
                        loc = xy_0[id_temp]
                        name = name_list[i]
                        res_patch_num += match(pv_img, name, loc, patch_dict[name], wh_dict[name])
                    else:
                        flag_cnt = 0
                        break
#                     for j in range(len(id_list[i])):

#                         if id_list[i][j] in valid_ids:
#                             id_list_adjusted.append(id_list[i][j])
#                             loc = xy_0[id_list_adjusted[i]]
#                             name = name_list[i]
# #                            res_patch_num += match(pv_img, name, loc, patch_dict[name], wh_dict[name])    
#                             num_id += 1
#                             break
#                     if num_id == 0:
#                         flag_cnt = 0
#                         break
#                print(flag_cnt) 
 #               print(id_list)     
#                print(res_patch_num == len(id_list))                  
 #               if flag_cnt and id_list and res_patch_num == len(id_list):
                flag_seen = 0
#                print(res_patch_num == len(id_list))
                if flag_cnt and id_list and res_patch_num == len(id_list):
 #               if flag_cnt and id_list:


                    print('re_id')
                    for i in range(len(id_list)):
                        loc = xy_0[id_list_adjusted[i]]
                        name = name_list[i]
                        threeD_loc = xyz[id_list_adjusted[i]]
                        src = pv_img  
                        for k, v in location_dict.items():
                            text = "{}!".format(str(name))
                            cv2.putText(src, text, (loc[0]-20, loc[1]-20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (199,21,133), 2)

                else:
                    print('detect')
                    results = model(pv_img.copy())   
                    for result in results.xywhn[0]:
                        name = results.names[int(result[5])]
                        loc = [int(result[0]*w), int(result[1]*h)]
                        if loc is not None:
                            threeD_loc = judge_loc(loc, xy, valid_ids, points) 
                            src = pv_img  
                            if threeD_loc[0] != 0 and threeD_loc[1] != 0 and threeD_loc[2] != 0:
                                if name not in location_dict:
                                    location_dict[name] = threeD_loc
                                    wh_dict[name] = [int(result[2]*w), int(result[3]*h)]
                                    patch_dict[name] = src[int((result[1]-result[3]/2)*h):int((result[1]+result[3]/2)*h), int((result[0]-result[2]/2)*w):int((result[0]+result[2]/2)*w)]
                    
                                else:
                                    text_all = []
                                    for k, v in location_dict.items():
                                        threeD_list_bool = judge_list(v, threeD_loc)
                                        if threeD_list_bool:
                                            text = "{}!".format(str(name))
                                            if text not in text_all:
                                                cv2.putText(src, text, (loc[0]-20, loc[1]-20), cv2.FONT_HERSHEY_COMPLEX, 1, (199,21,133), 2)
                                                text_all.append(text)
                                            if result==None:
                                                pt_lefttop = [int(loc[0]-wh_dict[name][0]/2), int(loc[1]-wh_dict[name][1]/2)]
                                                pt_rightbottom = [int(loc[0]+wh_dict[name][0]/2), int(loc[1]+wh_dict[name][1]/2)]
                                            else:
                                                pt_lefttop = [int((result[0]-result[2]/2)*w), int((result[1]-result[3]/2)*h)]
                                                pt_rightbottom = [int((result[0]+result[2]/2)*w), int((result[1]+result[3]/2)*h)]
                                            point_color = (0, 255, 0)
                                            cv2.rectangle(src, tuple(pt_lefttop), tuple(pt_rightbottom), point_color, 2, 4)    
                                            flag_seen += 1
                                            break
                                    if not threeD_list_bool:
                                        num[int(result[5])] += 1 
                                        name = name + '_' + str(num[int(result[5])])                                                             
                                        location_dict[name] = threeD_loc 
                                        wh_dict[name] = [int(result[2]*w), int(result[3]*h)]
                                        patch_dict[name] = src[int((result[0]-result[2]/2)*w):int((result[0]+result[2]/2)*w), int((result[1]-result[3]/2)*h):int((result[1]+result[3]/2)*h)]
                            
                            ax.set_xlim((-4, 1))
                            ax.set_ylim((-4, 1))
                            ax.set_zlim((-4, 1))    
                flag = 0     
                for name, loc in location_dict.items():    
                    flag += 1
                    ax.scatter(round(loc[0],3), round(loc[1],3), round(loc[2],3), marker='^')   
                    ax.text(round(loc[0],3), round(loc[1],3), round(loc[2],3), name, fontsize=9)                        
                    text = str(name) + " : [" + str(round(loc[0],3)) + ',' + str(round(loc[1],3)) + ',' + str(round(loc[2],3)) + ']'                              
                    x = 20 + 300*(flag//10)
                    y = 20 + 40*(flag % 10)
                    cv2.putText(src3, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (100, 200, 200), 1)
                
                file_name = 'C:/Users/peterlin/ptg-alphatesting/rgb_3D.jpg'
                plt.savefig(file_name,dpi = 96)
                src_temp = cv2.imread(file_name) 
                src4 = cv2.resize(src_temp, (760, 428))
                results.files = [str(indx) + '.jpg']
 #               cv2.imwrite('C:/Users/peterlin/ptg-alphatesting/rgb_results_2022_05_03/' + results.files[0], results.imgs[0])
                results.save('C:/Users/peterlin/ptg-alphatesting/rgb_results_2022_04_20/')   
                src2 = cv2.imread('C:/Users/peterlin/ptg-alphatesting/rgb_results_2022_04_20/' + str(indx) + '.jpg') 
                # cv2.imshow('img22',src2)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()                          
                img = np.hstack([src, src2])
                img2 = np.hstack([src3, src4])
                img_out = np.vstack([img, img2])
                print(indx)
                cv2.imwrite('C:/Users/peterlin/ptg-alphatesting/rgb_results_combined_2022_04_20/' + str(indx) + '.jpg', img_out)



