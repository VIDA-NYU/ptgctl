"""
Author Jianzhe Lin
May.2, 2020
"""
import numpy as np
import cv2



class Points3D:
    '''Given the Hololens image, depth, and spatial data, convert 2d points to 3d points.
    
    Arguments:
        rgb (np.ndarray): The rgb image. [height, width, channels]
        depth (np.ndarray): The depth image. [height, width]
        lut (np.ndarray): The depth point lookup table. [n_points, xyz coords]
        T_rig2world (np.ndarray): Spatial transformation from rig space to world space. [4, 4]
        T_rig2cam (np.ndarray): Spatial transformation from rig space to depth cam space. [4, 4]
        T_pv2world (np.ndarray): Spatial transformation from photo space to world space. [4, 4]
        focal_length (list): The x and y focal length. [xy]
        principal_point (list): The x and y principal point. [xy]
    '''
    def __init__(self, 
            rgb, depth, lut, 
            T_rig2world_depth, 
            T_rig2cam_depth, 
            T_pv2world, 
            focal_length, 
            principal_point):
        self.pv2world = T_pv2world
        
        # transform depth into world and image coordinates
        H, W = rgb.shape[:2]
        xyz_depth_cam = transform_magnitude2cam_space(depth, lut)
        xyz_depth_world = transform_cam2world_space(xyz_depth_cam, T_rig2cam_depth, T_rig2world_depth)
        xyz_depth_pv = transform_depth2image_space(xyz_depth_world, T_pv2world, focal_length, principal_point)

        # not sure why
        xyz_depth_pv[:, 0] = W - xyz_depth_pv[:, 0]
        # clip image boundaries
        valid_points = (
            ((0 <= xyz_depth_pv[:, 0]) & (xyz_depth_pv[:, 0] < W)) & 
            ((0 <= xyz_depth_pv[:, 1]) & (xyz_depth_pv[:, 1] < H)))
        self.xyz_depth_pv = xyz_depth_pv[valid_points]
        self.xyz_depth_world = xyz_depth_world[valid_points]

    def transform_points(self, xy):
        '''Transform points from 2d image space to 3d world space.
        
        Arguments:
            xy (np.ndarray): The box coordinates in image space. Shape should be [n_points, xy]

        Returns:
            np.ndarray: The points in 3d space. (n_points, 3)
            np.ndarray: distance in xy space between the photo and depth point. (n_points,)
                you can use this to filter out points that are too far away.
        '''
        xy_world, xy_pv, dists = transform_points2world_via_closest_depth(xy, self.xyz_depth_pv, self.xyz_depth_world)
        return xy_world, dists

    def transform_box(self, xyxy):
        '''Transform points from 2d image space to 3d world space.
        
        Arguments:
            xy (np.ndarray): The box coordinates in image space. Shape should be [n_points, xy]

        Returns:
            np.ndarray: The points in 3d space. (n_points, 3)
            np.ndarray: distance in xy space between the photo and depth point. (n_points,)
                you can use this to filter out points that are too far away.
        '''
        xy1 = xyxy[:, :2]
        xy2 = xyxy[:, 2:4]
        xyc = (xy2 + xy1) / 2
        xyzc_world, xyz_pv, dists = transform_points2world_via_closest_depth(
            xyc, self.xyz_depth_pv, self.xyz_depth_world)
        xyz_tl_world = transform_image2world(xy1, xyz_pv[:, 2], self.pv2world)
        xyz_br_world = transform_image2world(xy2, xyz_pv[:, 2], self.pv2world)
        xyz_tr_world = transform_image2world(np.concatenate([xy2[:, 0][:,None], xy1[:, 1][:,None]], axis=1), xyz_pv[:, 2], self.pv2world)
        xyz_bl_world = transform_image2world(np.concatenate([xy1[:, 0][:,None], xy2[:, 1][:,None]], axis=1), xyz_pv[:, 2], self.pv2world)
        return xyz_tl_world, xyz_br_world, xyz_tr_world, xyz_bl_world, xyzc_world, dists



def transform_magnitude2cam_space(img, lut, min_scale=1e-6):
    '''Get depth image in camera space using the direction lookup table.

    Arguments:
        img (np.ndarray): The depth image.
        lut (np.ndarray): The unit vector lookup table for each point.
        min_scale (float): Filter out points smaller than this.
    
    Returns:
        A np.ndarray of shape [-1, 3] in cam space.
    '''
    height, width = img.shape
    assert len(lut) == width * height
    points = img.reshape((-1, 1)) * lut
    return points[np.sum(points, axis=1) > min_scale] / 1000.

def transform_cam2world_space(points, rig2cam, rig2world):
    '''Transform points from camera space to world space via the intermediate rig space.

    Arguments:
        points (np.ndarray): A flattened array of points in camera space.
        rig2cam (np.ndarray): The transformation matrix between rig and cam space. [4, 4]
        rig2world (np.ndarray): The transformation matrix between rig and world space. [4, 4]
    
    Returns:
        points (xyz) in world space - a np.ndarray of shape [-1, 3]
    '''
    homog_points = np.hstack((points, np.ones((len(points), 1))))
    T_cam2world = rig2world @ np.linalg.inv(rig2cam)
    world_points = T_cam2world @ homog_points.T
    return world_points.T[:, :3]

def transform_depth2image_space(points, pv2world, focal_length, principal_point):
    #Second step: Project from depth to pv via world space, and in return get the 3D location on world space
    homog_points = np.hstack((points, np.ones((len(points), 1))))
    points_pv = (np.linalg.inv(pv2world) @ homog_points.T).T[:, :3]
    xy = cv2.projectPoints(
        points_pv, 
        rvec=np.zeros(3), tvec=np.zeros(3), 
        cameraMatrix=np.array([
            [focal_length[0], 0, principal_point[0]], 
            [0, focal_length[1], principal_point[1]], 
            [0, 0, 1]
        ]),
        distCoeffs=None)[0][:, 0]
    xyz_pv = np.concatenate([xy, points_pv[:, 2:3]], axis=1)
    return xyz_pv

def transform_points2world_via_closest_depth(pts, xyz_pv, xyz_depth_world):
    '''get the boxes world coordinates'''
    # find the closest depth point for each box within some threshold
    # if not dists.shape[-1]:
    #     return xyz_pv[:,:2], np.zeros((dists.shape[0],)).fill(np.nan)
    dists = np.linalg.norm(pts[:,:2][:,None] - xyz_pv[:,:2][None], axis=-1)
    closest = np.argmin(dists, axis=-1) if dists.shape[-1] else np.zeros((dists.shape[0], ))
    dists = np.take_along_axis(dists, closest[:,None], axis=-1)[:, 0]
    return xyz_depth_world[closest], xyz_pv[closest], dists

def transform_image2world(xy_pv, depth, pv2world):
    xy_pv = np.concatenate([xy_pv, depth[:, None], np.ones((len(xy_pv), 1))], axis=1)
    xy_world = xy_pv @ pv2world
    return xy_world[:, :3]

def find_close(loc, xyz_pv, xyz_world, buffer=7, top=4):
    # find the closest points within some threshold
    dist = np.linalg.norm(loc - xyz_pv[:,:2])
    topk = np.argpartition(-dist, top)[:top]
    matches = xyz_world[topk[dist[topk] < buffer]]
    return matches[0] if len(matches) else None



# sample


import asyncio


async def _main(**kw):
    import torch
    import ptgctl
    import ptgctl.holoframe

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')

    kw.setdefault('last_entry_id', 0)

    api = ptgctl.CLI(local=False)
    streams = ['main', 'depthlt']

    data = ptgctl.holoframe.load_all(api.data('depthltCal'))
    lut, T_rig2cam = ptgctl.holoframe.unpack(data, [
        'depthltCal.lut', 
        'depthltCal.rig2cam', 
    ])

    async with api.data_pull_connect('+'.join(streams), **kw) as ws:
        while True:
            data = await ws.recv_data()
            data = ptgctl.holoframe.load_all(data)

            (
                rgb, depth,
                T_rig2world, T_pv2world, 
                focalX, focalY, principalX, principalY,
            ) = ptgctl.holoframe.unpack(
                data, [
                'main.image', 
                'depthlt.image', 
                'depthlt.rig2world', 
                'main.cam2world', 
                'main.focalX', 
                'main.focalY', 
                'main.principalX',
                'main.principalY',
            ])

            pts3d = Points3D(
                rgb, depth, lut, 
                T_rig2world, T_rig2cam, T_pv2world, 
                [focalX, focalY], 
                [principalX, principalY])

            results = model(rgb)
            boxes = results.xywh[0].numpy()

            boxes_xyz_world, dist = pts3d.transform(boxes[:, :2])
            valid = dist < 7  # make sure the points aren't too far
            boxes = boxes[valid]
            boxes_xyz_world = boxes_xyz_world[valid]

            

def main(**kw):
    return asyncio.run(_main(**kw))
