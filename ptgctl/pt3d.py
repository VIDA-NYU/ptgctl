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
        xy_world, xy_pv, dists = transform_points2world_via_closest_depth(
            xy, self.xyz_depth_pv, self.xyz_depth_world)
        return xy_world, dists

    def _transform_box(self, xyxy, pts=None, ref=np.array([0.5, 0.5])):
        '''

        Arguments:
            xyxy (np.ndarray): The bounding box coordinates in the original image space.
            pts (np.ndarray): The reference points you want to 
            ref (np.ndarray): The point you want to use for depth estimation

        Returns:
            np.ndarray: The points in 3d space. (len(xyxy), 3)
            np.ndarray: distance in xy space between the photo and depth point. (len(xyxy),)
                you can use this to filter out points that are too far away.
            np.ndarray: [len(pts), len(xyxy), 2] Any additional points you want to calculate, relative to the 
                box e.g. to calculate the top center, provide: `pts=np.array([[0.5, 0]])`

        Example:
            assert xyxy.shape == (10, 4)
            ref = np.array([0.5, 0.5])
            pts = np.array([
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ])
            self._transform_box(xyxy, [0.5, 0.5], refs)
        '''
        xy1 = xyxy[:, :2]
        xy2 = xyxy[:, 2:4]
        diff = xy2 - xy1
        xyc = xy1 + diff*np.asarray(ref)  # get depth reference point
        xyzc_world, xyz_pv, dists = transform_points2world_via_closest_depth(
            xyc, self.xyz_depth_pv, self.xyz_depth_world)
        
        if pts is not None:
            # compute extra points on the image plane
            # [1, nbox, 2] + [npt, 1, 2]
            xypts = xy1[None] + diff*np.asarray(pts)[:, None]
            depth = np.broadcast_to(xyz_pv[None,:,2], (len(xypts), len(xyz_pv)))
            xyz_pts_world = transform_image2world(xypts, depth, self.pv2world)
        else:
            xyz_pts_world = np.zeros((len(xyxy), 0, 2))
        return xyzc_world, dists, xyz_pts_world

    def transform_center(self, xyxy):
        '''Transform points from 2d image space to 3d world space.
        
        Arguments:
            xy (np.ndarray): The box coordinates in image space. Shape should be [n_points, xy]

        Returns:
            np.ndarray: The points in 3d space. (n_points, 3)
            np.ndarray: distance in xy space between the photo and depth point. (n_points,)
                you can use this to filter out points that are too far away.
        '''
        xyzc_world, dists, _ = self._transform_box(xyxy)
        return xyzc_world, dists

    def transform_center_top(self, xyxy):
        xyzc_world, dists, xyz_pts_world = self._transform_box(xyxy, np.array([[0.5, 0]]))
        return xyzc_world, xyz_pts_world[0], dists

    def transform_corners(self, xyxy):
        xyzc_world, dists, xyz_pts_world = self._transform_box(xyxy, np.array([
            [0, 0], [1, 0],
            [0, 1], [1, 1],
        ]))
        return xyzc_world, xyz_pts_world, dists

    def transform_box(self, xyxy, return_corners=True):
        '''Transform points from 2d image space to 3d world space.
        
        Arguments:
            xy (np.ndarray): The box coordinates in image space. Shape should be [n_points, xy]

        Returns:
            np.ndarray: The points in 3d space. (n_points, 3)
            np.ndarray: distance in xy space between the photo and depth point. (n_points,)
                you can use this to filter out points that are too far away.
        '''
        if not return_corners:
            return self.transform_center(xyxy)
        xyzc_world, xyz_pts_world, dists = self.transform_corners(xyxy)
        return tuple(xyz_pts_world) + (xyzc_world, dists)



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
    xy_pv = np.concatenate([xy_pv, depth[..., None], np.ones(xy_pv.shape[:-1]+(1,))], axis=-1)
    xy_world = xy_pv @ pv2world
    return xy_world[...,:3]

def find_close(loc, xyz_pv, xyz_world, buffer=7, top=4):
    # find the closest points within some threshold
    dist = np.linalg.norm(loc - xyz_pv[:,:2])
    topk = np.argpartition(-dist, top)[:top]
    matches = xyz_world[topk[dist[topk] < buffer]]
    return matches[0] if len(matches) else None



# sample


import asyncio


async def _main(prefix='', **kw):
    import torch
    import ptgctl
    import ptgctl.holoframe

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')

    api = ptgctl.API(**kw)
    prefix = prefix or ''
    streams = ['main', 'depthlt', 'depthltCal']
    streams = [f'{prefix}{k}' for k in streams]

    data = ptgctl.holoframe.load_all(api.data('depthltCal'))

    async with api.data_pull_connect('+'.join(streams), **kw) as ws:
        while True:
            data.update(ptgctl.holoframe.load_all(await ws.recv_data()))
            try:
                main, depth, depthcal = [data[k] for k in streams]
            except KeyError as e:
                print('key error', e)
                await asyncio.sleep(0.2)
                continue
            pts3d = Points3D(
                main['image'], depth['image'], depthcal['lut'], 
                depth['rig2world'], depthcal['rig2cam'], main['cam2world'], 
                [main['focalX'], main['focalY']], 
                [main['principalX'], main['principalY']])

            results = model(main['image'])
            boxes = results.xywh[0].numpy()

            xyz_top, xyz_center, dist = pts3d.transform_center_top(boxes)
            valid = dist < 7  # make sure the points aren't too far
            boxes = boxes[valid]
            xyz_top = xyz_top[valid]
            xyz_center = xyz_center[valid]
            for b, xc, xt in zip(boxes, xyz_center, xyz_top):
                print(b, xc, xt)
            print()

            

def main(**kw):
    return asyncio.run(_main(**kw))

if __name__ == '__main__':
    import fire
    fire.Fire(main)