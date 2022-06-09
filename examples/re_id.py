"""
Author Jianzhe Lin
May.2, 2020
"""
import asyncio
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import ptgctl
import ptgctl.holoframe
from ptgctl.pt3d import Points3D


class ReId:
    MIN_DEPTH_POINT_DISTANCE = 7

    def __init__(self) -> None:
        with torch.autocast('cpu'):
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')
        self.location_memory = {}
        self.instance_count = defaultdict(lambda: 0)

    def process_frame(self, 
            rgb, depth, lut, 
            T_rig2world, 
            T_rig2cam, 
            T_pv2world, 
            focal_length, 
            principal_point,
        ):
        pts3d = Points3D(
            rgb, depth, lut, 
            T_rig2world, T_rig2cam, T_pv2world, 
            focal_length, principal_point)

        # get boxes
        results = self.model(rgb)
        boxes = results.xyxy[0].numpy()

        # get boxes with xyz coords and filter bad matches
        boxes_xyz_world, dist = pts3d.transform_box(boxes, return_corners=False)
        valid = dist < self.MIN_DEPTH_POINT_DISTANCE
        boxes = boxes[valid]
        boxes_xyz_world = boxes_xyz_world[valid]
        labels = np.asarray(results.names)[boxes[:, 5].astype(int)]

        # compare boxes with previous instances
        track_ids, seen_before = self.update_memory_batch(
            boxes_xyz_world, labels)

        # just add results to yolo detections object
        results.xyz_world = boxes_xyz_world
        results.track_ids = track_ids
        results.seen_before = seen_before
        return results

    def update_memory_batch(self, xyzs, labels):
        ids, seen = tuple(zip(*(
            self.update_memory(xyz, label)
            for xyz, label in zip(xyzs, labels)
        ))) or ((), ())
        return ids, seen

    def update_memory(self, xyz, label):
        # check memory
        for k, xyz_seen in self.location_memory.items():
            if self.memory_comparison(xyz_seen, xyz):
                return k, True

        # unique name for multiple instances
        if label in self.location_memory:
            self.instance_count[label] += 1
            i = self.instance_count[label]
            label = f'{label}_{i}'

        # TODO: add other info
        self.location_memory[label] = xyz
        return label, False

    def memory_comparison(self, seen, candidate):
        '''Compare a new instance to a previous one. Determine if they match.'''
        return np.linalg.norm(candidate - seen) < 0.8


class DrawResults:
    def __init__(self, memory):
        self.location_memory = memory

    def draw_4panel(self, results):
        rgb = results.imgs[0]
        return np.vstack([
            np.hstack([
                self.draw_memory_yolo(results), 
                self.draw_basic_yolo(results),
            ]), 
            np.hstack([
                self.draw_message_board(rgb.shape), 
                self.draw_3d_space(rgb.shape),
            ]),
        ])

    def draw_memory_yolo(self, results):
        img = results.imgs[0]
        for b, seen in zip(results.xywh[0], results.seen_before):
            img = draw_bbox(
                img, *b[:4], 
                color=(255, 0, 0) if seen else (0, 255, 0))
        draw_text_list(img, [
            f'hey I remember {name}'
            for name, seen in zip(results.track_ids, results.seen_before)
            if seen
        ])
        return img

    def draw_basic_yolo(self, results):
        return results.render()[0]

    def draw_message_board(self, shape):
        img = np.ones(shape, np.uint8) * 255
        img = draw_text_list(img, [
            f"{name}: [{', '.join(f'{x:.3f}' for x in loc)}]"
            for name, loc in self.location_memory.items()
        ])
        return img

    def draw_3d_space(self, shape):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # ax.set_xlim((-5, 5))
        # ax.set_ylim((-5, 5))
        # ax.set_zlim((-5, 5))

        for name, loc in self.location_memory.items():    
            ax.scatter(*loc, marker='^')   
            ax.text(*loc, name, fontsize=10)

        fig.canvas.draw()
        src = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        src = src.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        src = cv2.resize(src, shape[:2][::-1])
        plt.close()
        return src



# drawing

def draw_text_list(img, texts):
    for i, txt in enumerate(texts):
        cv2.putText(img, txt, (400, 200+30*i), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 48, 48), 2)
    return img


def draw_bbox(img, xc, yc, w, h, *, color=(0, 255, 0)):
    img = cv2.rectangle(
        img, 
        (int(xc - w/2), int(yc - h/2)), 
        (int(xc + w/2), int(yc + h/2)), 
        color, 2)
    return img




async def _main(**kw):
    kw.setdefault('last_entry_id', 0)

    reid = ReId()
    drawer = DrawResults(reid.location_memory)
    i = 0

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
                T_rig2world, 
                T_pv2world, 
                focalX, focalY, 
                principalX, principalY,
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
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            results = reid.process_frame(
                rgb, depth, lut, 
                T_rig2world, 
                T_rig2cam, 
                T_pv2world, 
                [focalX, focalY], 
                [principalX, principalY]
            )
            out_img = drawer.draw_4panel(results)
            cv2.imshow('main', out_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

def main_stream(**kw):
    return asyncio.run(_main(**kw))


# def main(**kw):
#     import ptgctl
#     import ptgctl.holoframe

#     api = ptgctl.CLI(local=False)
#     streams = ['main', 'depthlt', 'depthltCal']

#     reid = ReId()
#     drawer = DrawResults(reid.location_memory)

#     data = ptgctl.holoframe.load_all(api.data('depthltCal'))
#     lut, T_rig2cam = ptgctl.holoframe.unpack(data, [
#         'depthltCal.lut', 
#         'depthltCal.rig2cam', 
#     ])

#     last = 0
#     while True:
#         data = api.data('+'.join(streams), last_entry_id=last, **kw)
#         last = data[0][1] if data else last
#         data = ptgctl.holoframe.load_all(data)
#         (
#             rgb, depth,
#             T_rig2world, 
#             T_pv2world, 
#             focalX, focalY, 
#             principalX, principalY,
#         ) = ptgctl.holoframe.unpack(
#             data, [
#             'main.image', 
#             'depthlt.image', 
#             'depthlt.rig2world', 
#             'main.cam2world', 
#             'main.focalX', 
#             'main.focalY', 
#             'main.principalX',
#             'main.principalY',
#         ])
#         rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

#         results = reid.process_frame(
#             rgb, depth, lut, 
#             T_rig2world, 
#             T_rig2cam, 
#             T_pv2world, 
#             [focalX, focalY], 
#             [principalX, principalY],
#         )
#         out_img = drawer.draw_4panel(results)
#         cv2.imshow('main', out_img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break


if __name__ == '__main__':
    import fire
    fire.Fire(main_stream)
