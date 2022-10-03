import os
import tqdm
import numpy as np
import cv2

class VideoOutput:#'avc1', 'mp4v', 
    prev_im = None
    t_video = 0
    def __init__(self, src=None, fps=None, cc='mp4v', cc_fallback='avc1', fixed_fps=False, show=None):
        self.src: str = src
        self.cc = cc
        self.cc_fallback = cc_fallback
        self.fps: float = fps
        self.fixed_fps = fixed_fps
        self._show = not src if show is None else show
        self.active = self.src or self._show

    def __enter__(self):
        self.prev_im = None
        return self

    def __exit__(self, *a):
        if self._w:
            self._w.release()
        self._w = None
        self.prev_im = None
        if self._show:
            cv2.destroyAllWindows()
    async def __aenter__(self): return self.__enter__()
    async def __aexit__(self, *a): return self.__exit__(*a)

    def output(self, im, t=None):
        if issubclass(im.dtype.type, np.floating):
            im = (255*im).astype('uint8')
        if self.src:
            if self.fixed_fps and t is not None:
                self.write_video_fixed_fps(im, t)
            else:
                self.write_video(im)
        if self._show:
            self.show_video(im)

    _w = None
    def write_video(self, im):
        if not self._w:
            ccs = [self.cc, self.cc_fallback]
            for cc in ccs:
                os.makedirs(os.path.dirname(self.src) or '.', exist_ok=True)
                self._w = cv2.VideoWriter(
                    self.src, cv2.VideoWriter_fourcc(*cc),
                    self.fps, im.shape[:2][::-1], True)
                if self._w.isOpened():
                    break
                print(f"{cc} didn't work trying next...")
            else:
                raise RuntimeError(f"Video writer did not open - none worked: {ccs}")
        self._w.write(im)

    def write_video_fixed_fps(self, im, t):
        if self.prev_im is None:
            self.prev_im = im
            self.t_video = t

        while self.t_video < t:
            self.write_video(self.prev_im)
            self.t_video += 1./self.fps
        self.write_video(im)
        self.t_video += 1./self.fps
        self.prev_im = im

    def show_video(self, im):
        cv2.imshow('output', im)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration


class VideoInput:
    def __init__(self, 
            src, fps=None, size=None, give_time=True, 
            start_frame=None, stop_frame=None, 
            bad_frames_count=True, 
            include_bad_frame=False):
        self.src = src
        self.dest_fps = fps
        self.size = size
        self.bad_frames_count = bad_frames_count
        self.include_bad_frame = include_bad_frame
        self.give_time = give_time
        self.start_frame = start_frame
        self.stop_frame = stop_frame

    def __enter__(self):
        self.cap = cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.src}")
        self.src_fps = src_fps = cap.get(cv2.CAP_PROP_FPS)
        self.every = max(1, round(src_fps/(self.dest_fps or src_fps)))
        size = self.size or (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame = np.zeros(tuple(size)+(3,)).astype('uint8')

        if self.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        self.total = total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"{total/src_fps:.1f} second video. {total} frames @ {self.src_fps} fps,",
              f"reducing to {self.dest_fps} fps" if self.dest_fps else '')
        self.pbar = tqdm.tqdm(total=int(total))
        return self

    def __exit__(self, *a):
        self.cap.release()

    def read_all(self, limit=None):
        ims = []
        with self:
            for t, im in self:
                if limit and t > limit/self.dest_fps:
                    break
                ims.append(im)
        return np.stack(ims)

    def __iter__(self):
        i = self.start_frame or 0
        while not self.total or self.pbar.n < self.total:
            ret, im = self.cap.read()
            self.pbar.update()

            if not ret:
                self.pbar.set_description(f"bad frame: {ret} {im}")
                if not self.include_bad_frame:
                    continue
                im = self.frame
            self.frame = im

            i = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if self.stop_frame and i > self.stop_frame:
                break

            if i%self.every:
                continue
            if self.size:
                im = cv2.resize(im, self.size)

            t = i / self.src_fps
            self.pbar.set_description(f"t={t:.1f}s")
            yield t if self.give_time else i, im
