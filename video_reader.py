import json
import sys
import time
from datetime import datetime
from threading import Thread

import cv2
import torch

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, is_ascii, \
    non_max_suppression, scale_coords
from utils.torch_utils import select_device


def inRegion(c1, c2, rois):
    cx = (c1[0] + c2[0]) // 2
    cy = (c1[1] + c2[1]) // 2
    for i in range(len(rois)):
        roi = rois[i]
        if roi[0] < cx < roi[2] and roi[1] < cy < roi[3]:
            return i
    return None


class VideoReader(object):
    """
    This class is for a VideoReader Object that will run on single
    video using rois and will process the frames for occupancy and
    ppe detection
    Attributes:
            streamPort(str or int)	: The path of the videofile or the webcam port
            rois(list(list)) 	: The nested list of rois
            windowName(str)		: Window name to show the output frame
    """

    def __init__(self, name, streamArg):
        """
        Parameters:
                name(str)       : camera name
                streamArd(dict) : camera data

        """
        super(VideoReader, self).__init__()

        self.counter = 0

        self.current_time = 0
        self.window = name
        # cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(self.window, 640, 420)

        self.streamLink = streamArg["streamLink"]
        self.rois = streamArg["rois"]
        self.drawn_roi = self.rois
        self.roi_kit_label = [None] * len(self.rois)
        self.current_time = [None] * len(self.rois)
        self.capture = None
        self.dataset = None

        self.imgsz = 640
        self.max_det = 1000
        if torch.cuda.is_available():
            self.processing_device = select_device('0')
        else:
            self.processing_device = select_device('cpu')
        self.model = attempt_load(
            'new_model.pt', map_location=self.processing_device)
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.ascii = is_ascii(self.names)
        self.dt, self.seen = [0.0, 0.0, 0.0], 0

        # For videoFiles only
        self.dataset = LoadStreams(self.streamLink, img_size=self.imgsz,
                                  stride=self.stride, auto='.pt')

    def detect(self, img):
        pred = self.model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, 0.50, 0.45, None, False, max_det=self.max_det)

        return pred

    def process(self):
        prev_frame_time = 0

        prev_pred = {'roi0': None, 'roi1': None}
        roi0 = {'In-time': None, 'Out-time': None}
        roi1 = {'In-time': None, 'Out-time': None}
        prev_time = {'t0': None, 't1': None}

        for path, img, im0s, vid_cap in self.dataset:
            ro1_occupied, ro0_occupied = False, False
            new_frame_time = time.time()
            img = torch.from_numpy(img).to(self.processing_device)
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            self.roi_kit_label = [None] * len(self.rois)
            # NMS
            if self.counter == 30 or self.counter == 0:
                self.counter = 0
                now = datetime.now()
                pred = self.detect(img)

                for i, det in enumerate(pred):  # per image
                    p, s, im0, frame = path, '', im0s.copy(), self.dataset.count
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], im0.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])
                                      ), (int(xyxy[2]), int(xyxy[3]))

                            roi_index = inRegion(c1, c2, self.drawn_roi)

                            if roi_index is not None:

                                if roi_index == 0:
                                    ro0_occupied = True
                                    if roi0['In-time'] is None:
                                        roi0['In-time'] = now.strftime("%H:%M:%S")
                                    prev_pred['roi0'] = self.names[c]

                                elif roi_index == 1:
                                    ro1_occupied = True
                                    if roi1['In-time'] is None:
                                        roi1['In-time'] = now.strftime("%H:%M:%S")
                                    prev_pred['roi1'] = self.names[c]

                        if ro0_occupied is False and roi0['In-time'] is not None:
                            roi0['In-time'] = None
                            prev_time['t0'] = now.strftime("%H:%M:%S")
                        if ro1_occupied is False and roi1['In-time'] is not None:
                            roi1['In-time'] = None
                            prev_time['t1'] = now.strftime("%H:%M:%S")

                        xmin, ymin, xmax, ymax = self.drawn_roi[0]
                        cv2.putText(im0, self.roi_kit_label[0], (xmin + 10, ymax - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(im0, 'In-Time :' + str(roi0['In-time']), (xmin + 10, ymax - 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(im0, 'Out-Time :' + str(prev_time['t0']), (xmin + 10, ymax - 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        xmin1, ymin1, xmax1, ymax1 = self.drawn_roi[1]
                        cv2.putText(im0, self.roi_kit_label[1], (xmin1 + 10, ymax1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(im0, 'In-Time :' + str(roi1['In-time']), (xmin1 + 10, ymax1 - 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(im0, 'Out-Time :' + str(prev_time['t1']), (xmin1 + 10, ymax1 - 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)

                        # cv2.putText(im0, self.current_time[i], (xmin + 10, ymax - 100),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 2,
                        #             (0, 0, 255), 3, cv2.LINE_AA)

                im0 = cv2.resize(im0, (620, 400))
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps = int(fps)
                fps = str(fps)

                cv2.putText(im0, fps, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 2, cv2.LINE_AA)
                self.counter += 1
                frame_to_show = im0

            else:
                # print(prev_pred)

                xmin, ymin, xmax, ymax = self.drawn_roi[0]
                cv2.putText(im0s, prev_pred['roi0'], (xmin + 10, ymax - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(im0s, 'In-Time :' + str(roi0['In-time']), (xmin + 10, ymax - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(im0s, 'Out-Time :' + str(prev_time['t0']), (xmin + 10, ymax - 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
                xmin1, ymin1, xmax1, ymax1 = self.drawn_roi[1]
                cv2.putText(im0s, prev_pred['roi1'], (xmin1 + 10, ymax1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(im0s, 'In-Time :' + str(roi1['In-time']), (xmin1 + 10, ymax1 - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(im0s, 'Out-Time :' + str(prev_time['t1']), (xmin1 + 10, ymax1 - 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)

                im0s = cv2.resize(im0s, (620, 400))
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps = int(fps)
                fps = str(fps)

                cv2.putText(im0s, fps, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 2, cv2.LINE_AA)
                self.counter += 1
                frame_to_show = im0s
            cv2.imshow(self.window, frame_to_show)

            if cv2.waitKey(1) & 0xff == 27:
                return


if __name__ == '__main__':

    devices = json.loads(open('camera_ports.json').read())

    vrs = []

    for i, j in devices.items():
        vr = VideoReader(i, j)
        vrs.append(vr)

    threads = []

    for vr in vrs:
        threads.append(Thread(target=vr.process, args=()))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    sys.exit()
