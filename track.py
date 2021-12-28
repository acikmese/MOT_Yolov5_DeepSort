# limit the number of cpus used by high performance libraries
import os
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages
from fake_loader import LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

import pandas as pd
from datetime import datetime
from small_tools import convert_one_dimension
import subprocess
from pymediainfo import MediaInfo

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def distance(pixel_height, focal_length, average_height, cls):
    min_height = average_height - 0.2
    max_height = average_height + 0.2
    if cls == 0:  # human
        average_height = average_height
        min_height = min_height
        max_height = max_height
    elif cls == 2:  # car
        average_height = 1.5
        min_height = 1.4
        max_height = 1.9
    elif cls == 5:  # bus
        average_height = 3.0
        min_height = 2.5
        max_height = 4.0
    elif cls == 7:  # truck
        average_height = 3.0
        min_height = 2.0
        max_height = 4.0
    else:
        average_height = 0
    if average_height != 0:
        d = focal_length * average_height / pixel_height
        d_min = focal_length * min_height / pixel_height
        d_max = focal_length * max_height / pixel_height
    else:
        d = 0
        d_min = 0
        d_max = 0
    return d, d_min, d_max


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, \
    exist_ok, save_datetime, datetime_date, datetime_time, gps_path, focal_length, average_height, distance_thres = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok, opt.save_datetime, \
        opt.datetime_date, opt.datetime_time, opt.gps_path, opt.focal_length, opt.average_height, opt.distance_thres
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    # device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'
    df_path = str(Path(save_dir)) + '/' + txt_file_name + '_with_datetime.csv'
    save_path = vid_path

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        t4, t5 = 0, 0  # Initialize times for deep sort.

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                fps = vid_cap.get(cv2.CAP_PROP_FPS)

            video_height = im0.shape[0]
            if video_height != 1080:
                focal_l = focal_length * (video_height / 1080)
            else:
                focal_l = focal_length

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=1, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # Eliminate bus and truck low confidence boxes.
                # TODO: Eliminate low confidence before that part! It will eliminate unnecessary deepsort process.

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        height = output[3] - output[1]
                        # Distance
                        d = distance(height, focal_l, average_height, cls)

                        c = int(cls)  # integer class
                        label = f'{id}| {names[c]} | {conf:.2f} | {d[0]:.1f}m ({d[1]:.1f}, {d[2]:.1f})'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        if save_txt:
                            # Check if file is empty. If it is, add column names.
                            if not os.path.isfile(txt_path) or os.stat(txt_path).st_size == 0:
                                with open(txt_path, 'a') as f:
                                    f.write(('%s' + ',%s' * 10 + '\n') % ("frame_id", "id_type", "id", "confidence",
                                                                          "bbox_left", "bbox_top", "bbox_w", "bbox_h",
                                                                          "avg_distance", "min_distance",
                                                                          "max_distance")
                                            )

                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            d_avg = d[0]
                            d_min = d[1]
                            d_max = d[2]
                            conf_value = conf.item()
                            id_type = names[c]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g,%s' + ',%g' * 9 + '\n') % (frame_idx, id_type, id, conf_value,
                                                                        bbox_left, bbox_top, bbox_w, bbox_h,
                                                                        d_avg, d_min, d_max)
                                        )

            else:
                deepsort.increment_ages()

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        # if platform == 'darwin':  # MacOS
        #     os.system('open ' + save_path)

    # DISCARDED, THIS CAN ONLY WORK WITH VIDEO FILES! DOES NOT WORK WITH CAMERA STREAMS!
    if save_datetime:
        df = pd.read_csv(txt_path)  # Read text output.
        df['video_second'] = (df.frame_id // fps).astype(int)  # Generate video seconds according to frames.
        # Eliminate low confidence buses and trucks, because there are a lot of wrong detections.
        big_veh_drop = df.loc[((df.id_type == 'bus') | (df.id_type == 'truck')) & (df.confidence <= 0.75)].index.tolist()
        car_drop = df.loc[(df.id_type == 'car') & (df.confidence <= 0.5)].index.tolist()
        # Eliminate distant objects.
        distance_drop = []
        if distance_thres != 0:
            distance_drop = df.loc[df.avg_distance > distance_thres].index.tolist()
        indexes_to_drop = big_veh_drop + car_drop + distance_drop
        df = df.drop(indexes_to_drop)
        df_grp = df.groupby(by=['video_second', 'id_type'])['id'].unique()  # Get ids from each class for each second.
        df_grp = df_grp.unstack(level=1).reset_index()
        start_datetime = datetime.fromisoformat(datetime_date + ' ' + datetime_time)
        df_grp.insert(0, 'datetime', df_grp.video_second.astype('timedelta64[s]') + start_datetime)  # Add datetime.
        # Print total number of ids.
        class_columns = df_grp.drop(['datetime', 'video_second'], axis=1).columns.tolist()
        for c_id in class_columns:
            id_count = convert_one_dimension(df_grp[c_id]).nunique()
            print(f"Total number of {c_id}: {id_count}")

        # If we have gps data, sync gps according to datetime.
        if gps_path != '0' and save_datetime:
            # df_grp['datetime'] = pd.to_datetime(df_grp['datetime'], utc=True)
            gps = pd.read_csv(gps_path, parse_dates=['timestamp'])
            gps = gps.sort_values(by=['timestamp'])
            gps['timestamp'] = gps.timestamp.dt.tz_localize(None)
            df_f = pd.merge_asof(df_grp, gps, left_on='datetime', right_on='timestamp', direction='nearest')
            df_f['zipcode'] = df_f.reverse_geo.str.split(',').str[-1]
            df_f['market'] = df_f.reverse_geo.str.split(',').str[-2]
            df_f = df_f.drop(columns=['id', 'timestamp'])
            df_f.to_csv(df_path, index=False)
        else:  # Just write without gps info.
            df_grp.to_csv(df_path, index=False)

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5x.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x1_0')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-datetime', action='store_true', help='collect ids for each second starting from this datetime')
    parser.add_argument('--datetime-date', type=str, default='2021-01-01', help='starting date for datetime (year-month-day')
    parser.add_argument('--datetime-time', type=str, default='00:00:00', help='starting time for datetime (hour:minute:second')
    parser.add_argument('--gps_path', type=str, default='0', help='gps file for gps position sync')
    parser.add_argument('--focal_length', type=int, default=1000, help='focal length of the camera in pixels.')
    parser.add_argument('--average_height', type=float, default=1.75, help='average height of a human in meters.')
    parser.add_argument('--distance_thres', type=int, default=0, help='distance threshold to not count.')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        save_path = detect(opt)

    # Lower bitrate of output file for sharing!
    if opt.save_vid and os.path.exists(save_path):
        print("WAIT! Changing bitrate for smaller file size...")
        txt_file_name = opt.source.split('/')[-1].split('.')[0]
        mp4_path = save_path
        conv_path = str(save_path.split('.')[0] + '_yolov5' + '.mp4')
        # Change bitrate according to input video bitrate.
        input_media_info = MediaInfo.parse(save_path)
        output_media_info = MediaInfo.parse(mp4_path)
        input_bitrate = 0
        output_bitrate = 0
        for track in input_media_info.tracks:
            input_bitrate = track.bit_rate
        for track in output_media_info.tracks:
            output_bitrate = track.bit_rate
        if output_bitrate > input_bitrate:
            ffmpeg_command = ["ffmpeg", "-i", mp4_path, "-c:v", "libx264", "-b:v", str(int(input_bitrate * 1.5)),
                              conv_path]
            subprocess.run(ffmpeg_command, stderr=subprocess.DEVNULL)
            subprocess.run(["rm", mp4_path], stderr=subprocess.DEVNULL)
        print("Bitrate change finished!")
