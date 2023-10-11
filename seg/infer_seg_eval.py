import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import importlib
from tool import imutils
import argparse
import cv2
import os.path
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from tool.metrics import Evaluator
import PIL.Image as Image

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]


import imgviz



classes = np.array(('background',  # always index 0
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor'))

classes_coco = np.array(('background','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
              'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse',
              'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
              'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseballbat', 'baseballglove',
              'skateboard', 'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon',
              'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
              'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse',
              'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
              'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush'))

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1','True'):
        return True
    elif v.lower() in ('no','false','f','n','0','False'):
        return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU_id')
    parser.add_argument("--weights", default="", type=str)
    parser.add_argument("--network", default="", type=str)
    parser.add_argument("--gt_path", required=True, type=str)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--save_path_c", default=None, type=str)
    parser.add_argument("--list_path", default="./voc12/val_id.txt", type=str)
    parser.add_argument("--img_path", default="", type=str)
    parser.add_argument("--num_classes", default=81, type=int)
    parser.add_argument("--use_crf", default=False, type=str2bool)
    parser.add_argument("--scales", type=float, nargs='+')
    args = parser.parse_args()

    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    Path(args.save_path_c).mkdir(parents=True, exist_ok=True)


    seg_evaluator = Evaluator(num_class=args.num_classes)

    im_path = args.img_path
    img_list = open(args.list_path).readlines()

    colormap = imgviz.label_colormap()

    with torch.no_grad():
        for idx in tqdm(range(len(img_list))):
            i = img_list[idx]

            pred = Image.open(os.path.join(args.save_path_c, i.strip() + '.png'))
            pred = np.asarray(pred)

            gt = Image.open(os.path.join(args.gt_path, i.strip() + '.png'))
            gt = np.asarray(gt)

            seg_evaluator.add_batch(gt, pred)

        IoU, mIoU = seg_evaluator.Mean_Intersection_over_Union()

        str_format = "{:<15s}\t{:<15.2%}"
        filename = os.path.join(args.save_path, 'result.txt')
        with open(filename, 'w') as f:
            for k in range(args.num_classes):
                print(str_format.format(classes_coco[k], IoU[k]))
                f.write('class {:2d} {:12} IU {:.3f}'.format(k, classes_coco[k], IoU[k]) + '\n')
            f.write('mIoU = {:.3f}'.format(mIoU) + '\n')
        print(f'mIoU={mIoU:.3f}')