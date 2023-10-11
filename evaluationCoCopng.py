import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import time
import torch
np.set_printoptions(threshold=np.inf)


categories = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
              'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse',
              'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
              'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseballbat', 'baseballglove',
              'skateboard', 'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon',
              'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
              'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse',
              'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
              'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush']

COCO_LABEL_MAP ={ 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18:17, 19:18, 20:19, 21:20, 22:21, 23:22, 24:23, 25:24,
                  27:25, 28:26, 31:27, 32:28, 33:29, 34:30, 35:31, 36:32,
                  37:33, 38:34, 39:35, 40:36, 41:37, 42:38, 43:39, 44:40,
                  46:41, 47:42, 48:43, 49:44, 50:45, 51:46, 52:47, 53:48,
                  54:49, 55:50, 56:51, 57:52, 58:53, 59:54, 60:55, 61:56,
                  62:57, 63:58, 64:59, 65:60, 67: 61, 70: 62, 72:63, 73:64,
                  74:65, 75:66, 76:67, 77:68, 78:69, 79:70, 80:71, 81:72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89:79, 90:80}




def do_python_eval(predict_folder, gt_folder, name_list, num_cls=81, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    def compare(start, step, TP, P, T, input_type):
        for idx in range(start, len(name_list), step):
            name = name_list[idx]
            # print(f'name: {name}')
            # print('---------------------------------------------------------------------')
            if input_type == 'png':
                predict_file = os.path.join(predict_folder, '%s.png' % name)
                predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
                # print(f'predict-name: {predict_file}')
                # print(f'predict:{predict}')
                # print('predict---------------------------------------------------------------------')
                # if num_cls == 81:
                #     predict = predict - 91
                # print(f'predict-91-name: {predict_file}')
                # print(f'----predict-91:{predict}')
                # print('predict-91---------------------------------------------------------------------')

            gt_file = os.path.join(gt_folder, '%s.png' % name)
            gt = np.array(Image.open(gt_file))
            # print(f'gt-name: {gt_file}')
            # print(f'gt:{gt}')
            # print('gt---------------------------------------------------------------------')

            cal = gt < 255
            mask = (predict == gt) * cal

            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict == i) * cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt == i) * cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt == i) * mask)
                TP[i].release()

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T, input_type))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    for i in range(num_cls):
        IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
        T_TP.append(T[i].value / (TP[i].value + 1e-10))
        P_TP.append(P[i].value / (TP[i].value + 1e-10))
        FP_ALL.append((P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100

    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    fp = np.mean(np.array(FP_ALL))
    loglist['FP'] = fp * 100
    fn = np.mean(np.array(FN_ALL))
    loglist['FN'] = fn * 100
    if printlog:
        for i in range(num_cls):
            if i % 2 != 1:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100), end='\t')
            else:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100))
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', miou * 100))
        print('\n')
        print(f'FP = {fp * 100}, FN = {fn * 100}')
    return loglist


def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  ' % (key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)


def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath, 'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n' % comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='/data/c425/tjf/mct_mod27/coco/coco_train_labeled_list.txt', type=str)
    parser.add_argument("--predict_dir",
                        default='/data/c425/tjf/mct_mod27/MCTformer_results/MCTformer_v1/coco/fused-patchrefine-npy',
                        type=str)
    parser.add_argument("--gt_dir", default='/data/c425/tjf/datasets/COCO2014/train2014_annotations', type=str)
    parser.add_argument('--logfile', default='./evallog_coco.txt', type=str)
    parser.add_argument('--comment', required=True, type=str)
    parser.add_argument('--type', default='png', choices=['npy', 'png'], type=str)
    parser.add_argument('--t', default=None, type=float)
    parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument('--num_classes', default=81, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=60, type=int)
    args = parser.parse_args()

    if args.type == 'npy':
        assert args.t is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    if not args.curve:
        l = []
        max_mIoU = 0.0
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type,
                                 printlog=True)
        l.append(loglist['mIoU'])
        if loglist['mIoU'] > max_mIoU:
            max_mIoU = loglist['mIoU']
        print('mIoU: %.3f%%' % (max_mIoU))
        writelog(args.logfile, loglist, args.comment)

    # a = torch.tensor([[0, 1, 49], [7, 43, 49]])
    # print(a)
    # b = map(lambda x: COCO_LABEL_MAP[x], a[0])
    # # b = COCO_LABEL_MAP[43]
    # print(b)
    # print(list(b))
    # # COCO_LABEL_MAP
