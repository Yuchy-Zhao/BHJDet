import os
import sys
import cv2
import argparse
import numpy as np

sys.path.insert(0, './lib')
from utils import misc_utils, visual_utils

def draw_box(img, one_box, score=None, tag=None, line_thick=1, line_color='white'):
    width = img.shape[1]
    height = img.shape[0]
    one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                min(one_box[2], width - 1), min(one_box[3], height - 1)])
    x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
    cv2.rectangle(img, (x1,y1), (x2,y2), line_color, line_thick)
    if score is not None:
        text = "{} {:.3f}".format(tag, score)
        cv2.putText(img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 0.5, line_color, line_thick)
    return img

# img_root = '/data/CrowdHuman/images/'
img_root = '/data/yunqi/Track/BHJDet/data/CrowdHuman/Images/'
test_root = '/data/yunqi/Track/BHJDet/model/rcnn_fpn_baseline/outputs_pos/eval_dump/bf_match_bbox.json'
def eval_all(args):
    # json file
    assert os.path.exists(args.json_file), "Wrong json path!"
    misc_utils.ensure_dir('outputs')
    records_imgs = misc_utils.load_json_lines(test_root)
    records = misc_utils.load_json_lines(args.json_file)[:args.number]
    for i, record in enumerate(records):
        dtboxes = misc_utils.load_bboxes(
                record, key_name='dtboxes', key_box='box', key_score='score', key_tag='tag')
        gtboxes = misc_utils.load_bboxes(record, 'gtboxes', 'box')
        dtboxes = misc_utils.xywh_to_xyxy(dtboxes)
        gtboxes = misc_utils.xywh_to_xyxy(gtboxes)
        keep = dtboxes[:, -2] > args.visual_thresh
        dtboxes = dtboxes[keep]
        len_dt = len(dtboxes)
        len_gt = len(gtboxes)
        line = "{}: dt:{}, gt:{}.".format(record['ID'], len_dt, len_gt)
        print(line)
        img_path = img_root + record['ID']
        img = misc_utils.load_img(img_path)
        visual_utils.draw_boxes(img, dtboxes, line_thick=1, line_color='blue')
        visual_utils.draw_boxes(img, gtboxes, line_thick=1, line_color='white')
        fpath = 'outputs/{}.jpg'.format(record['ID'])
        cv2.imwrite(fpath, img)

        img = misc_utils.load_img(img_path)
        for det in records_imgs[0]:
            if det['image_id'] == i:
                color = np.random.randint(256, size=3)
                color = (int(color[0]), int(color[1]), int(color[2]))
                bbox = np.array(det['bbox'])
                bbox[2:4] += bbox[:2]
                score = det['score']
                if score > args.visual_thresh:
                    draw_box(img, bbox, line_thick=1, line_color=color)
                f_bbox = np.array(det['f_bbox'])
                f_bbox[2:4] += f_bbox[:2]
                f_score = det['f_score']
                if f_score > args.visual_thresh:
                    draw_box(img, f_bbox, line_thick=1, line_color=color)
        fpath = 'outputs/{}_joint.jpg'.format(record['ID'])
        cv2.imwrite(fpath, img)

def run_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', '-f', default=None, required=True, type=str)
    parser.add_argument('--number', '-n', default=3, type=int)
    parser.add_argument('--visual_thresh', '-v', default=0.5, type=float)
    args = parser.parse_args()
    eval_all(args)

if __name__ == '__main__':
    run_eval()
