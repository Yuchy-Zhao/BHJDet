export CUDA_VISIBLE_DEVICES=4,5,6,7

python3 test.py -md rcnn_fpn_baseline -r 30 -d 0-3 -c pos