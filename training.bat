DEL .\images\train.cache /F /Q
DEL .\images\valid.cache /F /Q
python train.py --batch 8 --cfg cfg/training/yolov7.yaml --epochs 300 --data ./images/data.yaml --weights 'yolov7.pt' --device 0 --name highway --hyp data/hyp.scratch.p6.yaml --workers 2
pause