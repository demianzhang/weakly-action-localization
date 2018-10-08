python extract_features.py -mode rgb -load_model models/rgb_imagenet.pt -gpu 0 -root /media/zjg/workspace/video/validation -split /media/zjg/workspace/action/data/thumos14_val.json -save_dir /media/zjg/workspace/action/npy/
python extract_features.py -mode rgb -load_model models/rgb_imagenet.pt -gpu 1 -root /media/zjg/workspace/video/test -split /media/zjg/workspace/action/data/thumos14_test.json -save_dir /media/zjg/workspace/action/npy/
python extract_features.py -mode flow -load_model models/flow_imagenet.pt -gpu 0 -root /media/zjg/workspace/video/validation -split /media/zjg/workspace/action/data/thumos14_val.json -save_dir /media/zjg/workspace/action/npy/
python extract_features.py -mode flow -load_model models/flow_imagenet.pt -gpu 1 -root /media/zjg/workspace/video/test -split /media/zjg/workspace/action/data/thumos14_test.json -save_dir /media/zjg/workspace/action/npy/

train:
python train_model.py -mode rgb -model_file thumos
python train_model.py -mode rgb -model_file thumos -resume True
python train_model.py -mode flow -model_file thumos
python train_model.py -mode flow -model_file thumos -resume True

CAM:
python train_model.py -mode rgb -model_file thumos -train False
python train_model.py -mode flow -model_file thumos -train False


test:
rgb: 0.351 0.279 0.213 0.150 0.102
flow: 0.408 0.340 0.269 0.205 0.145
all: - - - - -