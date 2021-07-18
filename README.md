# Human-Action-Recognition
**A Human Action Recognition pipeline using MMAction2 and kinetics400 dataset. MMAction2 is an open-source toolbox for video understanding based on PyTorch.**

## Installation and Getting Started

### Install MMAction2

```BASH
# install dependencies: (for cuda 10.1), remove "+cu101" and install for cuda 11+
pip install -U torch==1.8.0+cu101 torchvision==0.9.0+cu101 torchtext==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# install mmcv-full thus we could use CUDA operators
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html

# Install mmaction2
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2

pip install -e .

# Install some optional requirements
pip install -r requirements/optional.txt
```
### Check Installations

```Python
# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMAction2 installation
import mmaction
print(mmaction.__version__)

# Check MMCV installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())
```
### Downloading Pre-Trained Models

```BASH
mkdir checkpoints
wget -c https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
      -O checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth
```
You can find link to more pre-trained models below.

### Create Inference-run folder

```BASH
mkdir inference_run
cd inference_run
```
### Config file for saving paths
```BASH
config.py
```
```Python
configs={
	"Input_path": "../data/input_data/",
	"Output_path": "../data/output_data/",
	"Label": "../demo/label_map_k400.txt",
	"Checkpoint": "../checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth",
	"Recognizer_config": "../configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py"
}
```

### Performing inference with MMAction2 recognizer
```BASH
run.py
```
```Python
#import stuff
import torch, torchvision
import mmaction
from tqdm import tqdm
from pathlib import Path
import json
from config import configs
from mmaction.apis import inference_recognizer, init_recognizer

# Choose to use a config and initialize the recognizer

config = configs["Recognizer_config"]
# Setup a checkpoint file to load
checkpoint = configs["Checkpoint"]
input_path= configs["Input_path"]

output_path=configs["Output_path"]
label = configs["Label"]

# Initialize the recognizer
#model = init_recognizer(config, checkpoint, device='cuda:0') #for using gpu
model = init_recognizer(config, checkpoint, device='cpu')
vid_paths = [str(x) for x in Path(input_path).glob("*.mp4")]
vid_paths=sorted(vid_paths)

for in_path in tqdm(vid_paths):

	name=in_path.split('.mp4')
	name=name[0].split('/')
	name=name[3]
	results = inference_recognizer(model, in_path, label)
	actions={
		"Actions":{}
		}
	print("clip:", name )
	for result in results:
		actions["Actions"][result[0]]=str(result[1])
		
	with open(output_path+name+".json", "w") as f:
		json.dump(actions,f)
```


### Directory Structure
```BASH
./MMAction2/
```
![MMAction2 Directory](/images/main_directory.png)
```BASH
./MMaction2/inference_run/
```
![inference_run directory](/images/inference_run.png)

### Sample Output
![clip9_output](/images/clip9_img.png)

### References and Guides

1. [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)
2. [Microsoft Swin Transformer](https://github.com/microsoft/Swin-Transformer) - You can find more pre-trained models here
3. [MMAction2](https://github.com/open-mmlab/mmaction2) - MMAction2 is an open-source toolbox for video understanding based on PyTorch. You can find many benchmark models and datasets here.
4. [MMAction2 Tutorial](https://colab.research.google.com/github/open-mmlab/mmaction2/blob/master/demo/mmaction2_tutorial.ipynb) - Colab Notebook to perform inference with a MMAction2 recognizer and train a new recognizer with a new dataset.
