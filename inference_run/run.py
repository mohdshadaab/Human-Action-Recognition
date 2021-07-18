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
	
#results = inference_recognizer(model, video, label)

#for result in results:
    #print(f'{result[0]}: ', result[1])
