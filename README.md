# Work in Progress

I'm in progress of cleaning up my original code and putting it here. So please wait a little while.

__TODO__:
 - [X] Extracting code from notebook about postprocessing.
 - [X] Extracting code from notebook about visualization.
 - [X] Adding description on how to use the pretrined model.
 - [X] Adding description on how to train a new model from CUB.
 - [ ] Adding description on how to train a new model from own data.
 - [ ] Adding code for PCP evaluation.
 - [ ] Adding bounding box regression for better PCP.


# fast-bird-part-localization
Code for Fast Bird Part Localization part of the following paper:

[Fast Bird Part Localization for Fine-Grained Categorization](http://yassersouri.github.io/papers/fgvc-2015-fast-bird-part.pdf)    
Yaser Souri, Shohreh Kasaei    
The Third Workshop on Fine-Grained Visual Categorization (FGVC3) in conjunction with CVPR 2015

The code for classification part is very simple and not included in this repository.


## Requirements

0. Python 2.7. This might not work with Python 3.
1. A recent installation of [caffe](http://caffe.berkeleyvision.org) with its python wrapper. (I have installed [this version](https://github.com/BVLC/caffe/tree/72d70892ad489815589b8e680813c350610b3f2a) of caffe.)
2. OpenCV 2.4 with its Python wrapper.
3. All other python requirements are mentioned in `requirements.txt`

## Getting Started

For testing or training you need the pretrained CaffeNet network. You can download it from [this url](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel). After downloading it, make sure you change the `src/settings.py` file and change the `CAFFE_NET_PRETRAINED` variable accordingly.

## Training a new head detector

This can be done using the CUB dataset. First download the CUB-200-2011 dataset from [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and extract it.
Then change `src/settings.py` file and set the `CUB_FOLDER_PATH` varialble accordingly.

Then run the following command:
```shell
python create_rf_model.py
```

This will create a head detector for you in the models directory. To run this script you will night large amount of RAM (~30GB).

Changing `part_name` variable in `create_rf_model.py` file to `body` instead of `head` will create a detector for body.

### Testing

```python
import sys
sys.path.append('/path/to/projectroot/')
from fast_bird_part_localization import settings
sys.path.append(settings.CAFFE_PYTHON_PATH)
import caffe
from fast_bird_part_localization import detector

fbp = detector('/path/to/project/models/head_model.mdl')

img = caffe.io.load_image('/path/to/bird.jpg')
head, head_prob = fbp.detect(img)
fbp.draw(img, head, head_prob)
```
This is the result you get:

![Result](https://github.com/yassersouri/fast-bird-part-localization/blob/master/result.png)
