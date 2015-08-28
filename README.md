# Work in Progress

I'm in progress of cleaning up my original code and putting it here. So please wait a little while.

# fast-bird-part
Code for Fast Bird Part Localization (FGVC 2015)


## Requirements

0. Python 2.7. This might not work with Python 3.
1. A recent installation of [caffe](http://caffe.berkeleyvision.org) with its python wrapper. (I have installed [this version](https://github.com/BVLC/caffe/tree/72d70892ad489815589b8e680813c350610b3f2a) of caffe.)
2. OpenCV 2.4 with its Python wrapper.
3. All other python requirements are mentioned in `requirements.txt`

## Getting Started

For testing or training you need the pretrained CaffeNet network. You can download it from [this url](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel). After downloading it, make sure you change the `src/settings.py` file and change the `CAFFE_NET_PRETRAINED` variable accordingly.

### Testing

```python
import sys
sys.path.append('src')
import settings
sys.path.append(settings.CAFFE_PYTHON_PATH)
import caffe
import detector

img = caffe.io.load_image('bird.jpg')
head = detector.detect_head(img)
detector.draw_head(img, head)
```
