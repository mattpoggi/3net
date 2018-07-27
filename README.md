# 3net
This repository contains the source code of 3net, proposed in the paper "Learning monocular depth estimation with unsupervised trinocular assumptions", 3DV 2018.

<p align="center">
<iframe width="380" height="260" src="https://www.youtube.com/embed/uMA5YWJME4M?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
</p>

If you use this code in your projects, please cite our paper:

```
@inproceedings{3net18,
  title     = {Learning monocular depth estimation with unsupervised trinocular assumptions},
  author    = {Matteo Poggi and
               Fabio Tosi and
               Stefano Mattoccia},
  booktitle = {6th International Conference on 3D Vision (3DV)},
  year = {2018}
}
```

Demo video:
[youtube](https://www.youtube.com/watch?v=uMA5YWJME4M)

## Requirements

* `Tensorflow 1.8` (recomended) 
* `python packages` such as opencv, matplotlib

## Run 3net on webcam stream

To run 3net, just launch

```
python webcam.py --checkpoint_dir /checkpoint/3DV18/3net --mode [0,1,2]
```

While the demo is running, you can press:

* 'm' to change mode (0: depth-from-mono, 1: depth + view synthesis, 2: depth + view + SGM)
* 'p' to pause the stream
* 'ESC' to quit

## Train 3net from scratch

Code for training will be (eventually) uploaded.
Meanwhile, you can train pydnet by embedding it into https://github.com/mrharicot/monodepth

