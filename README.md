# Weakly Supervised Action Localization by Sparse Temporal Pooling Network

## Overview

This repository contains reproduced code reported in the paper "[Weakly Supervised Action Localization by Sparse Temporal Pooling Network](https://arxiv.org/abs/1712.05080)" by Phuc Nguyen and Ting Liu.
The paper was posted on arXiv in Dec 2017, published as a CVPR 2018 conference paper.

Disclaimer: This is the reproduced code, not an original code of the paper.

## Running the code

see the multi_run.sh

firstly you need to use [Dense_Flow](https://github.com/yjxiong/dense_flow) to extract rgb and optical flow

secondly you need to extract I3D feature using [thumos14-i3d](https://github.com/demianzhang/thumos14-i3d)

finally you can train, test and combine the rgb and flow result

### result

Model          |  0.1  |  0.2  |  0.3  |  0.4  |  0.5
-------------- | :---: | :---: | :---: | :---: | ----
RGB-I3D        | 0.351 | 0.279 | 0.213 | 0.150 | 0.102
Flow-I3D       | 0.408 | 0.340 | 0.269 | 0.205 | 0.144
Two-Stream I3D | - | - | - | - | -

