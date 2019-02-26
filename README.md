# Bottom-Up Features Extractor

This code implements an extraction of Bottom-up image features ([paper](https://arxiv.org/abs/1707.07998)). Based on the original [bottom-up attention model](https://github.com/peteanderson80/bottom-up-attention/) and [PyTorch implementation of Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch).

## Requirements
* Python 3.6
* PyTorch 0.4.0
* CUDA 9.0

**Note:** CPU version is not supported.

## Installation
1. Clone the code:
    ```
    git clone https://github.com/violetteshev/bottom-up-features.git
    ```

2. Install PyTorch with pip:
    ```
    pip install https://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
    ```
    or with Anaconda:
    ```
    conda install pytorch=0.4.0 cuda90 -c pytorch
    ```

3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Compile the code:
    ```
    cd lib
    sh make.sh
    ```

5. Download the [pretrained model](https://www.dropbox.com/s/qo4xf1dx3oxi1h6/bottomup_pretrained_10_100.pth?dl=0) and put it in models/ folder.

## Feature Extraction

1. To extract image features and store them in .npy format:
    ```
    python extract_features.py --image_dir images --out_dir features
    ```

2. To save bounding boxes use `--boxes` argument:
    ```
    python extract_features.py --image_dir images --out_dir features --boxes
    ```
