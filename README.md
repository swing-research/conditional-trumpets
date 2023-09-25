# Conditional Injective Flows for Bayesian Imaging

[![Paper](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2204.07664)
[![PWC](https://img.shields.io/badge/PWC-report-blue)](https://paperswithcode.com/paper/conditional-injective-flows-for-bayesian)

This repository is the official Tensorflow Python implementation of "[Conditional Injective Flows for Bayesian Imaging](https://arxiv.org/abs/2204.07664)" published in IEEE Transactions on Computational Imaging, 2023.

| [**Project Page**](https://sada.dmi.unibas.ch/en/research/injective-flows)  | 


<p float="center">
<img src="https://github.com/swing-research/conditional-trumpets/blob/main/figures/network.png" width="1000">
</p>



## Requirements
(This code is tested with tensorflow-gpu 2.3.0, Python 3.8.3, CUDA 11.0 and cuDNN 7.)
- numpy
- scipy
- matplotlib
- sklearn
- opencv-python
- tensorflow-gpu==2.3.0
- tensorflow-probability==0.11.1

## Installation

Run the following code to install all pip packages:
```sh
pip install -r requirements.txt 
```

## Experiments
### Limited-view CT:
We used [LoDoPaB-CT](https://www.nature.com/articles/s41597-021-00893-z) dataset. For resolution 64x64, download the dataset from [here](https://drive.switch.ch/index.php/s/tDym90atqRrLNm4) and unzip the file. Put the .npz and .npy files in folder datasets/limited-CT/.
For resolution 256x256, download the dataset from [here](https://drive.switch.ch/index.php/s/lQeYWmAIYcEEdlc) and unzip the file. Put the images folder in datasets/limited-CT/images.
Run the following command to train the model. For this problem, these specific arguments should be specified as well as the general arguments: resolution (64 or 256) and missing_cone (vertical or horizontal): This is an example of resolution 64 and vertical missing-cone:
```sh
python3 train.py --train 1 --num_epochs 300 --batch_size 64 --dataset limited-CT --lr 0.0001 --ml_threshold 150 --model_depth 3 --latent_depth 4 --learntop 1 --gpu_num 0 --remove_all 1 --desc default --problem limited-CT --resolution 64 --missing_cone vertical
```
The additive noise is 25dB for resolution 64x64 and 40dB for resolution 256x256. 
Each argument is explained in detail in utils.py script.


### Electromagnetic inverse scattering:
Download the datasets for different setups from [here](https://drive.switch.ch/index.php/s/ov6OiLyc3V2xEo2). Put the .npz files in folder datasets/scattering/.
Run the following command to train the model. For this problem, the specific arguments are: epsilon_r (1.5,2 or 6), setup (full or slice) and conditions (es or bp). This command is an example of epsilon_r=6, full angle view (slice is for top view) and using scattered fields as conditioning samples:
```sh
python3 train.py --train 1 --num_epochs 300 --batch_size 64 --dataset scattering --lr 0.0001 --ml_threshold 150 --model_depth 3 --latent_depth 4 --learntop 1 --gpu_num 0 --remove_all 1 --desc default --problem scattering --epsilon_r 6 --setup full --conditions es 
```
For all posible setups, 30dB noise is added to the scattered fields.

### Traveltime tomography:
Download the forward operator from [here](https://drive.switch.ch/index.php/s/z0FlOFdbtM3Koi7). Put the .npy files in folder datasets/traveltime/.
Run the following command to train the model. For this problem, the specific argument is: noise_snr. This command is an example of 40dB noise:
```sh
python3 train.py --train 1 --num_epochs 300 --batch_size 64 --dataset mnist --lr 0.0001 --ml_threshold 150 --model_depth 3 --latent_depth 4 --learntop 1 --gpu_num 0 --remove_all 1 --desc default --problem traveltime --noise_snr 40
```

### Image restoration tasks:
Download the Celeba dataset from [here](https://drive.switch.ch/index.php/s/bLwsT2zA0nKH4kT). Put the .npy file in folder datasets/celeba/.
Run the following command to train the model. For these problems, the specific arguments are: problem (denoising, sr, mask or random_mask), problem_factor (the specific factor of each problem) and noise_snr. This command is an example of the super-resolution problem with factor x4 and 30dB additive noise:
```sh
python3 train.py --train 1 --num_epochs 300 --batch_size 64 --dataset celeba --lr 0.0001 --ml_threshold 150 --model_depth 3 --latent_depth 4 --learntop 1 --gpu_num 0 --remove_all 1 --desc default --problem sr --problem_factor 4 --noise_snr 30
```
Denoising task does not have a problem_factor argument.

### Class-based image generation
We performed this task over MNIST and ten identities of the [Voxceleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) face dataset. Download Voxceleb dataset from [here](https://drive.switch.ch/index.php/s/XbemLQoUFDx1eCh). Put the .npy file in folder datasets/voxceleb/. Run the following command to train the model. This command is an example of class-based image generation task (cs argument for problem) over Voxceleb dataset:
```sh
python3 train.py --num_epochs 3000 --batch_size 64 --dataset voxceleb --lr 0.0001 --ml_threshold 1500 --model_depth 3 --latent_depth 4 --learntop 1 --gpu_num 0 --remove_all 1 --desc default --problem cs --train 1 
```
We trained C-Trumpets for 3000 epochs over Voxceleb (as the number of training data is small) and 300 epochs over MNIST.


## Citation
If you find the code or our dataset useful in your research, please consider citing the paper.

```
@article{khorashadizadeh2023conditional,
  title={Conditional injective flows for Bayesian imaging},
  author={Khorashadizadeh, AmirEhsan and Kothari, Konik and Salsi, Leonardo and Harandi, Ali Aghababaei and de Hoop, Maarten and Dokmani{\'c}, Ivan},
  journal={IEEE Transactions on Computational Imaging},
  volume={9},
  pages={224--237},
  year={2023},
  publisher={IEEE}
}
```
