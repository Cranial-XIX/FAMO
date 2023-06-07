# FAMO: Fast Adaptive Multitask Optimization

Official implementation of the **FAMO** optimizer for multitask learning. [[Paper]](https://arxiv.org/pdf/2306.03792.pdf)

One of the grand enduring goals of AI is to create generalist agents that can learn multiple different tasks from diverse data via multitask learning (MTL). However, gradient descent (GD) on the average loss across all tasks may yield poor multitask performance due to severe under-optimization of certain tasks. Previous approaches that manipulate task gradients for a more balanced loss decrease require storing and computing all task gradients ($\mathcal{O}(K)$ space and time where K is the number of tasks), limiting their use in large-scale scenarios. In this work, we introduce Fast Adaptive Multitask Optimization (**FAMO**), a dynamic weighting method that decreases task losses in a balanced way using $\mathcal{O}(1)$ space and time. We conduct an extensive set of experiments covering multi-task supervised and reinforcement learning problems. Our results indicate that **FAMO** achieves comparable or superior performance to state-of-the-art gradient manipulation techniques while offering significant improvements in space and computational efficiency.

---

<p align="center"> 
    <img src="https://github.com/Cranial-XIX/FAMO/blob/main/misc/fig.png" width="800">
</p>

**Top left**: The loss landscape, and individual task losses of a toy 2-task learning problem (★ represents the minimum of task losses). **Top right**: the runtime of different MTL methods for 50000 steps. Bottom: the loss trajectories of different MTL methods. ADAM fails in 1 out of 5 runs to reach the Pareto front due to CG. **FAMO** decreases task losses in a balanced way and is the only method matching the $\mathcal{O}(1)$ space/time complexity of ADAM.

---

<p align="center"> 
    <img src="https://github.com/Cranial-XIX/FAMO/blob/main/misc/famo.png" width="800">
</p>


## I. FAMO Example Usage
For the convenience of potential users of **FAMO**, we provide a simple example in ```famo.py``` so that users can easily adapt **FAMO** to their applications. The code requires installation of ```torch```, which can be installed via the setup in the next section. Check the file and simply run
```
python famo.py
```

## II. Image-to-Image Prediction

### Setup environment

Create the conda environment and install torch
```bash
conda create -n mtl python=3.9.7
conda activate mtl
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install the repo:
```bash
git clone https://github.com/Cranial-XIX/FAMO.git
cd FAMO
pip install -e .
```

### Download dataset

We follow the [MTAN](https://github.com/lorenmt/mtan) paper. The datasets could be downloaded from [NYU-v2](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) and [CityScapes](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0). To download the CelebA dataset, please refer to this [link](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg). The dataset should be put under ```experiments/EXP_NAME/dataset/``` folder where ```EXP_NAME``` is chosen from ```nyuv2, cityscapes, celeba```. Note that ```quantum_chemistry``` will download the data automatically.

The file hierarchy should look like
```
FAMO
 └─ experiments
     └─ utils.py                     (for argument parsing)
     └─ nyuv2
         └─ dataset                  (the dataset folder containing the MTL data)
         └─ trainer.py               (the main file to run the training)
         └─ run.sh                   (the command to reproduce FAMO's results)
     └─ cityscapes
         └─ dataset                  (the dataset folder containing the MTL data)
         └─ trainer.py               (the main file to run the training)
         └─ run.sh                   (the command to reproduce FAMO's results)
     └─ quantum_chemistry
         └─ dataset                  (the dataset folder containing the MTL data)
         └─ trainer.py               (the main file to run the training)
         └─ run.sh                   (the command to reproduce FAMO's results)
     └─ celeba
         └─ dataset                  (the dataset folder containing the MTL data)
         └─ trainer.py               (the main file to run the training)
         └─ run.sh                   (the command to reproduce FAMO's results)
 └─ methods
     └─ weight_methods.py            (the different MTL optimizers)
```

### Run experiment

To run experiments, go to the relevant folder with name ```EXP_NAME```
```bash
cd experiment/EXP_NAME
bash run.sh
```
You can check the ```run.sh``` for details about training with **FAMO**.

Following [NashMTL](https://github.com/AvivNavon/nash-mtl), we also support experiment tracking with **[Weights & Biases](https://wandb.ai/site)** with two additional parameters:
```bash
python trainer.py --method=famo --wandb_project=<project-name> --wandb_entity=<entity-name>
```

### MTL methods

We support the following MTL methods with a unified API. To run experiment with MTL method `X` simply run:
```bash
python trainer.py --method=X
```

| Method (code name) | Paper (notes) |
| :---: | :---: |
| FAMO (`famo`) | [Fast Adaptive Multitask Optimization](https://arxiv.org/abs/2306.03792.pdf) |
| Nash-MTL (`nashmtl`) | [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017v1.pdf) |
| CAGrad (`cagrad`) | [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048.pdf) |
| PCGrad (`pcgrad`) | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) |
| IMTL-G (`imtl`) | [Towards Impartial Multi-task Learning](https://openreview.net/forum?id=IMPnRXEWpvr) |
| MGDA (`mgda`) | [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650) |
| DWA (`dwa`) | [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704) |
| Uncertainty weighting (`uw`) | [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115v3.pdf) |
| Linear scalarization (`ls`) | - (equal weighting) |
| Scale-invariant baseline (`scaleinvls`) | - (see Nash-MTL paper for details) |
| Random Loss Weighting (`rlw`) | [A Closer Look at Loss Weighting in Multi-Task Learning](https://arxiv.org/pdf/2111.10603.pdf) |


## III. Multitask Reinforcement Learning (MTRL)
Following [CAGrad](https://github.com/Cranial-XIX/CAGrad), the MTRL experiments are conducted on [Metaworld](https://github.com/rlworkgroup/metaworld) benchmarks. In particular, we follow the [mtrl](https://github.com/facebookresearch/mtrl) codebase and the experiment setup in [this paper](http://proceedings.mlr.press/v139/sodhani21a/sodhani21a.pdf).

1. Install [mtrl](https://github.com/facebookresearch/mtrl) according to the instructions.

2. Git clone [Metaworld](https://github.com/rlworkgroup/metaworld) and change to `d9a75c451a15b0ba39d8b7a8b6d18d883b8655d8` commit (Feb 26, 2021). Install metaworld accordingly.

3. Copy the `mtrl_files` folder under mtrl of this repo to the cloned repo of mtrl. Then
```
cd PATH_TO_MTRL/mtrl_files/ && chmod +x mv.sh && ./mv.sh
```
Then follow the `run.sh` script to run experiments (We are still testing the results but the code should be runnable).


## IV. Citation

This repo is built upon [CAGrad](https://github.com/Cranial-XIX/CAGrad) and [NashMTL](https://github.com/AvivNavon/nash-mtl).
If you find **FAMO** to be useful in your own research, please consider citing the following papers:

```bib
@misc{liu2023famo,
      title={FAMO: Fast Adaptive Multitask Optimization}, 
      author={Bo Liu and Yihao Feng and Peter Stone and Qiang Liu},
      year={2023},
      eprint={2306.03792},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@article{liu2021conflict,
  title={Conflict-Averse Gradient Descent for Multi-task Learning},
  author={Liu, Bo and Liu, Xingchao and Jin, Xiaojie and Stone, Peter and Liu, Qiang},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@article{navon2022multi,
  title={Multi-Task Learning as a Bargaining Game},
  author={Navon, Aviv and Shamsian, Aviv and Achituve, Idan and Maron, Haggai and Kawaguchi, Kenji and Chechik, Gal and Fetaya, Ethan},
  journal={arXiv preprint arXiv:2202.01017},
  year={2022}
}
```
