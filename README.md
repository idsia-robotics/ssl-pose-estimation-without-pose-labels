# Self-Supervised Learning Of Visual Pose Estimation <br> Without Pose Labels By Classifying LED States

*Nicholas Carlotti, Mirko Nava, and Alessandro Giusti*

Dalle Molle Institute for Artificial Intelligence (IDSIA), USI-SUPSI, Lugano (Switzerland)

<!--
WARNING: WHAT IS WRITTEN BELOW IS JUST A PLACEHOLDER FROM PAST WORK WHICH NEEDS TO BE UPDATED, PLEASE IGNORE
### Abstract

We propose a novel self-supervised approach for learning to localize robots equipped with controllable LEDs visually. 
We rely on a few training samples labeled with position ground truth and many training samples in which only the LED state is known, whose collection is cheap. We show that using LED state prediction as a pretext task significantly helps to solve the visual localization end task.
The resulting model does not require knowledge of LED states during inference. <br>
We instantiate the approach to visual relative localization of nano-quadrotors: experimental results show that using our pretext task significantly improves localization accuracy (from 68.3% to 76.2%) and outperforms alternative strategies, such as a supervised baseline, model pre-training, or an autoencoding pretext task. We deploy our model aboard a 27-g Crazyflie nano-drone, running at 21 fps, in a position-tracking task of a peer nano-drone.
Our approach, relying on position labels for only 300 images, yields a mean tracking error of 4.2 cm versus 11.9 cm of a supervised baseline model trained without our pretext task.

<img src="https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_approach.png" width="850" alt="LEDs as Pretext approach" />

Figure 1: *Overview of our approach. A fully convolutional network model is trained to predict the drone position in the current frame by minimizing a loss **L**end defined on a small labeled dataset **T**l (bottom), and the state of the four drone LEDs, by minimizing **L**pretext defined on a large dataset **T**l joined with **T**u (top).*

<br>

Table 1: *Comparison of models, five replicas per row. Pearson Correlation Coefficient ρu and ρv , precision P30 and median of the error D tilde.*

<img src="https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_performance.png" width="900" alt="LEDs as Pretext performance" />

### Bibtex

```properties
@article{nava2024self,
  author={Nava, Mirko and Carlotti, Nicholas and Crupi, Luca and Palossi, Daniele and Giusti, Alessandro},
  journal={IEEE Robotics and Automation Letters}, 
  title={Self-Supervised Learning of Visual Robot Localization Using LED State Prediction as a Pretext Task}, 
  year={2024},
  volume={9},
  number={4},
  pages={3363-3370},
  doi={10.1109/LRA.2024.3365973},
}
```

### Video

[![Self-Supervised Learning of Visual Robot Localization Using Prediction of LEDs States as a Pretext Task](https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_video_preview.gif)](https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_video.mp4?raw=true)

### Code

The codebase for the approach is avaliable [here](https://github.com/idsia-robotics/leds-as-pretext/tree/main/code).

##### Requirements

- Python                       3.8.0
- h5py                         3.8.0
- numpy                        1.23.5
- scipy                        1.10.1
- torch                        1.13.1
- torchinfo                    1.8.0
- torchvision                  0.15.2
- tensorboard                  2.12.3
- torch-tb-profiler            0.4.1
- scikit-image                 0.21.0
- scikit-learn                 1.2.2

##### LED State Prediction Pretext (LED-P)
Our approach trains on samples divided into labeled and unlabeled ones. The training script can be invoked as follows, to train model `model_name` for `epochs` epochs with batch size `batch_size` with a lambda for the position loss (or 1 - lambda for the pretext loss) `lambda_loss_pos`, considering the fraction `fraction_labeled_samples` of the dataset to have labeled samples.
```bash
python train_led_pretext.py -e <epochs> -bs <batch_size> -wpos <lambda_loss_pos> -fpos <fraction_labeled_samples> -n <model_name>
```

__Note:__ The script will store inside the model's folder the tensorboard log containing all training and testing set metrics.

##### Efficient Deep Neural Networks (EDNN)
The procedure to train this model is essentially the same as for __LED-P__. The only difference is that the script for this model supports fine tuning of an available checkpoint.
For example, given a checkpoint from a model with name `pre_trained`, fine tuning a checkpoint can be done as follows:

```bash
python train_ednn.py --pre-trained-name <pre_trained>/checkpoint.tar
```

__Note:__ The script looks for pre-trained models in the `models/` folder.

##### Autoencoding Pretext (AE-P)
The autoencoder training script provides two ways of training the model: *autoencoder mode* and *position mode*.  
In autoencoder mode, the model is trained to learn a hidden representation for the input dataset. No validation is done during this phase. This mode is the default one.  
The position mode is enabled by adding the `--position-mode` flag to the script invocation command.
The common use case for this mode is that of starting from a pre-trained hidden representation and tune it to solve the position task.
The bottleneck size is chosen by the parameter `bottleneck`; the position mode training is started by running:
```bash
python train_autoencoder.py --position-mode --bottleneck-size <bottleneck> --pre-trained-name <hidden_repr>/checkpoint.tar
```

##### Contrastive Language–Image Pre-training (CLIP)
This script requires CLIP to be installed. Refer to the [official installation instructions](https://github.com/openai/CLIP#usage) to obtain it.  
The only parameter of interest for using this script is the `--clip-base-folder` flag, which tells the script where to download the CLIP checkpoint. The path is relative to the shell's current folder. To run the training execute the following:
```bash
python train_clip_head.py --clip-base-folder <./clip>
```
--!>
