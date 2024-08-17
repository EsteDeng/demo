# Source code for "Recurrent Cross-Modality Fusion for ToF Depth Denoising"
The widespread use of Time-of-Flight (ToF) depth cameras in academia and industry is limited by noise, such as Multi-Path-Interference (MPI) and shot noise, which hampers their ability to produce high-quality depth images. 
   Learning-based ToF denoising methods currently in existence often face challenges in delivering satisfactory performance in complex scenes. This is primarily attributed to the impact of multiple reflected signals on the formation of MPI, rendering it challenging to predict MPI directly through spatially-varying convolutions.
   To address this limitation, we propose a novel recurrent architecture that exploits the prior that MPI is decomposable into an additive combination of the geometric information for the neighboring pixels. 
   Our approach employs a gated recurrent unit to estimate a long-distance aggregation process, simplifying the MPI removal, updating depth correction over multiple steps. 
   Additionally, we introduce a global restoration module and a local update module to fuse depth and amplitude features, which improves denoising performance and prevents structural distortions. 
   Our experimental results on both synthetic and real-world datasets demonstrate the superiority of our approach over state-of-the-art methods.

## How to use the code

code running environment
```
tensorflow-gpu==1.12.0 
```

the project starts with the 'start.py'. Through this file, you can select different models, data sets and loss functions for training, and you can switch between train, eval and output modes by adjusting parameters
The parameters available are as follows

```
Arg
├───modelName		# Select the model required during training
│   ├───sample_pyramid_add_kpn                 # SHARP-Net
|   |───recurrent_corr_fusion                  # Ours
│   ├───dear_kpn_no_rgb                        # ToF-KPN
│   └───dear_kpn_no_rgb_DeepToF                # DeepToF
├───trainingSet		# Select the dataset required during training
│   ├───tof_FT3       # ToF-FlyingThings3D dataset (480, 640)
│   ├───HAMMER          # HAMMER dataset (424, 512)
│   └───CornellBox       # CornellBox dataset (640, 640)
├───flagMode		# Select the running mode of the code
│   ├───train                 # train model
│   ├───eval_ED               # evaluate model in test sets
│   ├───eval_TD               # evaluate model in training sets
│   └───output                # output depth prediction, offsets, weight
├───gpuNumber		# The number of GPU used in training
├───addGradient		# weather add the gradient loss function
├───decayEpoch		# after n epochs, decay the learning rate
├───lossType		# Select the loss function in training
│   ├───mean_l1               # the mean of L1 loss between input and gt
│   └───mean_l2               # the mean of L2 loss
└───lossMask	    # Select the loss mask to be used during training
    ├───gt_msk                # Non-zero region in groundtruth
    ├───depth_kinect_msk      # Non-zero region in depth input
    └───depth_kinect_with_gt_msk      # gt_msk with depth_kinect_msk
```
For example

```
python start.py -b 2 -s 200000 -m recurrent_corr_fusion -p size384 -k depth_kinect_with_gt_msk -l 0.0004 -t tof_FT3 -i 480 640 -o mean_l1 --addGradient sobel_gradient -g 4 -e 1200
```
