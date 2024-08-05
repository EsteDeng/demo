# Source code for "Recurrent Cross-Modality Fusion for ToF Depth Denoising"
The widespread use of Time-of-Flight (ToF) depth cameras in academia and industry is limited by noise, such as Multi-Path-Interference (MPI) and shot noise, which hampers their ability to produce high-quality depth images. 
   Learning-based ToF denoising methods currently in existence often face challenges in delivering satisfactory performance in complex scenes. This is primarily attributed to the impact of multiple reflected signals on the formation of MPI, rendering it challenging to predict MPI directly through spatially-varying convolutions.
   To address this limitation, we propose a novel recurrent architecture that exploits the prior that MPI is decomposable into an additive combination of the geometric information for the neighboring pixels. 
   Our approach employs a gated recurrent unit to estimate a long-distance aggregation process, simplifying the MPI removal, updating depth correction over multiple steps. 
   Additionally, we introduce a global restoration module and a local update module to fuse depth and amplitude features, which improves denoising performance and prevents structural distortions. 
   Our experimental results on both synthetic and real-world datasets demonstrate the superiority of our approach over state-of-the-art methods.
