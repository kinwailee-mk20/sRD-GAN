# Stacked Residual Dropout GAN (sRD-GAN)
Generate perceptually significant instance-diverse synthetic images via regularization. 

The sRD-GAN utilizes a regularization-based strategy in an Image-to-Image (I2I) translation setting to facilitate instance-level diversity. In this study, we show that the stacked dropout regularization in the generator model can induce significant latent-space stochasticity which generates perceptually significant structural dissimilarity in the output space.


## Advantages
1) Diverse patterns on specific instances, irrelevant structures are not affected. (proven working for features of ground glass opacities (GG0s) on COVID-19 infected lung CT images, Chest X-Ray images, and Community Acquired Pneumonia (CAP) on lung CT images).
2) This method is a strategic application of dropout regularization, which means the mechanism can be generalized across most generative models (proven working for GAN, CycleGAN, and One-to-one GAN).
3) Does not require any auxiliary condition to faciliate image diversity. 
4) Computation friendly and does not required any non-trivial modification on the model's architectures. 

## Drawbacks 
1) Diversity are not presented in large perceptual differences, and it focuses only on single instance, for example GGO for COVID-19 CT image. Experiments on colour images are in progress. 
2) Since sRD-GAN is trained using the unsupervised image-to-image framework, and the syntheiszation process does not required any auxiliary condition, the difference in attributes of the diverse images cannot be controlled. 

Paper is available at: https://www.mdpi.com/2306-5354/9/11/698

Dataset is available at https://www.kaggle.com/datasets/lee123456789/largescale-synthetic-covid19-ct-dataset

