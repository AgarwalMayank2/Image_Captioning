# Image Captioning

This project implements an Image Captioning system that generates natural language descriptions for input images. The model is trained on a curated image-caption dataset and aims to bridge the gap between computer vision and natural language processing. This architecture combines strong image understanding with the generative capabilities of transformers, producing impressive results even on unseen images.

## Model Architecture
`pre_processing.py`: Building the vocab from the data.

`resnet_encoder.py`: For encoding the given image. In this project I have used Resnet-50. The encoded data will be saved as `EncodedImageResnet.pkl`
`transformer_decoder.py`: For decoding the encoded images and predicting the caption.
`train.py`: Consists the main traning loop.
`main.py`: Intergrates all the files together for tranining purpose.
`model_test.ipynb`: Model testing has been done in this file.

