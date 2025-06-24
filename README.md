# Image Captioning

This project implements an Image Captioning system that generates natural language descriptions for input images. The model is trained on a curated image-caption dataset and aims to bridge the gap between computer vision and natural language processing. This architecture combines strong image understanding with the generative capabilities of transformers, producing impressive results even on unseen images.

## Model Architecture
`pre_processing.py`: Responsible for building the vocabulary from the dataset. It processes the captions and prepares token-to-index mappings.  
`resnet_encoder.py`: Encodes input images using the ResNet-50 architecture. The extracted image features are saved in a serialized file named `EncodedImageResnet.pkl`.  
`transformer_decoder.py`: Implements the Transformer-based decoder that takes the encoded image features and generates corresponding captions.  
`train.py`: Consists the main traning loop.  
`main.py`: Integrates all components—preprocessing, encoding, decoding, and training—into a cohesive pipeline for model training.  
`model_test.ipynb`:A Jupyter notebook used to test the trained model on sample images and visualize the generated captions.  

