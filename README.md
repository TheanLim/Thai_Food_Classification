# Fine-Grained Thai Food Image Classification

### Project description
Our project involves implementing a network and training it on a dataset of Thai food images. We will then compare its performance with that of three pre-trained models in PyTorch: DenseNet, ResNet50, and Vision Transformer (base-sized model). We will modify the pre-trained models to work with our dataset and visualize some of their layers to understand how they classify Thai dishes.

### Team 
- Thean Cheat Lim
- Wenlin Fang

### Links to project materials:
- [Dataset](https://huggingface.co/datasets/thean/THFOOD-50)
- [Model](https://huggingface.co/thean/swinv2-tiny-patch4-window8-256-finetuned-THFOOD-50)
- [Presentation Video](https://youtu.be/-N5n4PX8Iqs)
- [Paper](CS_5330_final.pdf)

### Required Packages
- datasets 
- transformers 
- evaluate

#### Python files
- `image_preprocessing.py`: run the main function to calculate the mean and the standard deviations of the training images 
- `custom.py`: run the main function to train the model and see the accuracies on the validation and test dataset
- `resnet_densenet.py`: 
  - Fine-tune ResNet and DenseNet models
  - Usage: python resnet_densenet.py resnet50 model_finetuned_outname 20 /train_data_dir /val_data_dir /test_data_dir
  - Usage: python resnet_densenet.py densenet161 model_finetuned_outname 20 /train_data_dir /val_data_dir /test_data_dir
- `swinv2.py`: 
    - Fine-Tune the SwinV2 (tiny) model using the THFOOD-50 dataset.
    - Usage: python swinv2.py 20 /output_dir    20 means training for 20 epoch
- `visualization.py`: run the main function to visualize the attention maps of the four models
- `pytorch_model_utils.py`: Utility functions for creating/training/testing Pytorch models.
