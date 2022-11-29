import segmentation_models_pytorch as smp

# Model class
Net = smp.Unet(
                            encoder_name="resnet18",        # encoder
                            encoder_weights="imagenet",     # `imagenet` pre-trained weights for encoder initialization
                            in_channels=1,                  # model input channels
                            classes=1,                      # model output channels
                        )
