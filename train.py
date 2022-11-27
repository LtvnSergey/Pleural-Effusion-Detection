import numpy as np
from time import time
from utils import load_config, save_history
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import model
import torch
from dice_coef import dice_coefficient

CONFIG_FILE = 'config.yaml'

if __name__ == '__main__':

    # Load configuration
    config = load_config(CONFIG_FILE)

    # Create train dataset
    train_dataset = ImageDataset(dir_input=config['dataset']['dir_processed'],
                                 test_size=config['train']['test_size'],
                                 is_valid=False,
                                 normalization=True)

    # Create validation dataset
    valid_dataset = ImageDataset(dir_input=config['dataset']['dir_processed'],
                                 test_size=config['train']['test_size'],
                                 is_valid=True,
                                 normalization=True)

    # Create dataloader for train dataset
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['train']['batch_size'],
                                  shuffle=True)

    # Create dataloader for validation dataset
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config['evaluate']['batch_size'],
                                  shuffle=False)

    print("[INFO] Data is ready for training")

    # Get model class from models.py
    NetClass = getattr(model, config['model']['architecture'])

    # Define model
    net = NetClass(outSize=(config['dataset']['image_height'],
                            config['dataset']['image_width'])).to(config['train']['device'])

    # Define loss function
    criterion = BCEWithLogitsLoss()

    # Define optimizer
    optimizer = Adam(net.parameters(),
                     lr=config['train']['learning_rate'],
                     betas=(config['train']['beta1'], config['train']['beta2']))

    print("[INFO] Training begin...")

    # Collect training results
    history = {}
    history['train_loss'] = []
    history['train_dice'] = []
    history['valid_loss'] = []
    history['valid_dice'] = []

    best_loss_score = np.Inf
    epochs = config['train']['num_epochs']
    start_time = time()

    # Iterate through epoches
    for epoch in range(epochs):
        # Set network to train mode
        net.train()

        # Initialize train and validation losses and dice values for an epoch
        train_loss_total = 0
        valid_loss_total = 0
        train_dice_total = 0
        valid_dice_total = 0

        # Iterate through dataloader
        for (image, mask) in train_dataloader:
            # Send input to device
            image = image.to(config['train']['device'])
            mask = mask.to(config['train']['device'])

            # Make prediction
            pred = net(image)

            # Calculate loss
            loss = criterion(pred, mask)

            # Zero previous gradient
            optimizer.zero_grad()

            # Perform backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Add loss
            train_loss_total += loss.item()

            # Add dice value
            pred_mask = (torch.sigmoid(pred) > config['evaluate']['threshold'])
            train_dice_total += dice_coefficient(pred_mask, mask)
            break

        # Turn off autogradient
        with torch.no_grad():

            # Set model to evaluation mode
            net.eval()

            # Iterate through dataloader
            for (image, mask) in valid_dataloader:
                # Send input to device
                image = image.to(config['train']['device'])
                mask = mask.to(config['train']['device'])

                # Make prediction
                pred = net(image)

                # Calculate loss
                loss = criterion(pred, mask)

                # Add loss
                valid_loss_total += loss.item()

                # Add dice value
                pred_mask = (torch.sigmoid(pred) > config['evaluate']['threshold'])
                valid_dice_total += dice_coefficient(pred_mask, mask)
                break

        # Calculate average loss and dice per epoch
        avg_train_loss = train_loss_total / (len(train_dataset) // config['train']['batch_size'])
        avg_valid_loss = valid_loss_total / (len(valid_dataset) // config['train']['batch_size'])
        avg_train_dice = train_dice_total / (len(train_dataset) // config['train']['batch_size'])
        avg_valid_dice = valid_dice_total / (len(valid_dataset) // config['train']['batch_size'])

        # Check if loss on current epoch is the best
        if avg_valid_loss < best_loss_score:
            best_loss_score = avg_valid_loss

        # Update training results
        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(avg_valid_loss)
        history['train_dice'].append(avg_train_dice)
        history['valid_dice'].append(avg_valid_dice)

        # Print epoch results
        print('Epoch %3d/%3d, train loss: %5.3f, val loss: %5.3f, train dice: %5.3f, val dice: %5.3f' % \
              (epoch+1, epochs, avg_train_loss, avg_valid_loss, avg_train_dice, avg_valid_dice))

    # Save history file
    save_history(history, output_folder=config['model']['dir_output'])

    # Print time statistics
    end_time = time()
    print()
    print('Time total:     %5.2f sec' % (end_time - start_time))
    print('Time per epoch: %5.2f sec' % ((end_time - start_time) / epochs))
    print()

    print("[INFO] Training complete")