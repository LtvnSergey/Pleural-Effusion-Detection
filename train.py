import numpy as np
from time import time
from utils import load_config, save_history, save_dice_image
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from  model import Net
import torch
from dice_coef import dice_coefficient
import os
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image


CONFIG_FILE = 'config.yaml'


if __name__ == '__main__':

    # Set fixed random number seed
    torch.manual_seed(1234)

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

    # Initialize Tensorboard writer
    writer = SummaryWriter()

    # Define model
    net = Net.to(config['train']['device'])

    # Define loss function
    criterion = smp.losses.DiceLoss(mode='binary')

    # Define optimizer
    optimizer = Adam(net.parameters(),
                     lr=config['train']['learning_rate'],
                     betas=(config['train']['beta1'], config['train']['beta2']))

    # Define learning rate scheduler
    lambda_decay = lambda epoch: config['train']['decay_rate'] ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_decay)

    print("[INFO] Training begin...")

    # Collect training results
    history = {}
    history['train_loss'] = []
    history['train_dice'] = []
    history['valid_loss'] = []
    history['valid_dice'] = []

    # initialize best model scores and epoch
    best_epoch = -999
    best_loss_score = np.Inf

    # Set number of epochs
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

        # Check learning rate
        learning_rate = optimizer.param_groups[0]["lr"]

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
            train_dice_total += dice_coefficient(pred_mask.detach().cpu(), mask.detach().cpu())

        # Turn off autogradient
        with torch.no_grad():
            # Set model to evaluation mode
            net.eval()

            i = 0
            # Iterate through dataloader
            for (image, mask) in valid_dataloader:
                # Send input to device
                image = image.to(config['train']['device'])
                mask = mask.to(config['train']['device'])

                if epoch == 0 and i == 0:
                    writer.add_graph(net, input_to_model=image, verbose=False)

                # Make prediction
                pred = net(image)

                # Calculate loss
                loss = criterion(pred, mask)

                # Add loss
                valid_loss_total += loss.item()

                # Add dice value
                pred_mask = (torch.sigmoid(pred) > config['evaluate']['threshold'])

                # Add examples to Tensorboard
                if i == 155:
                    plot_2d_or_3d_image(image[0], epoch + 1, writer, index=0, tag="Input image")
                    plot_2d_or_3d_image(mask[0], epoch + 1, writer, index=0, tag="Ground truth mask")
                    plot_2d_or_3d_image(pred_mask[0], epoch + 1, writer, index=0, tag="Predicted mask")

                valid_dice_total += dice_coefficient(pred_mask.detach().cpu(), mask.detach().cpu())
                i += 1

        # Step learning rate scheduler
        scheduler.step()

        # Calculate average loss and dice per epoch
        avg_train_loss = train_loss_total / (len(train_dataloader))
        avg_valid_loss = valid_loss_total / (len(valid_dataloader))
        avg_train_dice = train_dice_total / (len(train_dataloader))
        avg_valid_dice = valid_dice_total / (len(valid_dataloader))

        # Add losses and dice metric values to Tensorboard
        writer.add_scalar('Loss train', avg_train_loss, epoch)
        writer.add_scalar('Loss valid', avg_valid_loss, epoch)
        writer.add_scalar('Dice train', avg_train_dice, epoch)
        writer.add_scalar('Dice valid', avg_valid_dice, epoch)

        # Update training results
        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(avg_valid_loss)
        history['train_dice'].append(avg_train_dice)
        history['valid_dice'].append(avg_valid_dice)

        # Check if loss on current epoch is the best
        if avg_valid_loss < best_loss_score:
            best_loss_score = avg_valid_loss
            best_epoch = epoch
            torch.save(net.state_dict(),
                       os.path.join(config['model']['dir_output'],
                                    'model_state.ckpt'))

            # Save history file
            save_history(history, output_folder=config['model']['dir_output'])
            save_dice_image(epoch, history['valid_dice'],
                            output_dir=config['model']['dir_output'])


        # Print epoch results
        print('Epoch %3d/%3d, train loss: %5.3f, val loss: %5.3f, train dice: %5.3f, val dice: %5.3f, lr: %5.7f' % \
              (epoch + 1, epochs, avg_train_loss, avg_valid_loss, avg_train_dice, avg_valid_dice, learning_rate))

    # Save history file
    save_history(history, output_folder=config['model']['dir_output'])
    save_dice_image(epoch, history['valid_dice'],
                    output_dir=config['model']['dir_output'])

    # Print time statistics and number of best epoch
    end_time = time()
    print()
    print('Time total:     %5.2f sec' % (end_time - start_time))
    print('Time per epoch: %5.2f sec' % ((end_time - start_time) / epochs))
    print(f'Best epoch: {best_epoch:.0f}')
    print()

    print("[INFO] Training complete")
    writer.close()