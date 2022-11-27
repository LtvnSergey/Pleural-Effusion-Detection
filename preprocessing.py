from utils import preprocess_dataset, load_config

CONFIG_FILE = 'config.yaml'

if __name__ == '__main__':
    # Load configuration
    config = load_config(CONFIG_FILE)

    # Preprocess raw data
    preprocess_dataset(config['dataset']['dir_images'],
                       config['dataset']['dir_masks'],
                       config['dataset']['dir_processed'],
                       save=True, output=False)
