CONFIG = {
    # General Configuration
    "root_dir": "/data",
    "data_annotation_file": {
        "train": "./data_annotations_train.csv",
        "val": "./data_annotations_val.csv"
    },
    "num_classes": 1000,  # Number of prediction classes
    "batch_size": 384, #256,  # Batch size for training
    "epochs": 40,  # Total number of epochs
    "learning_rate": 0.8, #1e-6,  # Initial learning rate
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "augment_prob": 0.25,  # Probability for augmentations (e.g., HorizontalFlip)

    # Hardware Configuration
    "gpus": 1,  # Number of GPUs to use
    "precision": "16-mixed",  # FP16 mixed precision training

    # DataLoader Configuration
    "num_workers": 10,  # Number of DataLoader workers
    "pin_memory": True,  # Use pinned memory for DataLoader

    "check_val_every_n_epoch": 5,

    "lr_finder": False,
    
    # Data paths
    "train_dir": "/data/ILSVRC/Data/CLS-LOC/train",
    "class_index_file": "/data/working/class_map.csv",
    "val_dir": "/data/ILSVRC/Data/CLS-LOC/val",
    "val_mapping_file": "/data/LOC_val_solution.csv"
}
