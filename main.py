from ResnetNetwork import ResNet50LightningModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from util import get_data_loaders

if __name__ == "__main__":
    # data_dir = "path/to/imagenet100k"  # Replace with the actual path
    # batch_size = 64
    # learning_rate = 1e-3
    print("Starting training..")
    torch.cuda.empty_cache()
    max_epochs = 50

    train_loader, val_loader = get_data_loaders()
    model = ResNet50LightningModule(lr_dataloader=train_loader, lr_finder=False, num_classes=1000)
    model.hparams.learning_rate = 0.411  # 0.361

    # logger = TensorBoardLogger("/data/tb_logs", name="resnet50")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="resnet50-{epoch:02d}-{val_acc:.2f}",
        dirpath="/data/checkpoints"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Find learning rate
    # lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader)
    # print(f"Suggested learning rate: {lr_finder.suggestion()}")

    # Update the model with the suggested learning rate
    # model.hparams.learning_rate = lr_finder.suggestion()

    # Use torch_lr_finder to find the learning rate
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    # lr_finder = LRFinder(model, optimizer, criterion=model.criterion, device=device)
    # lr_finder.range_test(train_loader, end_lr=10, num_iter=200)
    # suggested_lr = lr_finder.plot()[-1]  # Plot the loss vs learning rate
    # print(f"Suggested learning rate: {suggested_lr}")
    # lr_finder.reset()

    # Update the model with the suggested learning rate
    # model.hparams.learning_rate = 1e-10

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices="auto",  # 1,  # Adjust devices based on available GPUs
        # strategy="ddp", # if devices > 1
        # strategy= "ddp_notebook", #"ddp_fork" # Use notebook-compatible strategy
        # logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16,
        # accumulate_grad_batches=2,
        # log_every_n_steps=50,  # Log less frequently to reduce overhead
        check_val_every_n_epoch=5  # max_epochs + 1, # skip validation
    )

    # Enable cuDNN benchmark for speed
    torch.backends.cudnn.benchmark = True

    # Train the model
    trainer.fit(model, train_loader, val_loader, ckpt_path="last")

if __name__ == "__main__":
    main()