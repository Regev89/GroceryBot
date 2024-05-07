import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import pytorch_lightning as pl

from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from builtins import zip

from Server.GrocerySentencesClassifier import GrocerySentenceEmbeddingClassifier


pd.set_option('display.max_colwidth', None)

 # Load data and set labels
shopping_sentences = pd.read_csv('data/shopping_sentences_labled.csv')

X = shopping_sentences.sentence.values
y = shopping_sentences.label.values

X_train, X_val, y_train, y_val =\
    train_test_split(X, y, test_size=0.1, random_state=2020)


train_data = list(zip(X_train, y_train))
val_data = list(zip(X_val, y_val))

# Tokenize data
tokenizer = lambda x: x  # Use identity function since we're working with sentences

# Create datasets
train_dataset = [(tokenizer(x), y) for x, y in train_data]
val_dataset = [(tokenizer(x), y) for x, y in val_data]

#Settings for WandbLogger
lr_monitor = LearningRateMonitor(logging_interval='step')

# Configure ModelCheckpoint to monitor 'val_accuracy' for the best model saving
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',     # Name of the metric to monitor
    dirpath='.\BestModels',    # Directory path where checkpoints will be saved
    filename='{epoch}-{val_acc:.2f}',  # Checkpoint file name format
    save_top_k=1,           # Number of best models to save based on the monitored metric
    mode='max',             # Criterion to evaluate the monitored value ('min' for minimum, 'max' for maximum)
    save_weights_only=True # If True, then only the model’s weights will be saved (`model.state_dict()`), else the full model is saved
)
# Initialize WandbLogger for logging experiments
wandb_logger = WandbLogger(project='ShoppingList',
                           log_model='all')  # Log all new checkpoints during training. This integrates with W&B to not only log metrics but also save model checkpoints automatically to the W&B server.

freeze_epochs=1
max_epochs = 1
batch_size = 64
total_steps = len(train_dataset) // batch_size * max_epochs

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12,persistent_workers=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=12,persistent_workers=True)


def train_model():

    # Initialize the model
    model = GrocerySentenceEmbeddingClassifier('mixedbread-ai/mxbai-embed-large-v1', total_steps=total_steps)
    # After setting up your model and before training
    additional_hyperparams = {
        'freeze_epochs': freeze_epochs,
        'max_epochs': max_epochs,
        'batch_size': batch_size,
        'total_steps': total_steps,
    }

    # Assuming wandb_logger is your WandbLogger instance
    wandb_logger.experiment.config.update(additional_hyperparams)

    # Before starting training, freeze embeddings if required by the model's logic
    model.set_freeze_embedding(True)

    # Log the model with W&B
    wandb_logger.watch(model, log='all', log_freq=100)

    trainer = pl.Trainer(
    max_epochs=freeze_epochs,                # Set the maximum number of training epochs
    enable_progress_bar=True,                # Enable the progress bar during training
    logger=wandb_logger,                     # Integrate W&B for experiment logging. Metrics, system info, and other logs are automatically tracked.
    callbacks=[
        checkpoint_callback,                 # Use the configured ModelCheckpoint callback for model saving based on 'val_accuracy'.
        lr_monitor                           # log learning rates
        ],
    )

    trainer.fit(model, train_loader, val_loader)
    # model.to_onnx(".\BestModels\model.onnx")
    # wandb.save(".\BestModels\model.onnx")

# Necessary for Windows. If you use 'spawn' or 'forkserver' start methods, this is required on Unix as well.
if __name__ == '__main__':
    # Train the model
    torch.set_float32_matmul_precision('medium')
    train_model()

    #Print config file
    # model = SentenceEmbeddingClassifier('mixedbread-ai/mxbai-embed-large-v1', total_steps=total_steps)
    # print(model.get_configoration())
    # config file:
    #'{"arch": "CustomModel", "input_size": 1024, "output_size": 2, "learning_rate": 5e-6}'
