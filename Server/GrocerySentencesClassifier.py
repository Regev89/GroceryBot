from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, roc_curve, auc , roc_auc_score

def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')

    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')

    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Define the evaluation function for plotting ROC AUC
def evaluate_roc(probs, y_true):
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)

    print(f'\nAUC: {roc_auc:.4f}')
    accuracy = accuracy_score(y_true, np.where(preds > 0.5, 1, 0))
    print(f'Accuracy: {accuracy*100:.2f}%')

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


class GrocerySentenceEmbeddingClassifier(pl.LightningModule):
        def __init__(
                self,
                model_name,
                num_classes=2,
                hidden_size=768,
                dropout=0.3,
                total_steps=None,
                freeze_embeddings=True,
        ):
            super().__init__()

            self.total_steps = total_steps
            # Lists to collect all predictions and labels across validation steps
            self.all_preds = []
            self.all_labels = []
            self.save_hyperparameters()
            self.embedding_model = SentenceTransformer(model_name)
            self.input_size = self.embedding_model.get_sentence_embedding_dimension()
            self.output_size = num_classes
            self.dropout = nn.Dropout(dropout)
            # Increase the initial layer size
            self.fc1 = nn.Linear(self.embedding_model.get_sentence_embedding_dimension(), hidden_size)
            # New intermediate layer with increased capacity
            self.fc_mid1 = nn.Linear(hidden_size, hidden_size // 2)
            self.batch_norm1 = nn.BatchNorm1d(hidden_size // 2)  # Batch normalization
            self.fc_mid2 = nn.Linear(hidden_size // 2, hidden_size // 4)  # Additional layer
            self.layer_norm1 = nn.LayerNorm(hidden_size // 4)
            self.fc2 = nn.Linear(hidden_size // 4, num_classes)

            self.set_freeze_embedding(freeze_embeddings)

        def forward(self, input_ids):
            embeddings = self.embedding_model.encode(input_ids, convert_to_tensor=True)
            x = self.dropout(embeddings)
            x = self.fc1(x)
            x = nn.GELU()(x)  # Keep using GELU here
            x = self.fc_mid1(x)
            x = self.batch_norm1(x)  # Applying batch normalization
            x = nn.GELU()(x)
            x = self.fc_mid2(x)
            x = self.layer_norm1(x)
            x = nn.GELU()(x)  # Consistency with the rest of the model
            x = self.dropout(x)
            x = self.fc2(x)
            return x

        def set_freeze_embedding(self, freeze: bool):
            for param in self.embedding_model.parameters():
                param.requires_grad = not freeze

        def configure_optimizers(self):
            optimizer = AdamW(self.parameters(), lr=5e-6)
            if self.total_steps is not None:
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                            num_training_steps=self.total_steps)
                return [optimizer], [scheduler]
            else:
                return optimizer

        def training_step(self, batch, batch_idx):
            input_ids, labels = batch
            logits = self(input_ids)
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == labels).float() / labels.size(0)
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            input_ids, labels = batch
            logits = self(input_ids)
            loss = F.cross_entropy(logits, labels)
            probs = F.softmax(logits, dim=1)

            # Calculate accuracy
            preds = torch.argmax(probs, dim=1)
            correct = torch.eq(preds, labels).float()
            acc = correct.mean()

            self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

            # Ensure probs and labels are detached and have consistent shapes
            self.all_preds.append(probs.detach())
            self.all_labels.append(labels.detach())

            return {'loss': loss, 'probs': probs.detach(), 'labels': labels.detach()}

        def on_validation_epoch_end(self):
            # Concatenate all predictions and labels from the current epoch
            if self.all_preds and self.all_labels:
                probs = torch.cat(self.all_preds, dim=0).cpu().numpy()
                labels = torch.cat(self.all_labels, dim=0).cpu().numpy()

                # Compute ROC AUC

                roc_auc = roc_auc_score(labels, probs[:, 1])
                self.log('val_roc_auc', roc_auc, prog_bar=True, on_epoch=True)
                print(f'\nValidation ROC AUC: {roc_auc:.4f}')

                # Since self.log already logs the mean values for 'val_loss' and 'val_acc',
                # you can directly access them through self.trainer.logged_metrics (if available)
                val_loss_mean = self.trainer.logged_metrics.get('val_loss', 'Metric Not Found')
                val_acc_mean = self.trainer.logged_metrics.get('val_acc', 'Metric Not Found')
                print(f'Average Validation Loss: {val_loss_mean}')
                print(f'Average Validation Accuracy: {val_acc_mean}')

            # Clear the lists for the next epoch
            self.all_preds.clear()
            self.all_labels.clear()

            # Optional: Store final probabilities and labels for plotting after training
            self.final_probs = probs
            self.final_labels = labels

        def on_train_end(self):
            # Plot ROC curve using the stored final probabilities and labels
            evaluate_roc(self.final_probs, self.final_labels)

        def get_configoration(self):
            return f'{{"arch": "CustomModel","input_size": {self.input_size},"output_size": {self.output_size},"learning_rate": 5e-6}}'
            #return type(self.input_size)


