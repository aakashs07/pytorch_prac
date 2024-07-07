import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import Dataset


def split_dataset(df):
    from sklearn.model_selection import train_test_split

    # Separate the rows by outcome label
    df_minority = df[df['Outcome'] == 1]
    df_majority = df[df['Outcome'] == 0]

    # Randomly select 80% of rows with label 1 for the training set
    train_minority = df_minority.sample(frac=0.8, random_state=42)

    # Select an equal number of rows with label 0 for the training set
    train_majority = df_majority.sample(n=len(train_minority), random_state=42)

    # Combine the selected rows to create the training set
    training_df = pd.concat([train_minority, train_majority])

    # Get the remaining rows
    remaining_minority = df_minority.drop(train_minority.index)
    remaining_majority = df_majority.drop(train_majority.index)

    # Combine the remaining rows
    remaining_df = pd.concat([remaining_minority, remaining_majority])

    # Split the remaining data into equal halves for validation and test sets
    validation_df, testing_df = train_test_split(remaining_df, test_size=0.5, random_state=42)

    return training_df, validation_df, testing_df


class DiabetesDataset(Dataset):
    def __init__(self, data_df):
        super().__init__()
        # Convert training data to a NumPy array and assign to self.train_data
        self.data = data_df.to_numpy()

    # Implement __len__ to return the number of data samples
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        # Assign last data column to label
        label = self.data[idx, -1]
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return features, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define the three linear layers
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        # Pass x through linear layers adding activations
        x = nn.functional.elu(self.fc1(x))
        x = nn.functional.elu(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        return x


class TrainModelPipeline:
    def __init__(self, model, task_type: str = "binary", optimization_type: str = "SGD",
                 lr: float = 0.001, momentum: float = 0.0, weight_decay: float = 0):
        """
        Initializes the loss function and optimizer based on the provided inputs.
        :param model: The model object of the class Net to be trained.
        :param task_type: The type of loss function to use ('binary', 'mse', etc.)
        :param optimization_type: The type of optimizer to use ('SGD', 'Adam', etc.)
        :param lr: Learning rate for the optimizer.
        :param momentum: Momentum for optimizers to optimize training.
        :param weight_decay: Weight decay (L2 penalty) for the optimizer.
        """
        self.model = model
        self.task_type = "binary"  # this might change later

        if task_type == "binary":
            self.task_type = "binary"
            self.criterion = nn.BCELoss()
        elif task_type == "mse":
            self.criterion = nn.MSELoss()

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        if optimization_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                       weight_decay=self.weight_decay)
        elif optimization_type == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model_trained = False

    def evaluate(self, eval_data):
        """
        Validates the model using the validation data.
        :param eval_data: DataLoader object for the validation dataset.
        :return: Loss and Accuracy of the model on the validation dataset.
        """
        # set the model in evaluation mode
        self.model.eval()
        # Directly access the entire dataset from the DataLoader
        eval_dataset = eval_data.dataset

        # Prepare inputs and targets
        inputs = torch.stack([eval_dataset[i][0] for i in range(len(eval_dataset))])
        targets = torch.stack([eval_dataset[i][1] for i in range(len(eval_dataset))])

        eval_acc = Accuracy(task="binary")

        # Set up accuracy metric
        if self.task_type == "binary" or not self.task_type:
            eval_acc = Accuracy(task="binary")
        elif self.task_type == "multiclass":
            eval_acc = Accuracy(task="multiclass")

        with torch.no_grad():
            outputs = self.model(inputs)
            eval_loss = self.criterion(outputs, targets.view(-1, 1)).item()

            # Calculate overall accuracy
            eval_predictions = (outputs >= 0.5).float()
            eval_accuracy = eval_acc(eval_predictions, targets.view(-1, 1)).item()

        if self.model_trained:
            print(f'Evaluation step')
            print(f'Sample size: {len(eval_dataset)}')
            print(f'Loss: {eval_loss}, Accuracy: {eval_accuracy:.4f}%')

        # Set the model back to training mode
        self.model.train()
        return eval_loss, eval_accuracy

    def train(self, training_data, validation_data, num_epochs: int = 1000, plot_loss_curve: bool = True):
        """
        Trains the model using the provided training dataset.
        :param training_data: DataLoader object for the training dataset.
        :param validation_data: DataLoader object for the validation dataset.
        :param num_epochs: Number of epochs to train the model.
        :param plot_loss_curve: Whether to plot the loss curve.
        :return:
        """
        training_epoch_losses = []
        training_epoch_accuracies = []
        validation_epoch_losses = []
        validation_epoch_accuracies = []

        for epoch in range(num_epochs):
            running_loss = 0
            batch_acc = Accuracy(task="binary")
            for features, labels in training_data:
                self.optimizer.zero_grad()
                # forward pass
                outputs = self.model(features)
                # convert outputs to float
                outputs = outputs.float()
                # compute predictions
                predictions = (outputs >= 0.5).float()
                # compute loss
                batch_loss = self.criterion(outputs, labels.view(-1, 1))
                # compute gradients with backward pass
                batch_loss.backward()
                # update weights
                self.optimizer.step()
                # compute for cumulative loss in an epoch
                running_loss += batch_loss.item()
                # compute accuracy
                batch_acc(predictions, labels.view(-1, 1))

            epoch_loss = running_loss / len(training_data)
            epoch_accuracy = batch_acc.compute()

            if epoch % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, '
                      f'Accuracy: {epoch_accuracy:.4f}%')

            # Store training loss and accuracy stats
            training_epoch_losses.append(epoch_loss)
            training_epoch_accuracies.append(epoch_accuracy)

            val_loss, val_accuracy = self.evaluate(validation_data)
            validation_epoch_losses.append(val_loss)
            validation_epoch_accuracies.append(val_accuracy)

        if plot_loss_curve:
            from matplotlib import pyplot as plt
            plt.plot(training_epoch_losses, label='train_loss')
            plt.plot(validation_epoch_losses, label='val_loss')
            plt.legend()
            plt.show()

        self.model_trained = True
        print("Model trained successfully!")
