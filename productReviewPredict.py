import torch
from torchtext import data
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np
import re

device = torch.device('cpu') # using cps as 'mps' option is not training the model (just bough a new macbook). Need to investigate.
#device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)


def tokenise(sample):
    return sample.split()

def preprocessing(sample):
    return [re.sub("[^a-zA-Z']+", "", word).strip() for word in sample]

stopWords = {'a', 'an', 'the', 'and'}
vec_dim = 300
wordVectors = GloVe(name='6B', dim=vec_dim)

class network(tnn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.dropout_reg = tnn.Dropout(0.3)
        self.hidden_layer = tnn.Linear(in_features=200, out_features=200)
        self.relu = tnn.ReLU()

        self.lstm_rat = tnn.LSTM(input_size=vec_dim, hidden_size=200, num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)
        self.linear_rat = tnn.Linear(in_features=200, out_features=2)

        self.lstm_cat = tnn.LSTM(input_size=vec_dim, hidden_size=200, num_layers=2, batch_first=True, dropout=0.05, bidirectional=True)
        self.linear_cat = tnn.Linear(in_features=200, out_features=5)

    def forward(self, input, length):
        dropout = self.dropout_reg(input)
        lstm_output_rat, (hidden_rat, _) = self.lstm_rat(dropout)
        lstm_output_cat, (hidden_cat, _) = self.lstm_cat(dropout)
        
        hidden_layer_rat = self.hidden_layer(hidden_rat[-1])
        hidden_layer_cat = self.hidden_layer(hidden_cat[-1])

        relu_rat = self.relu(hidden_layer_rat)
        relu_cat = self.relu(hidden_layer_cat)

        return self.linear_rat(relu_rat), self.linear_cat(relu_cat)

class loss(tnn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.loss_function = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        return self.loss_function(ratingOutput, ratingTarget) + self.loss_function(categoryOutput, categoryTarget)

net = network()
lossFunc = loss()

trainValSplit = 0.8
batchSize = 128
epochs = 10
optimiser = toptim.AdamW(net.parameters(), lr=0.001)

def main():
    print(f"Using device: {str(device)}\n")

    # Define text and label fields
    textField = data.Field(lower=True, include_lengths=True, batch_first=True,
                           tokenize=tokenise,
                           preprocessing=preprocessing,
                           stop_words=stopWords)
    labelField = data.Field(sequential=False, use_vocab=False, is_target=True)

    # Create dataset
    dataset = data.TabularDataset('train.json', 'json',
                                 {'reviewText': ('reviewText', textField),
                                  'rating': ('rating', labelField),
                                  'businessCategory': ('businessCategory', labelField)})

    # Build vocab and split data
    textField.build_vocab(dataset, vectors=wordVectors)
    train, validate = dataset.split(split_ratio=trainValSplit)

    # Data loaders
    trainLoader, valLoader = data.BucketIterator.splits((train, validate),
                                                        shuffle=True, batch_size=batchSize,
                                                        sort_key=lambda x: len(x.reviewText),
                                                        sort_within_batch=True)
    
    # Transfer model and loss function to device
    local_net = net.to(device)
    local_lossFunc = lossFunc
    local_optimiser = optimiser

    # Initialize variables for model evaluation
    correctRatingOnlySum = 0
    correctCategoryOnlySum = 0
    bothCorrectSum = 0
    totalSamples = 0

    # Training loop
    for epoch in range(epochs):
        runningLoss = 0
        for i, batch in enumerate(trainLoader):
            inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
            length = batch.reviewText[1].to(device)
            rating = batch.rating.to(device)
            businessCategory = batch.businessCategory.to(device)

            local_optimiser.zero_grad()

            ratingOutput, categoryOutput = local_net(inputs, length)
            loss = local_lossFunc(ratingOutput, categoryOutput, rating, businessCategory)
            
            loss.backward()
            local_optimiser.step()

            runningLoss += loss.item()
            if i % 32 == 31:
                print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {runningLoss / 32:.3f}")
                runningLoss = 0

    # Save the model
    torch.save(local_net.state_dict(), 'savedModel.pth')
    print("\nModel saved to savedModel.pth")

    # Model evaluation
    local_net.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i, batch in enumerate(valLoader):
            inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
            length = batch.reviewText[1].to(device)
            rating = batch.rating.to(device)
            businessCategory = batch.businessCategory.to(device)

            # Forward pass through the network for validation data
            ratingOutputVal, categoryOutputVal = local_net(inputs, length)
            
            # Calculate loss for the validation data
            valLoss = local_lossFunc(ratingOutputVal, categoryOutputVal, rating, businessCategory)
            
            # Calculate performance for the validation data
            valRatingOutputs = torch.argmax(ratingOutputVal, dim=1)
            valCategoryOutputs = torch.argmax(categoryOutputVal, dim=1)

            correctRating = (rating == valRatingOutputs)
            correctCategory = (businessCategory == valCategoryOutputs)
            
            correctRatingOnlySum += torch.sum(correctRating & ~correctCategory).item()
            correctCategoryOnlySum += torch.sum(correctCategory & ~correctRating).item()
            bothCorrectSum += torch.sum(correctRating & correctCategory).item()
            totalSamples += len(batch)

    # Calculate accuracies
    ratingAccuracy = (correctRatingOnlySum + bothCorrectSum) / totalSamples * 100
    categoryAccuracy = (correctCategoryOnlySum + bothCorrectSum) / totalSamples * 100
    bothCorrectAccuracy = bothCorrectSum / totalSamples * 100

    print(f"\nValidation Rating Accuracy: {ratingAccuracy:.2f}%")
    print(f"Validation Category Accuracy: {categoryAccuracy:.2f}%")
    print(f"Both Correct Validation Accuracy: {bothCorrectAccuracy:.2f}%")


if __name__ == '__main__':
    main()
