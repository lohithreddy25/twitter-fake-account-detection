from model import RGCN
from preprocessing import complete_bot_detection_preprocessing
import torch
from torch import nn
from utils import accuracy, init_weights
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
device = 'cpu'
embedding_size, dropout, lr, weight_decay = 32, 0.2, 1e-3, 5e-3
EPOCHS = 40
# Data loading
des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx = complete_bot_detection_preprocessing(include_all_features=True)

# Data preparation
des_tensor = des_tensor.to(device)
tweets_tensor = tweets_tensor.to(device)
num_prop = num_prop.to(device)
category_prop = category_prop.to(device)
edge_index = edge_index.to(device)
edge_type = edge_type.to(device)
labels = labels.to(device)
train_idx = train_idx.to(device)
val_idx = val_idx.to(device)
test_idx = test_idx.to(device)

# Model initialization
num_prop_size = num_prop.shape[1]
cat_prop_size = category_prop.shape[1]

model = RGCN(
    des_size=des_tensor.shape[1],
    tweet_size=tweets_tensor.shape[1], 
    num_prop_size=num_prop_size,
    cat_prop_size=cat_prop_size,
    embedding_dimension=embedding_size,
    dropout=dropout
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def train(epoch):
    model.train()
    output = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
    loss_train = loss_fn(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    
    with torch.no_grad():
        model.eval()
        val_output = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
        acc_val = accuracy(val_output[val_idx], labels[val_idx])
        model.train()
    
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'acc_val: {:.4f}'.format(acc_val.item()))
    return acc_train, loss_train

def test():
    model.eval()
    with torch.no_grad():
        output = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
        loss_test = loss_fn(output[test_idx], labels[test_idx])
        acc_test = accuracy(output[test_idx], labels[test_idx])

        predictions = output.max(1)[1].to('cpu').detach().numpy()
        label_np = labels.to('cpu').detach().numpy()
        test_idx_np = test_idx.to('cpu').detach().numpy()

        f1 = f1_score(label_np[test_idx_np], predictions[test_idx_np])
        precision = precision_score(label_np[test_idx_np], predictions[test_idx_np])
        recall = recall_score(label_np[test_idx_np], predictions[test_idx_np])

        probs = torch.softmax(output, dim=1)[:, 1].to('cpu').detach().numpy()
        fpr, tpr, _ = roc_curve(label_np[test_idx_np], probs[test_idx_np], pos_label=1)
        auc_score = auc(fpr, tpr)

        print("Test set results:",
                "test_loss= {:.4f}".format(loss_test.item()),
                "test_accuracy= {:.4f}".format(acc_test.item()),
                "precision= {:.4f}".format(precision),
                "recall= {:.4f}".format(recall),
                "f1_score= {:.4f}".format(f1),
                "auc= {:.4f}".format(auc_score))


# Initialize model weights
model.apply(init_weights)

# Training loop
for epoch in range(EPOCHS):
    train(epoch)

# Final test
test()
