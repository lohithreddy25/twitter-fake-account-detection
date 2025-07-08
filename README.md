# Twitter Fake Account Detection using R-GCN

This project implements a fake account detection system using Relational Graph Convolutional Networks (R-GCNs) on the Cresci-2015 dataset.

##  Dataset

We use the [Cresci-2015 dataset] which includes labeled Twitter user accounts categorized into genuine and fake (bot) users. It contains:

- Tweets and user metadata
- Relationships such as followers, friends, retweets, etc.

##  Objective

To classify Twitter accounts as **real** or **fake** based on their behavioral and relational graph structure.

##  Tech Stack

- Python 3.x
- PyTorch & DGL (Deep Graph Library)
- NetworkX
- scikit-learn
- pandas, numpy

##  Files

- `preprocessing.py`: Parses and processes raw dataset into a heterogeneous graph for R-GCN input.
- `model.py`: Defines a 2-layer R-GCN model.
- `train.py`: Trains the model and evaluates performance.

##  Preprocessing

Run the script to generate graph and features:

```bash
python preprocessing.py
