import torch
from torch import nn
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F

class RGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, embedding_dimension=128, dropout=0.3):
        super(RGCN, self).__init__()
        self.dropout = dropout

        # Feature processing layers
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension/4)),
            nn.LeakyReLU()
        )

        # Input processing layer
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        # 2-layer RGCN
        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

        # Output layers
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        # Feature processing
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        # Input processing
        x = self.linear_relu_input(x)

        # 2-layer RGCN with dropout
        x = self.rgcn(x, edge_index, edge_type)  # RGCN Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)  # RGCN Layer 2

        # Output processing
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)  # Final output: 2 classes

        return x

