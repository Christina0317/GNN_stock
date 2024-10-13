# GNN-GRU Time Series Prediction

This project implements a neural network model that combines GNN and GRU for predicting time series data. The model captures both spatial and temporal dependencies in the data by leveraging the GNN for spatial relationships and the GRU for temporal sequence modeling.

### Key Components:
1. **GRU (Gated Recurrent Unit)**: The GRU is used to capture the temporal dependencies across the nodes in the graph, allowing the model to understand sequential relationships over time.
2. **GCN (Graph Convolutional Network)**: The GCN leverages the adjacency matrix or edge list (based on node pairwise differences `f1` and element-wise multiplications `f2`) to propagate information across connected nodes in the graph, learning spatial dependencies.
3. **Attention Mechanism**: The model computes a learned adjacency matrix (`A_hat`) based on node features, capturing dynamic relationships between nodes.
4. **Edge Weight Prediction**: Edge weights are computed between nodes based on their learned features, and these are used in the GCN to influence prediction.
5. **Leaky ReLU and Dropout**: Non-linearities and regularization are added to the model using the LeakyReLU activation function and Dropout.

## Model Architecture

- **GRU Layer**: This layer processes the input sequence of node features and outputs the hidden state for each time step.
- **Learned Adjacency Matrix**: The model dynamically constructs an adjacency matrix based on node pairwise differences (`f1`) and element-wise multiplications (`f2`).
- **Graph Convolution**: The GCN applies graph convolutions on the node features, weighted by the learned adjacency matrix, to propagate information between nodes in the graph.
- **Linear Layers**: The final layers map the hidden representations to the desired output dimensions, and Batch Normalization is used for faster convergence.


