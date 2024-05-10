+++
title = "Transformers Implementation and Training with Pytorch and Lightning"
description = "A super comprehensive and detailed guide to understanding the Transformer model and implementing it using PyTorch and Pytorch Lightning."
date = 2024-05-10

[taxonomies]
tags = ["ML Research Paper Implementation", "Lightning"]
+++

### Introduction

The Transformer, with its parallel processing capabilities, allowed for more efficient and scalable models, making it easier to train them on large datasets. It also demonstrated superior performance in several NLP tasks, such as sentiment analysis and text generation tasks. The Transformer model has since become the foundation for many state-of-the-art NLP models, such as BERT, GPT-2, and T5.

### Main Components of the Transformer Architecture

![Transformer Architecture](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers-dark.svg)

The Transformer Architecture Consists of 2 main Components ->

- Encoder -  The encoder receives an input and builds an Embedding of it's Features. This means that the model learns to understand the association of words or sequence.

- Decoder - The decoder uses the encoder’s output embeddings along with other inputs to generate a target sequence.

### Input Embeddings

Whenever we use a dataset , and try to Train a Model on it , we always convert explicitly or implicitly to a representation which the model can interpret / understand and then reconvert it into a representation we understand , the Function of Input Embedding Block in the Transformer Architecture is just that only. In the Orignal Paper the Authors used , the Input Block with an Embedding Dimension of 512.To prevent the input Embeddings from being extremely small , we normalize them by Multiplying the by root of EmbeddingDimension

#### Implementation of the Input Embedding Block

```python
    import torch
    import torch.nn as nn
    import math
        
    class InputEmbeddingBlock(nn.Module):

        def __init__(self,embedding_dim:int,vocab_size:int):
                
            super().__init__()
            self.embedding_dim = embedding_dim # Reffered in the paper as d_model, (Size == 512)
            self.vocab_size = vocab_size ## Size of the Vocabulary of the input 
            self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)

        def forward(self,x):
            return self.embedding_dim(x)*math.sqrt(self.embedding_dim) ## This is done to help Prevent the Size of Input Embedding being diminished

```

### Positional Encoding

The input now is converted into input Embeddings of Dimension 512 , but unless we don't provide a signal for the encoder on the relative or absolute position of the tokens in the sequence the Model can't learn the corresponding association
to get around that probelm the authors have provided a positional Encoding for a token based on if its index is an even number or an odd number , these encodings are computed only once and in the paper are not learned by the model.

#### Implementation of the Positional Encoding Block

```python
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Transformer models.

    Args:
        embedding_dim (int): The dimension of the input embeddings.
        sequence_len (int): The length of the input sequence.
        dropout (float): The dropout probability.
    """
    def __init__(self, embedding_dim: int, sequence_len: int, dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sequence_len = sequence_len
        self.dropout = nn.Dropout(dropout)

        # Creating a matrix of size (sequence_len,embedding_dim)
        positional_encoding = torch.zeros(sequence_len, self.embedding_dim)

        # Create a vector of shape (sequence_len,1)
        position = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(dim=1)
        division_term = torch.exp(torch.arange(0, embedding_dim, 2)).float() * (-torch.log(10000.0) / embedding_dim)

        # Apply the sin formula to the even positions and cosine formula to the odd positions
        positional_encoding[:, 0::2] = torch.sin(position * division_term) # Every two Terms even -> 0 -> 2 -> 4 
        positional_encoding[:, 1::2] = torch.cos(position * division_term) # Every two Terms odd -> 1 -> 3 -> 5
        
        positional_encoding = positional_encoding.unsqueeze(dim=0)
        self.register_buffer('positional_encoding',positional_encoding)
        
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)

```

### Layer Normalization

These are the Add and Norm Layer in the Architecture these help scaling input tensor with Layer the LayerNormalization Block is already implemented in Pytorch

```python

import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    Applies layer normalization to the input tensor.

    Args:
        eps (float, optional): A value added to the denominator for numerical stability. Default is 1e-5.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = eps

    def forward(self, x):
        """
        Applies layer normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return nn.LayerNorm(x, eps=self.eps)


```

### Simple MLP or FeedForward Block

Really Simple MLP consisting of 2 Linear Layers with the ReLU activation function b/w them also using Dropout Layer for overfitting prevention.

### Implemention of the FeedForward Block

```python
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    """
    A feed-forward block in the Transformer model.

    Args:
        embedding_dim (int): The dimensionality of the input embeddings.
        feed_forward_dim (int): The dimensionality of the hidden layer in the feed-forward network.
        dropout (float): The dropout probability.

    Attributes:
        linear_1 (nn.Linear): The first linear layer.
        dropout (nn.Dropout): The dropout layer.
        linear_2 (nn.Linear): The second linear layer.
    """

    def __init__(self, embedding_dim, feed_forward_dim, dropout) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, feed_forward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(feed_forward_dim, embedding_dim)
        
    def forward(self, x):
        """
        Forward pass of the feed-forward block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.linear_1(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x 
```