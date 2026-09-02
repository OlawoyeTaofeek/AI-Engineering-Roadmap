"""Fine-tune a pretrained GPTModel for classification by replacing the output
head with a small linear classifier on top of the final token's hidden state.

TODO: freeze most of the backbone, add nn.Linear(emb_dim, num_classes), train
on a labeled dataset with cross-entropy over class labels instead of vocab.
"""
raise NotImplementedError
