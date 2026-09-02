"""Top-k router: a linear layer producing per-expert logits, softmax, then
select the top-k experts per token."""
raise NotImplementedError("TODO: nn.Linear(emb_dim, num_experts), torch.topk over the softmax output")
