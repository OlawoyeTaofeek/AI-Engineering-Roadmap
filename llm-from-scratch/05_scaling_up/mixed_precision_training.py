"""Mixed precision training via torch.autocast + GradScaler."""
raise NotImplementedError("TODO: wrap forward pass in autocast(device_type, dtype=torch.bfloat16), "
                           "scale loss with GradScaler before backward() if using fp16")
