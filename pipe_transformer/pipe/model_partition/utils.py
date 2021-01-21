
def count_parameters(model, b_is_required_grad=True):
    if b_is_required_grad:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        params = sum(p.numel() for p in model.parameters())
    return params / 1000000


