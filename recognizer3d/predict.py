import torch
def custom_prediction(y_out: torch.Tensor, threshold: float, index: int) -> torch.Tensor:
    """returns indexes of predicted shapes
    in case of don't recognized returns default index"""
    y_pred = [
        torch.argmax(p, dim=1) if torch.max(p) > threshold else index for p in torch.softmax(y_out, dim=1)
    ]
    return torch.tensor(y_pred)
