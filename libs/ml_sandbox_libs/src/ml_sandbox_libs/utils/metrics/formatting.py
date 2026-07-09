import torch
from torchmetrics import Metric


def format_metrics_dict(metrics_dict: dict[str, float | torch.Tensor | Metric]) -> str:
    """Format a metric dictionary as a compact string.

    Args:
        metrics_dict: Dictionary mapping metric names to scalar values, tensors, or Metric objects.

    Returns:
        formatted string
    """
    formatted_metrics: list[str] = []
    for key, value in metrics_dict.items():
        if isinstance(value, Metric):
            scalar_value: float | torch.Tensor = value.compute()
        else:
            scalar_value = value
        if isinstance(scalar_value, torch.Tensor):
            scalar_value = scalar_value.item()
        formatted_metrics.append(f"{key}: {float(scalar_value):.4f}")
    return " ".join(formatted_metrics)
