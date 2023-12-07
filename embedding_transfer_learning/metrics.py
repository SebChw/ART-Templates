import torch
from torchmetrics import Metric


class HitAtKMetric(Metric):
    def __init__(self, top_k=1):
        super().__init__(dist_sync_on_step=False)
        self.top_k = top_k
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.__class__.__name__ = f"HitAtMetric{top_k}"

    def update(self, preds, target):
        # Ensure predictions are 2D (n_queries, n_targets)
        if preds.ndim != 2:
            raise ValueError("Predictions should be 2D (n_queries, n_targets)")

        # Ensure target is 1D
        if target.ndim != 1:
            raise ValueError("Target should be 1D (n_queries)")

        # Increment count by number of queries
        self.count += target.size(0)

        # Get top-k indices for each query
        topk_indices = torch.topk(preds, min(self.top_k,target.size(0)), dim=1).indices
        target_expanded = target.unsqueeze(1).expand_as(topk_indices)
        hits = torch.any(torch.eq(target_expanded, topk_indices), dim=1)

        # Update total hits
        self.total += hits.sum()
    def compute(self):
        # Compute final result
        return self.total.float() / self.count

class MRRMetric(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("total_reciprocal_rank", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        # Ensure predictions are 2D (n_queries, n_targets)
        if preds.ndim != 2:
            raise ValueError("Predictions should be 2D (n_queries, n_targets)")

        # Ensure target is 1D
        if target.ndim != 1:
            raise ValueError("Target should be 1D (n_queries)")

        # Increment count by number of queries
        self.count += target.size(0)

        # Sort predictions to get rank
        sorted_indices = torch.argsort(preds, dim=1, descending=True)

        # Find the rank of the correct prediction
        target_expanded = target.unsqueeze(1).expand_as(sorted_indices)
        ranks = (sorted_indices == target_expanded).nonzero(as_tuple=True)[1] + 1

        # Calculate reciprocal rank, if the target is not in predictions, rank will be zero and reciprocal_rank will be zero
        reciprocal_ranks = 1.0 / ranks.float()

        # Update total reciprocal rank
        self.total_reciprocal_rank += reciprocal_ranks.sum()

    def compute(self):
        # Compute mean of reciprocal ranks
        return self.total_reciprocal_rank / self.count


