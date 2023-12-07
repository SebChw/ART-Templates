import torch.nn as nn
import torch
from lightning import seed_everything

from metrics import MRRMetric


class ApproximateMRR(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def __call__(self, cosine_similarities, labels: torch.LongTensor):
        #sx targeet scores normalized rowwise 0 to 1
        min_values = cosine_similarities.min(dim=1, keepdim=True).values
        max_values = cosine_similarities.max(dim=1, keepdim=True).values
        scores_normalized = (cosine_similarities - min_values)/(max_values - min_values)
        sx = scores_normalized[torch.arange(0, scores_normalized.shape[0]), labels]
        #sxy = sx - sy
        sxy = sx.view(-1,1) - scores_normalized
        #sigmoid
        sxy_less_zero_approx = torch.sigmoid(-self.alpha*sxy)
        position_approx = 1 + torch.sum(sxy_less_zero_approx, dim=1) - sxy_less_zero_approx.diagonal()
        mrr = 1/position_approx
        return 1-torch.mean(mrr)





if __name__ == "__main__":
    seed_everything(3553768778)
    n = 10
    # similarities = torch.linspace(-1, 1, n)
    # similarities = torch.stack([similarities] * n)
    similarities = torch.randn(n, n)
    labels = torch.arange(0, n, dtype=torch.long)
    criterion = ApproximateMRR(100.0)
    loss = criterion(similarities, labels)
    print(loss)
    metric = MRRMetric()
    metric.update(similarities, labels)
    print(metric.compute())
