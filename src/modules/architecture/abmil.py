import torch
import torch.nn as nn
import torch.nn.functional as F

class ABMIL(nn.Module):
    def __init__(self, conf):
        super(ABMIL, self).__init__()
        self.M = conf.D_feat
        self.L = conf.D_inner
        self.n_class = conf.n_class
        self.ATTENTION_BRANCHES = 1

        # Attention branch remains the same
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES)  # vector w if ATTENTION_BRANCHES==1
        )
        
        # Depending on n_class, we define the output heads.
        if self.n_class == 1:
            # Single output (survival risk only)
            self.classifier = nn.Sequential(
                nn.Linear(self.M, 1)
            )
        elif self.n_class == 2:
            # Dual outputs: one head for survival risk and one head for classification (2 logits)
            self.survival_head = nn.Linear(self.M, 1)
            self.classification_head = nn.Linear(self.M, 2)
        else:
            raise ValueError("n_class must be either 1 or 2")

    def forward(self, x, return_att=False):
        # Expecting x shape to be [1, K, M]; squeeze the first dim to get [K, M]
        H = x.squeeze(0)  # H: [K, M]
        A = self.attention(H)  # [K, ATTENTION_BRANCHES]
        A = torch.transpose(A, 1, 0)  # [ATTENTION_BRANCHES, K]
        A = F.softmax(A, dim=1)  # Softmax over K
        Z = torch.mm(A, H)  # [ATTENTION_BRANCHES, M]

        if self.n_class == 1:
            # Return survival risk only
            risk = self.classifier(Z)  # [ATTENTION_BRANCHES, 1]
            if return_att:
                return risk, A
            else:
                return risk
        elif self.n_class == 2:
            # Compute survival and classification outputs
            survival_pred = self.survival_head(Z)       # [ATTENTION_BRANCHES, 1]
            class_pred = self.classification_head(Z)      # [ATTENTION_BRANCHES, 2]
            if return_att:
                return (survival_pred, class_pred), A
            else:
                return survival_pred, class_pred

    # AUXILIARY METHODS (unchanged)
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X, return_att=True)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X, return_att=True)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))
        return neg_log_likelihood, A
