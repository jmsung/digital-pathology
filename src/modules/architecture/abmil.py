import torch
import torch.nn as nn
import torch.nn.functional as F

class ABMIL(nn.Module):
    def __init__(self, conf, num_organs: int = 1):
        super().__init__()
        self.M = conf.D_feat
        self.L = conf.D_inner
        self.n_class = conf.n_class
        self.ATTENTION_BRANCHES = 1

        # learnable organ embeddings
        self.organ_embed = nn.Embedding(num_organs, self.M)

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES),
        )

        if self.n_class == 1:
            self.classifier = nn.Linear(self.M, 1)
        else:
            self.survival_head       = nn.Linear(self.M, 1)
            self.classification_head = nn.Linear(self.M, 2)

    def forward(self, x, organ_idx: torch.LongTensor = None, return_att=False):
        # x: [1, K, M]  →  H: [K, M]
        H = x.squeeze(0)

        # 1) Add organ embedding (if given) to every instance
        if organ_idx is not None:
            # embedding returns [1, M] if organ_idx is [1], so squeeze to [M]
            e = self.organ_embed(organ_idx).squeeze(0)   # → [M]
            H = H + e                                   # → broadcast to [K, M]

        # 2) Compute attention scores
        A = self.attention(H)   # → [K, 1]
        # flatten into a 1×K row vector, then softmax over the K patches
        A = A.view(1, -1)       # → [1, K]
        A = F.softmax(A, dim=1) # → [1, K]

        # 3) Pool
        Z = torch.mm(A, H)      # → [1, M]

        # 4) Heads
        if self.n_class == 1:
            risk = self.classifier(Z)                # → [1,1]
            return (risk, A) if return_att else risk

        surv = self.survival_head(Z)                  # → [1,1]
        cls  = self.classification_head(Z)            # → [1,2]
        if return_att:
            return (surv, cls), A
        else:
            return surv, cls

    # AUXILIARY METHODS (unchanged)
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        # When return_att=True, forward returns (Y_prob, A); here we ignore A.
        if self.n_class == 2:
            _, Y_pred = self.forward(X, return_att=True)
        else:
            Y_pred = self.forward(X, return_att=False)
        # For classification error, assume Y_pred are logits; compute prediction by argmax.
        pred_class = torch.argmax(Y_pred, dim=1)
        error = 1. - pred_class.eq(Y.long()).cpu().float().mean().item()
        return error, pred_class

    def calculate_objective(self, X, Y):
        Y = Y.float()
        if self.n_class == 2:
            # For dual output, we use the survival output for the objective.
            Y_prob, _ = self.forward(X, return_att=True)
        else:
            Y_prob = self.forward(X, return_att=False)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))
        return neg_log_likelihood, None


class GatedABMIL(nn.Module):
    def __init__(self, conf):
        super(GatedABMIL, self).__init__()
        self.M = conf.D_feat
        self.L = conf.D_inner
        self.n_class = conf.n_class
        self.ATTENTION_BRANCHES = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)  # matrix w (or vector w if ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M, self.n_class),
        )

    def forward(self, x, return_att=False):
        H = x.squeeze(0)
        A_V = self.attention_V(H)  # shape: [K, L]
        A_U = self.attention_U(H)  # shape: [K, L]
        A = self.attention_w(A_V * A_U)  # shape: [K, ATTENTION_BRANCHES]
        A = torch.transpose(A, 1, 0)  # shape: [ATTENTION_BRANCHES, K]
        A = F.softmax(A, dim=1)  # softmax over K
        Z = torch.mm(A, H)  # shape: [ATTENTION_BRANCHES, M]
        Y_prob = self.classifier(Z)  # shape: [ATTENTION_BRANCHES, n_class]
        if return_att:
            return Y_prob, A
        else:
            return Y_prob

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X, return_att=True)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X, return_att=True)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))
        return neg_log_likelihood, A
