#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Contrastive loss module.

This module implements a loss combining a contrastive 
loss term with a regularization on the input
"""

__all__ = ["ContrastiveLoss", "contrastive_loss"]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch
import math

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class ContrastiveLoss(torch.nn.Module):
    """Compute a loss function that combines a contrastive spectral objective 
    with an L2 regularization term on the learned representations.
    """

    def __init__(self, 
                mode: str = "l2", 
                reg: float = 1e-5):
        """Compute a contrastive loss combining spectral objective with an L2 
        regularization term on the learned representations.

        Parameters
        ----------
        mode : str, optional
            Contrastive loss type, by default "l2". 
            Possible modes are:
                - "l2": L2 decorrelation loss (closely related to the VAMP-2 score)
                - "kl_DV": KL-based loss via Donsker-Varadhan bound
                - "kl_NWJ": KL-based loss via Nguyen-Wainwright-Jordan bound
        reg : float, optional
            Regularization coefficient (default: 1e-5).
        """
        super().__init__()
        self.mode = mode
        self.reg = reg

    def forward(
        self, 
        inputs: torch.Tensor, 
        lagged: torch.Tensor,
        traj_id: torch.Tensor = None,
        remove_average: bool = True,
    ) -> torch.Tensor:
        """
        Compute the regularized spectral loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Representations at time t.
        lagged : torch.Tensor
            Representations at time t+τ.
        remove_average : bool, optional
            Whether to subtract the mean from the input representations
            before computing time-correlation matrices.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        return contrastive_loss(
            inputs,
            lagged,
            traj_id=traj_id,
            reg=self.reg,
            mode=self.mode,
            remove_average=remove_average,
        )

    def noreg(
        self, 
        inputs: torch.Tensor, 
        lagged: torch.Tensor,
        traj_id: torch.Tensor = None,
        remove_average=True
    ) -> torch.Tensor:
        """
        Compute the contrastive term only (without regularization).
        """
        return contrastive_loss(
            inputs,
            lagged,
            traj_id=traj_id,
            reg=0.0,
            mode=self.mode,
            remove_average=remove_average,
        )
    

def contrastive_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    traj_id: torch.Tensor = None,
    mode: str = "l2",
    reg: float = 1e-5,
    remove_average: bool = True,
) -> torch.Tensor:
    """
    Compute the contrastive loss.

    If traj_id is None, all off-diagonal terms are used as negative pairs.
    If traj_id is provided, only off-diagonal terms with different traj_id
    are used as negative pairs.

    Parameters
    ----------
    x, y : torch.Tensor
        Input tensors of shape (n_samples, n_features), representing
        configurations at time t and t+tau.

    traj_id : torch.Tensor, optional
        Trajectory/walker ID for each sample. If provided, negative pairs are
        constructed only between different trajectories.

    mode : str, optional
        Contrastive loss type: {"l2", "kl_DV", "kl_NWJ"}, by default "l2".
    reg : float, optional
        Regularization strength, by default 1e-5.
    remove_average : bool, optional
        Whether to subtract the (weighted) mean from the input representations
        before computing time-correlation matrices, by default True.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")
    if x.ndim != 2:
        raise ValueError("Inputs must be 2D tensors.")

    npts, dim = x.shape
    if npts < 2:
        raise ValueError("Contrastive loss requires at least 2 samples.")

    # remove mean
    if remove_average:
        x = x - torch.mean(x, dim=0)
        y = y - torch.mean(y, dim=0)

    # similarity matrix D_ij = <x_i, y_j>
    sim_mat = torch.matmul(x, y.T)

    # positive term: diagonal matched pairs
    pos_term = torch.mean(x * y) * dim

    # negative mask
    eye = torch.eye(npts, dtype=torch.bool, device=x.device)

    if traj_id is None:
        # default: all off-diagonal terms are negatives
        neg_mask = ~eye
    else:
        traj_id = torch.as_tensor(traj_id, device=x.device)

        if traj_id.ndim != 1:
            traj_id = traj_id.view(-1)

        if traj_id.shape[0] != npts:
            raise ValueError(
                f"traj_id has length {traj_id.shape[0]}, but batch size is {npts}."
            )

        same_traj = traj_id[:, None] == traj_id[None, :]
        neg_mask = (~same_traj) & (~eye)

    if not torch.any(neg_mask):
        raise ValueError(
            "No valid negative pairs found. "
            "If traj_id is provided, make sure each batch contains multiple trajectories."
        )

    sim_neg = sim_mat[neg_mask]
    n_neg = sim_neg.numel()

    # contrastive loss term
    if mode == "l2":
        diag = 2.0 * pos_term
        neg_term = (sim_neg ** 2).mean()
        loss = neg_term - diag

    elif mode == "kl_DV":
        log_term = torch.logsumexp(sim_neg, dim=0)
        log_term = log_term - math.log(n_neg)
        loss = log_term - pos_term

    elif mode == "kl_NWJ":
        exp_term = (sim_neg - 1.0).exp().mean()
        loss = exp_term - pos_term

    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported: l2, kl_DV, kl_NWJ")

    # regularization
    if reg > 0.0:
        x_norm2 = torch.linalg.matrix_norm(x, ord="fro") ** 2
        y_norm2 = torch.linalg.matrix_norm(y, ord="fro") ** 2
        loss = loss + reg * (x_norm2 + y_norm2) / 2.0

    return loss