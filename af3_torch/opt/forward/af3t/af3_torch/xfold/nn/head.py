# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import torch
import torch.nn as nn
import einops

from xfold import feat_batch
from xfold.constants import atom_types
from xfold.nn import template, atom_layout, pairformer

from xfold import fastnn

_CONTACT_THRESHOLD = 8.0
_CONTACT_EPSILON = 1e-3

import os
# Row-blocked confidence head (ConfidenceHead._forward_rows) from this many (padded) tokens: above the 4544 bucket, so inputs up to 4500 tokens
# run forward() unchanged; ROWS_BLOCK_GIB = the transient budget per row block. Env overrides are for experiments.
ROWS_MIN_TOKENS = int(os.environ.get("AF3T_MEMCHUNK_MIN_TOKENS", "4608"))
ROWS_BLOCK_GIB = float(os.environ.get("AF3T_MEMCHUNK_GIB", "4"))


def _block_rows(n, bytes_per_row):
    r = int(ROWS_BLOCK_GIB * 2 ** 30 // max(1, bytes_per_row)) // 64 * 64
    return max(64, min(n, r))


def _tm_term(num_interface_tokens, bin_centers, pae_probs):
    """get_tmscore_adjusted_pae (ConfidenceHead._get_tmscore_adjusted_pae) on a row block."""
    clipped_num_res = torch.clamp(num_interface_tokens, min=19)
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8
    d0 = d0[:, :, None]
    tm_per_bin = 1.0 / (1 + torch.square(bin_centers[None, None, :]) / torch.square(d0))
    return torch.sum(pae_probs * tm_per_bin, dim=-1)


class DistogramHead(nn.Module):
    def __init__(self,
                 c_pair: int = 128,
                 num_bins: int = 64,
                 first_break: float = 2.3125,
                 last_break: float = 21.6875) -> None:
        super(DistogramHead, self).__init__()

        self.c_pair = c_pair
        self.num_bins = num_bins
        self.first_break = first_break
        self.last_break = last_break

        self.half_logits = nn.Linear(self.c_pair, self.num_bins, bias=False)

        breaks = torch.linspace(
            self.first_break,
            self.last_break,
            self.num_bins - 1,
        )

        self.register_buffer('breaks', breaks)

        bin_tops = torch.cat(
            (breaks, (breaks[-1] + (breaks[-1] - breaks[-2])).reshape(1)))
        threshold = _CONTACT_THRESHOLD + _CONTACT_EPSILON
        is_contact_bin = 1.0 * (bin_tops <= threshold)

        self.register_buffer('is_contact_bin', is_contact_bin)

    def forward(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): pair embedding
                [*, N_token, N_token, C_z]

        Returns:
            torch.Tensor: distogram probability distribution
                [*, N_token, N_token, num_bins]
        """

        pair_act = embeddings['pair']
        seq_mask = batch.token_features.mask.to(dtype=torch.bool)
        pair_mask = seq_mask[:, None] * seq_mask[None, :]

        n = pair_act.shape[0]
        if n >= ROWS_MIN_TOKENS:                                    # the same statements per block of rows (the symmetrisation reads the whole half logits)
            rows = _block_rows(n, n * self.num_bins * 4 * 3)
            half = None
            for r0 in range(0, n, rows):
                blk = self.half_logits(pair_act[r0:r0 + rows])
                if half is None:
                    half = torch.empty((n, n, blk.shape[-1]), dtype=blk.dtype, device=blk.device)
                half[r0:r0 + rows] = blk
                del blk
            contact_probs = None
            for r0 in range(0, n, rows):
                logits = half[r0:r0 + rows] + half[:, r0:r0 + rows].transpose(-2, -3)
                probs = torch.softmax(logits, dim=-1)
                blk = torch.einsum('ijk,k->ij', probs, self.is_contact_bin)
                if contact_probs is None:
                    contact_probs = torch.empty((n, n), dtype=blk.dtype, device=blk.device)
                contact_probs[r0:r0 + rows] = pair_mask[r0:r0 + rows] * blk
                del logits, probs, blk
            return {'bin_edges': self.breaks, 'contact_probs': contact_probs}

        left_half_logits = self.half_logits(pair_act)
        right_half_logits = left_half_logits
        logits = left_half_logits + right_half_logits.transpose(-2, -3)
        probs = torch.softmax(logits, dim=-1)
        contact_probs = torch.einsum('ijk,k->ij', probs, self.is_contact_bin)

        contact_probs = pair_mask * contact_probs

        return {
            'bin_edges': self.breaks,
            'contact_probs': contact_probs,
        }


class ConfidenceHead(nn.Module):
    """
    Implements Algorithm 31 in AF3
    """

    def __init__(self, c_single: int = 384, c_pair: int = 128, c_target_feat: int = 447, n_pairformer_layers=4):
        super(ConfidenceHead, self).__init__()

        self.c_single = c_single
        self.c_pair = c_pair
        self.c_target_feat = c_target_feat

        self.dgram_features_config = template.DistogramFeaturesConfig()

        self.num_bins = 64
        self.max_error_bin = 31.0

        self.pae_num_bins = 64
        self.pae_max_error_bin = 31.0

        self.num_plddt_bins = 50
        self.num_atom = atom_types.DENSE_ATOM_NUM
        self.bin_width = 1.0 / self.num_plddt_bins

        self.left_target_feat_project = nn.Linear(
            self.c_target_feat, self.c_pair, bias=False)
        self.right_target_feat_project = nn.Linear(
            self.c_target_feat, self.c_pair, bias=False)
        self.distogram_feat_project = nn.Linear(
            self.dgram_features_config.num_bins, self.c_pair, bias=False)

        self.confidence_pairformer = nn.ModuleList([
            pairformer.PairformerBlock(
                c_single=self.c_single,
                c_pair=self.c_pair,
                with_single=True,
            ) for _ in range(n_pairformer_layers)
        ])

        self.logits_ln = fastnn.LayerNorm(self.c_pair)
        self.left_half_distance_logits = nn.Linear(
            self.c_pair, self.num_bins, bias=False)

        self.register_buffer('distance_breaks', torch.linspace(
            0.0, self.max_error_bin, self.num_bins - 1))
        self.register_buffer(
            'step', self.distance_breaks[1] - self.distance_breaks[0])
        self.register_buffer(
            'bin_centers', self.distance_breaks + self.step / 2)
        self.bin_centers = torch.concatenate(
            [self.bin_centers, self.bin_centers[-1:] + self.step], dim=0
        )

        self.pae_logits_ln = fastnn.LayerNorm(self.c_pair)
        self.pae_logits = nn.Linear(self.c_pair, self.pae_num_bins, bias=False)

        self.register_buffer('pae_breaks', torch.linspace(
            0.0, self.pae_max_error_bin, self.pae_num_bins - 1))
        self.register_buffer(
            'pae_step', self.pae_breaks[1] - self.pae_breaks[0])

        pae_bin_centers_ = self.pae_breaks + self.pae_step / 2
        self.register_buffer(
            'pae_bin_centers', torch.concatenate(
                [pae_bin_centers_, pae_bin_centers_[-1:] + self.pae_step], dim=0
            )
        )

        self.register_buffer('plddt_bin_centers', torch.arange(
            0.5 * self.bin_width, 1.0, self.bin_width))

        self.plddt_logits_ln = fastnn.LayerNorm(self.c_single)
        self.plddt_logits = nn.Linear(
            self.c_single, self.num_atom * self.num_plddt_bins, bias=False)

        self.experimentally_resolved_ln = fastnn.LayerNorm(self.c_single)
        self.experimentally_resolved_logits = nn.Linear(
            self.c_single, self.num_atom * 2, bias=False)

    def _embed_features(
        self,
        dense_atom_positions: torch.Tensor,
        token_atoms_to_pseudo_beta: atom_layout.GatherInfo,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:

        out = self.left_target_feat_project(target_feat)[..., None, :, :] \
            + self.right_target_feat_project(target_feat)[..., None, :]

        positions = atom_layout.convert(
            token_atoms_to_pseudo_beta,
            dense_atom_positions,
            layout_axes=(-3, -2),
        )

        dgram = template.dgram_from_positions(
            positions, self.dgram_features_config
        )

        dgram *= pair_mask[..., None]

        out += self.distogram_feat_project(dgram)

        return out

    def forward(
        self,
        dense_atom_positions: torch.Tensor,
        embeddings: dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
        token_atoms_to_pseudo_beta: atom_layout.GatherInfo,
        asym_id: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            target_feat (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            dense_atom_positions (torch.Tensor): array of positions.
                [N_tokens, N_atom, 3] 
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            token_atoms_to_pseudo_beta (atom_layout.GatherInfo): Pseudo beta info for atom tokens.
        """

        dtype = dense_atom_positions.dtype
        if seq_mask.shape[0] >= ROWS_MIN_TOKENS:
            return self._forward_rows(dense_atom_positions, embeddings, seq_mask, token_atoms_to_pseudo_beta, asym_id)

        seq_mask_cast = seq_mask.to(dtype=dtype)
        pair_mask = seq_mask_cast[:, None] * seq_mask_cast[None, :]
        pair_mask = pair_mask.to(dtype=dtype)

        pair_act = embeddings['pair'].clone().to(dtype=dtype)
        single_act = embeddings['single'].clone().to(dtype=dtype)
        target_feat = embeddings['target_feat'].clone().to(dtype=dtype)

        pair_act += self._embed_features(
            dense_atom_positions, token_atoms_to_pseudo_beta, pair_mask, target_feat)

        # pairformer stack
        for layer in self.confidence_pairformer:
            pair_act, single_act = layer(
                pair_act, pair_mask, single_act, seq_mask)

        # Produce logits to predict a distogram of pairwise distance errors
        # between the input prediction and the ground truth.

        left_distance_logits = self.left_half_distance_logits(
            self.logits_ln(pair_act))
        right_distance_logits = left_distance_logits
        distance_logits = left_distance_logits + \
            torch.transpose(right_distance_logits, -2, -3)

        distance_probs = torch.softmax(distance_logits, dim=-1)
        pred_distance_error = (
            torch.sum(distance_probs * self.bin_centers, dim=-1) * pair_mask
        )
        average_pred_distance_error = torch.sum(
            pred_distance_error, dim=[-2, -1]
        ) / torch.sum(pair_mask, dim=[-2, -1])

        # Predicted aligned error
        pae_outputs = {}
        pae_logits = self.pae_logits(self.pae_logits_ln(pair_act))
        pae_probs = torch.softmax(pae_logits, dim=-1)

        pair_mask_bool = pair_mask.to(dtype=torch.bool)

        pae = torch.sum(pae_probs * self.pae_bin_centers,
                        dim=-1) * pair_mask_bool
        pae_outputs.update({
            'full_pae': pae,
        })

        tmscore_adjusted_pae_global, tmscore_adjusted_pae_interface = (
            self._get_tmscore_adjusted_pae(
                asym_id=asym_id,
                seq_mask=seq_mask,
                pair_mask=pair_mask_bool,
                bin_centers=self.pae_bin_centers,
                pae_probs=pae_probs,
            )
        )

        pae_outputs.update({
            'tmscore_adjusted_pae_global': tmscore_adjusted_pae_global,
            'tmscore_adjusted_pae_interface': tmscore_adjusted_pae_interface,
        })

        # pLDDT
        plddt_logits = self.plddt_logits(self.plddt_logits_ln(single_act))
        plddt_logits = einops.rearrange(
            plddt_logits, '... (n_atom n_bins) -> ... n_atom n_bins', n_bins=self.num_plddt_bins)
        predicted_lddt = torch.sum(
            torch.softmax(plddt_logits, dim=-1) * self.plddt_bin_centers, dim=-1
        )
        predicted_lddt = predicted_lddt * 100.0

        # Experimentally resolved
        experimentally_resolved_logits = self.experimentally_resolved_logits(
            self.experimentally_resolved_ln(single_act))
        experimentally_resolved_logits = einops.rearrange(
            experimentally_resolved_logits, '... (n_atom n_bins) -> ... n_atom n_bins', n_bins=2)

        predicted_experimentally_resolved = torch.softmax(
            experimentally_resolved_logits, dim=-1
        )[..., 1]

        return {
            'predicted_lddt': predicted_lddt,
            'predicted_experimentally_resolved': predicted_experimentally_resolved,
            'full_pde': pred_distance_error,
            'average_pde': average_pred_distance_error,
            **pae_outputs,
        }

    def _forward_rows(self, dense_atom_positions, embeddings, seq_mask, token_atoms_to_pseudo_beta, asym_id):
        """forward() for large inputs: the same statements with every [N, N, c] intermediate outside the pairformer stack formed per block of
        rows (ROWS_BLOCK_GIB of transients per block) -- the feature embedding added into the pair in place, the PDE / PAE / TM-adjusted maps
        written row block by row block. embeddings['pair'] may live on the host (the caller parks it there): it is copied to the device once.
        Same values as forward() up to the GEMM leading size per block and the summation order of average_pde (tolerance class)."""
        dtype = dense_atom_positions.dtype
        dev = dense_atom_positions.device
        N = seq_mask.shape[0]
        seq_mask_cast = seq_mask.to(dtype=dtype)
        pair_act = embeddings['pair'].to(device=dev, dtype=dtype, copy=True)
        single_act = embeddings['single'].clone().to(dtype=dtype)
        target_feat = embeddings['target_feat'].clone().to(dtype=dtype)
        pair_mask = seq_mask_cast[:, None] * seq_mask_cast[None, :]                 # [N, N] (the pairformer's mask, as forward())
        rows = _block_rows(N, N * self.c_pair * 4 * 3)

        left = self.left_target_feat_project(target_feat)                            # out[i, j] = left[j] + right[i] (forward()'s broadcast)
        right = self.right_target_feat_project(target_feat)
        positions = atom_layout.convert(token_atoms_to_pseudo_beta, dense_atom_positions, layout_axes=(-3, -2))
        cfg = self.dgram_features_config
        lower = torch.square(torch.linspace(cfg.min_bin, cfg.max_bin, cfg.num_bins, device=dev))
        upper = torch.concatenate([lower[1:], torch.ones(1, device=dev) * 1e8], dim=-1)
        for r0 in range(0, N, rows):
            r1 = min(N, r0 + rows)
            dist2 = torch.sum(torch.square(positions[r0:r1, None, :] - positions[None, :, :]), dim=-1, keepdims=True)
            dgram = (dist2 > lower).to(dtype=torch.float32) * (dist2 < upper).to(dtype=torch.float32)
            dgram *= pair_mask[r0:r1, :, None]
            out = left[None, :, :] + right[r0:r1, None, :]
            out += self.distogram_feat_project(dgram)
            pair_act[r0:r1] += out
            del dist2, dgram, out

        for layer in self.confidence_pairformer:
            pair_act, single_act = layer(pair_act, pair_mask, single_act, seq_mask)

        pair_mask_bool = pair_mask.to(dtype=torch.bool)
        pde = torch.empty((N, N), dtype=torch.float32, device=dev)
        pae = torch.empty((N, N), dtype=torch.float32, device=dev)
        tmg = torch.empty((N, N), dtype=torch.float32, device=dev)
        tmi = torch.empty((N, N), dtype=torch.float32, device=dev)
        half = None
        for r0 in range(0, N, rows):                                                  # left half distance logits, whole (the symmetrisation reads columns)
            r1 = min(N, r0 + rows)
            blk = self.left_half_distance_logits(self.logits_ln(pair_act[r0:r1]))
            if half is None:
                half = torch.empty((N, N, blk.shape[-1]), dtype=blk.dtype, device=dev)
            half[r0:r1] = blk
            del blk
        for r0 in range(0, N, rows):
            r1 = min(N, r0 + rows)
            logits = half[r0:r1] + torch.transpose(half[:, r0:r1], -2, -3)
            probs = torch.softmax(logits, dim=-1)
            pde[r0:r1] = torch.sum(probs * self.bin_centers, dim=-1) * pair_mask[r0:r1]
            del logits, probs
        del half
        average_pred_distance_error = torch.sum(pde, dim=[-2, -1]) / torch.sum(pair_mask, dim=[-2, -1])

        x_all = asym_id[None, :] == asym_id[:, None]                                  # [N, N] bool (small); _get_tmscore_adjusted_pae's counts
        num_chain_tokens = torch.sum(x_all * pair_mask_bool, dim=-1, dtype=torch.int32)
        n_glob = seq_mask.sum()
        for r0 in range(0, N, rows):
            r1 = min(N, r0 + rows)
            probs = torch.softmax(self.pae_logits(self.pae_logits_ln(pair_act[r0:r1])), dim=-1)
            pmb = pair_mask_bool[r0:r1]
            pae[r0:r1] = torch.sum(probs * self.pae_bin_centers, dim=-1) * pmb
            nit = num_chain_tokens[None, :] + num_chain_tokens[r0:r1, None]
            xs = x_all[r0:r1]
            nit -= xs * (nit // 2)
            nit = nit * pmb
            ngt = torch.ones(size=pmb.shape, dtype=torch.int32, device=dev)
            ngt *= n_glob
            tmg[r0:r1] = _tm_term(ngt, self.pae_bin_centers, probs)
            tmi[r0:r1] = _tm_term(nit, self.pae_bin_centers, probs)
            del probs, nit, ngt
        del pair_act

        plddt_logits = self.plddt_logits(self.plddt_logits_ln(single_act))
        plddt_logits = einops.rearrange(
            plddt_logits, '... (n_atom n_bins) -> ... n_atom n_bins', n_bins=self.num_plddt_bins)
        predicted_lddt = torch.sum(
            torch.softmax(plddt_logits, dim=-1) * self.plddt_bin_centers, dim=-1) * 100.0
        experimentally_resolved_logits = self.experimentally_resolved_logits(
            self.experimentally_resolved_ln(single_act))
        experimentally_resolved_logits = einops.rearrange(
            experimentally_resolved_logits, '... (n_atom n_bins) -> ... n_atom n_bins', n_bins=2)
        predicted_experimentally_resolved = torch.softmax(experimentally_resolved_logits, dim=-1)[..., 1]
        return {
            'predicted_lddt': predicted_lddt,
            'predicted_experimentally_resolved': predicted_experimentally_resolved,
            'full_pde': pde,
            'average_pde': average_pred_distance_error,
            'full_pae': pae,
            'tmscore_adjusted_pae_global': tmg,
            'tmscore_adjusted_pae_interface': tmi,
        }

    def _get_tmscore_adjusted_pae(self,
                                  asym_id: torch.Tensor,
                                  seq_mask: torch.Tensor,
                                  pair_mask: torch.Tensor,
                                  bin_centers: torch.Tensor,
                                  pae_probs: torch.Tensor,
                                  ):

        def get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs):
            # Clip to avoid negative/undefined d0.
            clipped_num_res = torch.clamp(num_interface_tokens, min=19)

            # Compute d_0(num_res) as defined by TM-score, eqn. (5) in
            # http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
            # Yang & Skolnick "Scoring function for automated
            # assessment of protein structure template quality" 2004.
            d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

            # Make compatible with [num_tokens, num_tokens, num_bins]
            d0 = d0[:, :, None]
            bin_centers = bin_centers[None, None, :]

            # TM-Score term for every bin.
            tm_per_bin = 1.0 / \
                (1 + torch.square(bin_centers) / torch.square(d0))
            # E_distances tm(distance).
            predicted_tm_term = torch.sum(pae_probs * tm_per_bin, dim=-1)
            return predicted_tm_term

        # Interface version
        x = asym_id[None, :] == asym_id[:, None]
        num_chain_tokens = torch.sum(x * pair_mask, dim=-1, dtype=torch.int32)
        num_interface_tokens = num_chain_tokens[None,
                                                :] + num_chain_tokens[:, None]
        # Don't double-count within a single chain
        num_interface_tokens -= x * (num_interface_tokens // 2)
        num_interface_tokens = num_interface_tokens * pair_mask

        num_global_tokens = torch.ones(
            size=pair_mask.shape, dtype=torch.int32, device=x.device
        )
        num_global_tokens *= seq_mask.sum()

        assert num_global_tokens.dtype == torch.int32
        assert num_interface_tokens.dtype == torch.int32
        global_apae = get_tmscore_adjusted_pae(
            num_global_tokens, bin_centers, pae_probs
        )
        interface_apae = get_tmscore_adjusted_pae(
            num_interface_tokens, bin_centers, pae_probs
        )
        return global_apae, interface_apae
