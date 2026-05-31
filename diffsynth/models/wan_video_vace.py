import torch
from .wan_video_dit import DiTBlock
from ..core.gradient import gradient_checkpoint_forward


class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs, x_subset=None):
        """If `x_subset` is given, it's the slice of `x` at the masked positions
        and is used in place of `x` for the block-0 `before_proj` residual.
        Downstream blocks still operate on the short `c` sequence."""
        if self.block_id == 0:
            c = self.before_proj(c) + (x_subset if x_subset is not None else x)
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        return torch.stack(all_c)


class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # vace blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(
        self, x, vace_context, context, t_mod, freqs,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        mask_bool: torch.Tensor = None,
    ):
        # 1. Patch-embed VACE context and flatten
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]   # each: (1, gp_seq, dim)

        if mask_bool is None:
            # Upstream behavior: zero-pad to DiT seq length
            c = torch.cat([
                torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))], dim=1)
                for u in c
            ])
            x_subset = None
            scatter_back = False
        else:
            # Inpaint-local branch: slice to mask tokens
            assert mask_bool.dtype == torch.bool and mask_bool.dim() == 1, \
                "mask_bool must be 1-D bool"
            assert mask_bool.numel() == x.shape[1], \
                "mask_bool must match DiT seq length"
            # Zero-pad c to DiT seq length, then slice to masked positions
            c = torch.cat([
                torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))], dim=1)
                for u in c
            ])
            c = c[:, mask_bool, :]
            x_subset = x[:, mask_bool, :]
            # Slice freqs to the masked positions so rope_apply sees correct seq length
            freqs = freqs[mask_bool]
            scatter_back = True

        for block in self.vace_blocks:
            c = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                c, x, context, t_mod, freqs, x_subset,
            )

        # Unbind hints. With mask_bool, hints are at masked-position length;
        # scatter into full-length zero buffers so downstream code is unchanged.
        hints = torch.unbind(c)[:-1]
        if not scatter_back:
            return hints
        scattered = []
        for h in hints:
            buf = h.new_zeros(h.shape[0], x.shape[1], h.shape[-1])
            buf[:, mask_bool, :] = h
            scattered.append(buf)
        return tuple(scattered)
