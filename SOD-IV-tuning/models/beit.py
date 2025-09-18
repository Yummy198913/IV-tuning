# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.runner.checkpoint import _load_checkpoint
from scipy import interpolate

from mmengine.dist import get_dist_info
from mmengine.model import is_model_wrapper



def load_checkpoint(model, filename, map_location="cpu", strict=False, logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "module" in checkpoint:
        state_dict = checkpoint["module"]
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # for MoBY, load model of online branch
    if sorted(list(state_dict.keys()))[0].startswith("encoder"):
        state_dict = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }

    # reshape absolute position embedding for Swin
    if state_dict.get("absolute_pos_embed") is not None:
        absolute_pos_embed = state_dict["absolute_pos_embed"]
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            logger.warning("Error in loading absolute_pos_embed, pass")
        else:
            state_dict["absolute_pos_embed"] = absolute_pos_embed.view(
                N2, H, W, C2
            ).permute(0, 3, 1, 2)

    rank, _ = get_dist_info()
    if "rel_pos_bias.relative_position_bias_table" in state_dict:
        if rank == 0:
            print("Expand the shared relative position embedding to each layers. ")
            num_layers = model.get_num_layers()
            rel_pos_bias = state_dict["rel_pos_bias.relative_position_bias_table"]
            for i in range(num_layers):
                state_dict[
                    "blocks.%d.attn.relative_position_bias_table" % i
                ] = rel_pos_bias.clone()

        state_dict.pop("rel_pos_bias.relative_position_bias_table")

    all_keys = list(state_dict.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            state_dict.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = state_dict[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (
                dst_patch_shape[1] * 2 - 1
            )
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                if rank == 0:
                    print(
                        "Position interpolate for %s from %dx%d to %dx%d"
                        % (key, src_size, src_size, dst_size, dst_size)
                    )
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r**n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.13492:
                #     q = 1.13492

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)
                if rank == 0:
                    print("x = {}".format(x))
                    print("dx = {}".format(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind="cubic")
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy))
                        .contiguous()
                        .view(-1, 1)
                        .to(rel_pos_bias.device)
                    )

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                state_dict[key] = new_rel_pos_bias

    if "pos_embed" in state_dict:
        pos_embed_checkpoint = state_dict["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            if rank == 0:
                print(
                    "Position interpolate from %dx%d to %dx%d"
                    % (orig_size, orig_size, new_size, new_size)
                )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict["pos_embed"] = new_pos_embed

    # interpolate position bias table if needed
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k
    ]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {table_key}, pass")
        else:
            if L1 != L2:
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                table_pretrained_resized = F.interpolate(
                    table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                    size=(S2, S2),
                    mode="bicubic",
                )
                state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(
                    1, 0
                )

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint



def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.
    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_model_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            "unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n'
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n'
        )

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)