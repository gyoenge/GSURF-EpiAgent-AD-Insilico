import torch
import numpy as np
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from collections import Counter


def infer_cell_embeddings_from_trainloader(model, device, train_loader,
                                           normalize=True, use_cls=True,
                                           rebuild_noshuffle=True):

    assert device.type == 'cuda', "use_flash_attn=True이면 CUDA가 필요합니다."

    if rebuild_noshuffle:
        loader = DataLoader(
            dataset=train_loader.dataset,
            batch_size=getattr(train_loader, "batch_size", 8),
            shuffle=False,
            num_workers=getattr(train_loader, "num_workers", 0),
            collate_fn=getattr(train_loader, "collate_fn", None),
            pin_memory=True
        )
    else:
        loader = train_loader

    model.eval()
    all_chunks = []

    amp_dtype = torch.float16  

    with torch.inference_mode():
        for batch in loader:
            cell_ids = batch[0].to(device, non_blocking=True)  # (cell_ids, signals, ex_ids_list, ex_acc, y)

            with torch.cuda.amp.autocast(dtype=amp_dtype):
                out = model(
                    input_ids=cell_ids,
                    return_transformer_output=True,
                    calculate_rlm_loss=False,
                    calculate_cca_loss=False,
                    calculate_sr_loss=False,
                )

                h = out['transformer_outputs']     # [B, L, D]
                h = h[:, 0, :] if use_cls else h.mean(dim=1)  # [B, D]
                if normalize:
                    h = F.normalize(h, dim=-1)

            all_chunks.append(h.detach().cpu())

    return torch.cat(all_chunks, dim=0).numpy()
