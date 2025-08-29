import os
import math
import torch
import logging
import torch.nn.functional as F
from typing import Optional
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR


def _build_noam_scheduler(optimizer, warmup_steps: int):
    def lr_lambda(step: int):
        # step은 0부터 시작하므로 +1
        s = step + 1
        return min(s ** -0.5, s * (warmup_steps ** -1.5))
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def train_with_contrastive_cca_sr(
    model: torch.nn.Module,
    train_loader,
    *,
    device: Optional[torch.device] = None,
    # Optim
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    use_noam: bool = False,
    warmup_steps: int = 10_000,
    # Losses
    criterion_con=None,                # e.g., SupConLoss(temperature=0.07)
    lambda_con: float = 1.0,
    lambda_cca: float = 1.0,
    lambda_sr: float  = 1.0,
    # Loop
    epochs: int = 3,
    log_every: int = 50,
    grad_clip: Optional[float] = None, # e.g., 1.0
    amp_dtype: Optional[torch.dtype] = torch.float16,
    # Checkpoint & logging
    save_dir: Optional[str] = None,
    save_every_steps: Optional[int] = None,
    enable_logging: bool = True,
):
    """
    EpiAgent를 Con(대조학습) + CCA + SR의 합산 손실로 미세조정하는 학습 루프.

    batch = (cell_ids, signals, ex_ids_list, ex_acc, y) 형태를 가정합니다.
      - cell_ids: [B, L] (token ids)
      - signals:  (SR 입력용 시그널 텐서)
      - ex_ids_list: 길이 B의 list[LongTensor], 각 샘플의 cCRE id들
      - ex_acc: [sum_k M_k] or [B, M] 형태의 접근성 라벨 텐서 (모델이 내부에서 맞춰 씀)
      - y: [B] (SupConLoss용 클래스 라벨)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer: requires_grad 파라미터만 (LoRA만 학습)
    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(opt_params, lr=lr, weight_decay=weight_decay)

    # Scheduler (선택)
    scheduler = _build_noam_scheduler(optimizer, warmup_steps) if use_noam else None

    # AMP
    scaler = GradScaler()

    # 로깅 설정
    if enable_logging and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_fp = os.path.join(save_dir, "train.log")
        logging.basicConfig(
            filename=log_fp,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger("").addHandler(console)

    model.train()

    global_step = 0
    for epoch in range(1, epochs + 1):
        run_con = run_cca = run_sr = run_total = 0.0

        for step, batch in enumerate(train_loader, start=1):
            # 언패킹 & 디바이스 이동
            cell_ids, signals, ex_ids_list, ex_acc, y = batch
            cell_ids = cell_ids.to(device)
            signals  = signals.to(device)
            ex_acc   = ex_acc.to(device)
            y        = y.to(device)

            # list[Tensor]는 개별 .to 필요
            ex_ids_list = [t.to(device) for t in ex_ids_list]

            optimizer.zero_grad(set_to_none=True)

            # AMP 컨텍스트
            use_amp = (device.type == "cuda") and (amp_dtype is not None)
            amp_ctx = autocast(dtype=amp_dtype) if use_amp else torch.cpu.amp.autocast(enabled=False)

            with amp_ctx:
                out = model(
                    input_ids=cell_ids,
                    return_transformer_output=True,
                    calculate_rlm_loss=False,
                    calculate_cca_loss=True,
                    calculate_sr_loss=True,
                    ex_cell_ccre_ids=ex_ids_list,
                    ex_cell_ccre_accessibility=ex_acc,
                    signals=signals,
                )

                # 1) Contrastive (CLS -> 정규화 후 SupConLoss)
                h_cls = out["transformer_outputs"][:, 0, :]   # [B, D]
                z = F.normalize(h_cls, dim=-1)                 # 파라미터 X
                if criterion_con is None:
                    raise ValueError("criterion_con (e.g., SupConLoss) 를 전달해야 합니다.")
                loss_con = criterion_con(z, y)

                # 2) CCA / SR (모델이 반환한 스칼라 손실)
                loss_cca = out["cca_loss"]
                loss_sr  = out["sr_loss"]

                loss = lambda_con * loss_con + lambda_cca * loss_cca + lambda_sr * loss_sr

            # 역전파
            scaler.scale(loss).backward()

            # (선택) gradient clipping
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(opt_params, grad_clip)

            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            # 로그 누적
            global_step += 1
            run_con   += float(loss_con.detach().cpu())
            run_cca   += float(loss_cca.detach().cpu())
            run_sr    += float(loss_sr.detach().cpu())
            run_total += float(loss.detach().cpu())

            # 주기적 로그
            if (step % log_every) == 0:
                denom = log_every
                msg = (f"[Ep {epoch}] Step {step}/{len(train_loader)} | "
                       f"Total {run_total/denom:.4f} | Con {run_con/denom:.4f} | "
                       f"CCA {run_cca/denom:.4f} | SR {run_sr/denom:.4f}")
                if enable_logging:
                    logging.info(msg)
                else:
                    print(msg)
                run_con = run_cca = run_sr = run_total = 0.0

            # 체크포인트 저장
            if save_dir and save_every_steps and (global_step % save_every_steps == 0):
                ckpt = os.path.join(save_dir, f"checkpoint_step_{global_step}.pth")
                torch.save(model.state_dict(), ckpt)
                if enable_logging:
                    logging.info(f"Checkpoint saved: {ckpt}")

        # 에폭 종료 로그
        if enable_logging:
            logging.info(f"End Epoch {epoch}")

    return model
