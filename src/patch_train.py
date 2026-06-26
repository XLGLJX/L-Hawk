import os
import random
import sys
from tqdm import tqdm
from pathlib import Path

from src.train_eval import *
from src.Lhawk import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # program root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def _is_batched_positions(pos):
    return (
        isinstance(pos, (list, tuple))
        and len(pos) > 0
        and isinstance(pos[0], (list, tuple))
    )


def _random_positions(patch_obj, cfg, image_shape, batch_size):
    return [patch_obj.random_pos(cfg, image_shape) for _ in range(batch_size)]


def _mean(values):
    return sum(float(v) for v in values) / max(len(values), 1)


def _value_at(value, idx):
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value[idx].item())
    if isinstance(value, (list, tuple)):
        return value[idx]
    return value


def _apply_patch_batch(patch_obj, img, positions, **kwargs):
    if not _is_batched_positions(positions):
        out = patch_obj.apply(img, positions, **kwargs)
        return out, [patch_obj.last_scale for _ in range(img.size(0))]

    outputs = []
    scales = []
    set_resize = kwargs.pop("set_resize", None)
    set_rotate = kwargs.pop("set_rotate", None)
    for idx, pos in enumerate(positions):
        out = patch_obj.apply(
            img[idx:idx + 1],
            pos,
            set_resize=_value_at(set_resize, idx),
            set_rotate=_value_at(set_rotate, idx),
            **kwargs,
        )
        outputs.append(out)
        scales.append(patch_obj.last_scale)
    patch_obj.last_scale = _mean(scales)
    return torch.cat(outputs, dim=0), scales


def _align_trigger_to_patch(cfg, trigger_mask, pos, patch_obj, image_shape):
    """Move each synthetic laser stripe vertically so it covers the current patch.

    The trigger tensor remains a batch tensor, so the image composition and model
    forward pass stay vectorized.  Alignment is enabled by demo.py only for
    --trigger-source laser; fixed captured triggers keep their original layout.
    """
    if not getattr(cfg.ATTACKER, "ALIGN_TRIGGER_TO_PATCH", False):
        return trigger_mask
    if trigger_mask.dim() != 4:
        return trigger_mask

    image_h = int(image_shape[-2])
    if image_h <= 0:
        return trigger_mask

    patch_top = int(pos[0])
    patch_h = int(getattr(patch_obj, "h", 0))
    if patch_h <= 0:
        return trigger_mask

    patch_center = patch_top + patch_h * 0.5
    source_top = getattr(cfg.ATTACKER, "TRIGGER_STRIPE_TOP", None)
    stripe_h = getattr(cfg.ATTACKER, "TRIGGER_STRIPE_ROWS", None)
    if source_top is not None and stripe_h is not None:
        source_top = max(0, min(int(source_top), image_h - 1))
        stripe_h = max(1, min(int(stripe_h), image_h - source_top))
        if stripe_h >= image_h:
            return trigger_mask

        target_top = int(round(patch_center - stripe_h * 0.5))
        target_top = max(0, min(target_top, image_h - stripe_h))
        aligned = torch.zeros_like(trigger_mask)
        aligned[:, :, target_top:target_top + stripe_h, :] = (
            trigger_mask[:, :, source_top:source_top + stripe_h, :]
        )
        return aligned

    aligned = torch.zeros_like(trigger_mask)
    row_activity = trigger_mask.detach().abs().sum(dim=(1, 3))
    active_rows = row_activity > 0

    for idx in range(trigger_mask.size(0)):
        rows = torch.nonzero(active_rows[idx], as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        src_top = int(rows[0].item())
        src_bottom = int(rows[-1].item()) + 1
        stripe_h = src_bottom - src_top
        if stripe_h >= image_h:
            aligned[idx] = trigger_mask[idx]
            continue

        target_top = int(round(patch_center - stripe_h * 0.5))
        target_top = max(0, min(target_top, image_h - stripe_h))
        aligned[idx, :, target_top:target_top + stripe_h, :] = trigger_mask[idx, :, src_top:src_bottom, :]

    return aligned


def _apply_trigger_to_patch_aligned(cfg, imgn, trigger_mask, pos, patch_obj):
    """Apply a trigger to the current batch, aligning laser stripes to the patch.

    For synthetic laser triggers the source stripe layout is known, so only the
    affected row slice is copied and added for the whole batch.  This avoids
    materializing a full-image aligned mask in the common path.
    """
    if not getattr(cfg.ATTACKER, "ALIGN_TRIGGER_TO_PATCH", False):
        return torch.clamp(torch.add(imgn, trigger_mask / 255), 0, 1)
    if trigger_mask.dim() != 4:
        return torch.clamp(torch.add(imgn, trigger_mask / 255), 0, 1)
    batch_size = imgn.size(0)
    if trigger_mask.size(0) == 1 and batch_size > 1:
        trigger_mask = trigger_mask.expand(batch_size, -1, -1, -1)
    elif trigger_mask.size(0) != batch_size:
        raise ValueError(
            f"trigger batch size {trigger_mask.size(0)} must be 1 or match image batch size {batch_size}"
        )
    if _is_batched_positions(pos):
        image_h = int(imgn.shape[-2])
        patch_h = int(getattr(patch_obj, "h", 0))
        source_top = getattr(cfg.ATTACKER, "TRIGGER_STRIPE_TOP", None)
        stripe_h = getattr(cfg.ATTACKER, "TRIGGER_STRIPE_ROWS", None)
        if patch_h <= 0 or source_top is None or stripe_h is None:
            outputs = [
                _apply_trigger_to_patch_aligned(cfg, imgn[idx:idx + 1], trigger_mask[idx:idx + 1], p, patch_obj)
                for idx, p in enumerate(pos)
            ]
            return torch.cat(outputs, dim=0)

        source_top = max(0, min(int(source_top), image_h - 1))
        stripe_h = max(1, min(int(stripe_h), image_h - source_top))
        if stripe_h >= image_h:
            return torch.clamp(torch.add(imgn, trigger_mask / 255), 0, 1)

        imgp = imgn.clone()
        for idx, p in enumerate(pos):
            patch_center = int(p[0]) + patch_h * 0.5
            target_top = int(round(patch_center - stripe_h * 0.5))
            target_top = max(0, min(target_top, image_h - stripe_h))
            imgp[idx:idx + 1, :, target_top:target_top + stripe_h, :] = torch.clamp(
                imgp[idx:idx + 1, :, target_top:target_top + stripe_h, :]
                + trigger_mask[idx:idx + 1, :, source_top:source_top + stripe_h, :] / 255,
                0,
                1,
            )
        return imgp

    image_h = int(imgn.shape[-2])
    if image_h <= 0:
        return torch.clamp(torch.add(imgn, trigger_mask / 255), 0, 1)

    patch_top = int(pos[0])
    patch_h = int(getattr(patch_obj, "h", 0))
    source_top = getattr(cfg.ATTACKER, "TRIGGER_STRIPE_TOP", None)
    stripe_h = getattr(cfg.ATTACKER, "TRIGGER_STRIPE_ROWS", None)
    if patch_h <= 0 or source_top is None or stripe_h is None:
        aligned_mask = _align_trigger_to_patch(cfg, trigger_mask, pos, patch_obj, imgn.shape[-2:])
        return torch.clamp(torch.add(imgn, aligned_mask / 255), 0, 1)

    source_top = max(0, min(int(source_top), image_h - 1))
    stripe_h = max(1, min(int(stripe_h), image_h - source_top))
    if stripe_h >= image_h:
        return torch.clamp(torch.add(imgn, trigger_mask / 255), 0, 1)

    patch_center = patch_top + patch_h * 0.5
    target_top = int(round(patch_center - stripe_h * 0.5))
    target_top = max(0, min(target_top, image_h - stripe_h))

    imgp = imgn.clone()
    imgp[:, :, target_top:target_top + stripe_h, :] = torch.clamp(
        imgp[:, :, target_top:target_top + stripe_h, :]
        + trigger_mask[:, :, source_top:source_top + stripe_h, :] / 255,
        0,
        1,
    )
    return imgp


def train(cfg, model, relpos, relpos3, patch, patch2, trigger_mask,
          content_loss, tv_loss, nps_loss, quick_load, train_loader, device, e):
    if cfg.DETECTOR.NAME == "faster_rcnn":
        model.train()
    else:
        model.eval()
    total_loss = torch.zeros(1, device=device)
    log_loss = torch.zeros(5, device=device)
    update_count = 0
    batch_records = []
    max_train_batches = getattr(cfg.ATTACKER, "TRAIN_BATCH", None)
    total_batches = max_train_batches if max_train_batches is not None else len(train_loader)
    for i, img in tqdm(enumerate(train_loader), desc=f'Training epoch {e}', total=total_batches):
        if max_train_batches is not None and i >= max_train_batches:
            break
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        batch_size = img.size(0)
        h, w = img.shape[-2:]
        random_index = torch.randint(0, trigger_mask.size(0), (batch_size,), device=device)
        selected_mask = torch.index_select(trigger_mask, 0, random_index)
        batch_loss = torch.zeros(5, device=device)
        batch_count = 0
        batch_grad_norm = 0.0
        batch_update_l2 = 0.0
        for j in range(cfg.ATTACKER.REPEAT):
            if cfg.ATTACKER.TYPE == "CA" or cfg.ATTACKER.TYPE == "TA-C" or cfg.ATTACKER.TYPE == "TA-D":
                pos = _random_positions(patch, cfg, (h, w), batch_size)
                imgn, scales = _apply_patch_batch(patch, img, pos, do_random_color=True)
                if cfg.ATTACKER.TYPE == "CA" or cfg.ATTACKER.TYPE == "TA-D":
                    gt_box, dummy_box, start_row, end_row = _make_boxes(
                        patch, pos, cfg.DETECTOR.NAME[:4].upper(), batch_size=batch_size, scales=scales)
                    if cfg.ATTACKER.TYPE == "TA-D":
                        gt_box_origin, _, _, _ = _make_boxes(
                            patch, pos, cfg.DETECTOR.NAME[:4].upper(), origin_index=cfg.origin_index,
                            batch_size=batch_size, scales=scales)
                last_scale = _mean(scales)
            elif cfg.ATTACKER.TYPE == "HA":
                pos = _random_positions(patch2, cfg, (h, w), batch_size)
                dx, dy = random.randint(-5, 5), random.randint(-5, 5)
                relpos2 = (relpos[0] + dx, relpos[1] + dy)
                patch2.data = patch.apply(quick_load("assets/stop_sign.png"), relpos2, do_random_color=True)
                if cfg.ATTACKER.DOUBLE_APPLY:
                    dx, dy = random.randint(-5, 5), random.randint(-5, 5)
                    relpos2 = (relpos3[0] + dx, relpos3[1] + dy)
                    patch2.data = patch.apply(patch2.data, relpos2, do_random_color=True)
                imgn, scales = _apply_patch_batch(patch2, img, pos, do_random_color=False)
                gt_box, dummy_box, start_row, end_row = _make_boxes(
                    patch2, pos, cfg.DETECTOR.NAME[:4].upper(), batch_size=batch_size, scales=scales)
                last_scale = _mean(scales)
            imgp = _apply_trigger_to_patch_aligned(
                cfg, imgn, selected_mask, pos, patch2 if cfg.ATTACKER.TYPE == "HA" else patch)

            if cfg.ATTACKER.TYPE == "HA":
                loss1 = model(imgp, gt_box, hiding=True)
                loss2 = model(imgn, gt_box)
            elif cfg.ATTACKER.TYPE == "CA":
                loss1 = model(imgp.clone(), gt_box)
                loss2 = model(imgn, gt_box, hiding=True)
            elif cfg.ATTACKER.TYPE == "TA-D":
                loss1 = model(imgp, gt_box)
                loss2 = model(imgn, gt_box_origin)
            elif cfg.ATTACKER.TYPE == "TA-C":
                loss1 = model(imgp, patch.target, tr=True)
                loss2 = model(imgn, patch.target, tr=False)

            loss3 = tv_loss(patch.data)
            loss4 = content_loss(patch.data)
            loss5 = nps_loss(patch.data)
            loss = (1 / last_scale ** 2) * (loss1 + cfg.ATTACKER.ALPHA * loss2) + cfg.ATTACKER.BETA * loss3 + cfg.ATTACKER.CETA * loss4 + cfg.ATTACKER.DELTA * loss5
            if torch.isnan(loss).any(): continue
            total_loss += loss.detach()
            log_loss += torch.tensor((loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()),
                                     device=device)
            update_count += 1
            batch_loss += torch.tensor((loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()),
                                       device=device)
            batch_count += 1
            grad_norm, update_l2 = patch.update(loss)
            batch_grad_norm += grad_norm
            batch_update_l2 += update_l2

        if batch_count > 0:
            avg = (batch_loss / batch_count).tolist()
            batch_records.append({
                "epoch": e,
                "batch": i,
                "total": float(avg[0] + avg[1] + avg[2] + avg[3] + avg[4]),
                "loss1_triggered_attack": float(avg[0]),
                "loss2_aux_no_trigger": float(avg[1]),
                "loss3_tv": float(avg[2]),
                "loss4_content": float(avg[3]),
                "loss5_nps": float(avg[4]),
                "patch_gradient": batch_grad_norm / batch_count,
                "patch_update_l2": batch_update_l2 / batch_count,
            })
        del imgn, imgp, last_scale, loss1, loss2, loss3, loss4, loss5, loss
        torch.cuda.empty_cache()

    if update_count == 0:
        return {
            "total": 0.0,
            "loss1_triggered_attack": 0.0,
            "loss2_aux_no_trigger": 0.0,
            "loss3_tv": 0.0,
            "loss4_content": 0.0,
            "loss5_nps": 0.0,
        }, []
    avg_losses = (log_loss / update_count).detach().cpu().tolist()
    return {
        "total": float((total_loss / update_count).item()),
        "loss1_triggered_attack": float(avg_losses[0]),
        "loss2_aux_no_trigger": float(avg_losses[1]),
        "loss3_tv": float(avg_losses[2]),
        "loss4_content": float(avg_losses[3]),
        "loss5_nps": float(avg_losses[4]),
    }, batch_records


def score_trigger_loss(
    cfg,
    model,
    relpos,
    relpos3,
    patch,
    patch2,
    trigger_mask,
    quick_load,
    data_loader,
    device,
    e,
    search_batch,
    zeta=1.0,
    psi=None,
    quiet=True,
):
    """Score one trigger candidate with the paper's Eq. 9-style objective.

    The patch is fixed. Lower returned loss is better:
        zeta * loss_attack(x, delta*, t) + psi * loss_benign(x, delta*)
    """
    if psi is None:
        psi = cfg.ATTACKER.ALPHA
    if cfg.DETECTOR.NAME == "faster_rcnn":
        model.train()
    else:
        model.eval()
    total_loss = torch.zeros(1, device=device)
    evaluated = 0
    set_resize = torch.empty(search_batch, device=patch.device).uniform_(
        cfg.EVAL.SCALE_EVAL, cfg.EVAL.SCALE_EVAL)
    set_rotate = torch.empty(search_batch, device=patch.device).uniform_(
        -cfg.EVAL.ANGLE_EVAL, cfg.EVAL.ANGLE_EVAL)
    pbar = tqdm(desc=f'Trigger loss epoch {e}', total=search_batch, disable=quiet)
    with torch.no_grad():
        for img in data_loader:
            if evaluated >= search_batch:
                break
            if isinstance(img, list) or isinstance(img, tuple):
                img = img[0]
            remaining = search_batch - evaluated
            img = img[:remaining].to(patch.device)
            batch_size = img.size(0)
            batch_start = evaluated
            h, w = img.shape[-2:]

            if cfg.ATTACKER.TYPE == "CA" or cfg.ATTACKER.TYPE == "TA-C" or cfg.ATTACKER.TYPE == "TA-D":
                pos = _random_positions(patch, cfg, (h, w), batch_size)
                imgn, scales = _apply_patch_batch(
                    patch, img, pos, test_mode=True,
                    set_resize=set_resize[batch_start:batch_start + batch_size],
                    set_rotate=set_rotate[batch_start:batch_start + batch_size],
                    do_random_color=True)
                if cfg.ATTACKER.TYPE == "CA" or cfg.ATTACKER.TYPE == "TA-D":
                    gt_box, _, _, _ = _make_boxes(
                        patch, pos, cfg.DETECTOR.NAME[:4].upper(), batch_size=batch_size, scales=scales)
                    if cfg.ATTACKER.TYPE == "TA-D":
                        gt_box_origin, _, _, _ = _make_boxes(
                            patch, pos, cfg.DETECTOR.NAME[:4].upper(),
                            origin_index=cfg.origin_index, batch_size=batch_size, scales=scales)
            elif cfg.ATTACKER.TYPE == "HA":
                pos = _random_positions(patch2, cfg, (h, w), batch_size)
                dx, dy = 0, 0
                relpos2 = (relpos[0] + dx, relpos[1] + dy)
                patch2.data = patch.apply(
                    quick_load("assets/stop_sign.png"), relpos2, test_mode=True, do_random_color=True)
                if cfg.ATTACKER.DOUBLE_APPLY:
                    dx, dy = 0, 0
                    relpos2 = (relpos3[0] + dx, relpos3[1] + dy)
                    patch2.data = patch.apply(
                        patch2.data, relpos2, test_mode=True, do_random_color=True)
                imgn, scales = _apply_patch_batch(
                    patch2, img, pos, test_mode=True,
                    set_resize=set_resize[batch_start:batch_start + batch_size],
                    set_rotate=set_rotate[batch_start:batch_start + batch_size],
                    do_random_color=False)
                gt_box, _, _, _ = _make_boxes(
                    patch2, pos, cfg.DETECTOR.NAME[:4].upper(), batch_size=batch_size, scales=scales)

            if trigger_mask.size(0) == 1:
                selected_mask = trigger_mask
            else:
                random_index = torch.randint(0, trigger_mask.size(0), (batch_size,), device=device)
                selected_mask = torch.index_select(trigger_mask, 0, random_index)
            imgp = _apply_trigger_to_patch_aligned(
                cfg, imgn, selected_mask, pos, patch2 if cfg.ATTACKER.TYPE == "HA" else patch)

            if cfg.ATTACKER.TYPE == "HA":
                loss_attack = model(imgp, gt_box, hiding=True)
                loss_benign = model(imgn, gt_box)
            elif cfg.ATTACKER.TYPE == "CA":
                loss_attack = model(imgp, gt_box)
                loss_benign = model(imgn, gt_box, hiding=True)
            elif cfg.ATTACKER.TYPE == "TA-D":
                loss_attack = model(imgp, gt_box)
                loss_benign = model(imgn, gt_box_origin)
            elif cfg.ATTACKER.TYPE == "TA-C":
                loss_attack = model(imgp, patch.target, tr=True)
                loss_benign = model(imgn, patch.target, tr=False)

            loss = zeta * loss_attack + psi * loss_benign
            if torch.isnan(loss).any():
                continue
            total_loss += loss.detach() * batch_size
            evaluated += batch_size
            pbar.update(batch_size)
            del imgn, imgp, loss_attack, loss_benign, loss
            torch.cuda.empty_cache()
    pbar.close()
    if evaluated == 0:
        raise RuntimeError("Trigger loss scoring dataloader did not yield any valid samples.")
    return float((total_loss / evaluated).item())


def eval(cfg, model, relpos, relpos3, patch, patch2, trigger_mask, quick_load, test_loader, e,
         eval_batch=None, quiet=False, capture_sample=False):
    model.eval()
    success, success_1, success_2 = 0, 0, 0
    total_eval = eval_batch or cfg.ATTACKER.EVAL_BATCH
    sample_index = random.randrange(total_eval) if capture_sample else None
    visualization_sample = None
    set_resize = torch.empty(total_eval, device=patch.device).uniform_(
        cfg.EVAL.SCALE_EVAL, cfg.EVAL.SCALE_EVAL)
    set_rotate = torch.empty(total_eval, device=patch.device).uniform_(
        -cfg.EVAL.ANGLE_EVAL, cfg.EVAL.ANGLE_EVAL)
    evaluated = 0
    pbar = tqdm(desc=f'Testing epoch {e}', total=total_eval, disable=quiet)
    for img in test_loader:
        if evaluated >= total_eval:
            break
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        remaining = total_eval - evaluated
        img = img[:remaining].to(patch.device)
        batch_size = img.size(0)
        batch_start = evaluated
        h, w = img.shape[-2:]

        if cfg.ATTACKER.TYPE != "HA":
            pos = _random_positions(patch, cfg, (h, w), batch_size)
            imgn, scales = _apply_patch_batch(
                patch, img, pos, test_mode=True,
                set_resize=set_resize[batch_start:batch_start + batch_size],
                set_rotate=set_rotate[batch_start:batch_start + batch_size],
                do_random_color=True)
        else:
            pos = _random_positions(patch2, cfg, (h, w), batch_size)
            dx, dy = 0, 0
            relpos2 = (relpos[0] + dx, relpos[1] + dy)
            patch2.data = patch.apply(quick_load("assets/stop_sign.png"), relpos2, test_mode=True, do_random_color=True)
            if cfg.ATTACKER.DOUBLE_APPLY:
                dx, dy = 0, 0
                relpos2 = (relpos3[0] + dx, relpos3[1] + dy)
                patch2.data = patch.apply(patch2.data, relpos2, test_mode=True, do_random_color=True)
            imgn, scales = _apply_patch_batch(
                patch2, img, pos, test_mode=True,
                set_resize=set_resize[batch_start:batch_start + batch_size],
                set_rotate=set_rotate[batch_start:batch_start + batch_size],
                do_random_color=False)
        imgp = _apply_trigger_to_patch_aligned(
            cfg, imgn, trigger_mask, pos, patch2 if cfg.ATTACKER.TYPE == "HA" else patch)
        if cfg.ATTACKER.TYPE != "TA-C":
            pred1 = model(imgn)
            pred2 = model(imgp)
        else:
            pred1 = model(imgn)
            pred2 = model(imgp)

        if sample_index is not None and batch_start <= sample_index < batch_start + batch_size:
            sample_offset = sample_index - batch_start
            clean_pred = model(img)
            visualization_sample = {
                "clean_image": img[sample_offset].detach().cpu(),
                "attacked_image": imgp[sample_offset].detach().cpu(),
                "clean_prediction": (
                    clean_pred[sample_offset].detach().cpu()
                    if cfg.ATTACKER.TYPE != "TA-C"
                    else clean_pred[sample_offset:sample_offset + 1].detach().cpu()
                ),
                "attacked_prediction": (
                    pred2[sample_offset].detach().cpu()
                    if cfg.ATTACKER.TYPE != "TA-C"
                    else pred2[sample_offset:sample_offset + 1].detach().cpu()
                ),
            }

        box_patch = patch2 if cfg.ATTACKER.TYPE == "HA" else patch
        eval_boxes = []
        for p, s in zip(pos, scales):
            eval_boxes.append([
                p[1] + (1 - s) * box_patch.w * 0.5,
                p[0] + (1 - s) * box_patch.h * 0.5,
                p[1] + (1 + s) * box_patch.w * 0.5,
                p[0] + (1 + s) * box_patch.h * 0.5,
                patch.target,
            ])
        gt_box = torch.tensor(eval_boxes, device=patch.device)
        if cfg.ATTACKER.TYPE == "TA-D":
            eval_origin_boxes = [box[:4] + [cfg.origin_index] for box in eval_boxes]
            gt_box_origin = torch.tensor(eval_origin_boxes, device=patch.device)

        if cfg.ATTACKER.TYPE != "TA-C":
            flags1 = [isappear(pred1[k], gt_box[k:k + 1]) for k in range(batch_size)]
            flags2 = [isappear(pred2[k], gt_box[k:k + 1]) for k in range(batch_size)]
            if cfg.ATTACKER.TYPE == "TA-D":
                flags3 = [isappear(pred1[k], gt_box_origin[k:k + 1]) for k in range(batch_size)]
        else:
            flags1 = (pred1 != patch.target)
            flags2 = (pred2 == patch.target)

        if cfg.ATTACKER.TYPE == "HA":
            success += sum(flag1 and not flag2 for flag1, flag2 in zip(flags1, flags2))
            success_1 += sum(flags1)
            success_2 += sum(not flag2 for flag2 in flags2)
        elif cfg.ATTACKER.TYPE == "CA":
            success += sum((not flag1) and flag2 for flag1, flag2 in zip(flags1, flags2))
            success_1 += sum(not flag1 for flag1 in flags1)
            success_2 += sum(flags2)
        elif cfg.ATTACKER.TYPE == "TA-D":
            success += sum(flag3 and flag2 for flag3, flag2 in zip(flags3, flags2))
            success_1 += sum(flags3)
            success_2 += sum(flags2)
        elif cfg.ATTACKER.TYPE == "TA-C":
            success += int((flags1 & flags2).sum().item())
            success_1 += int(flags1.sum().item())
            success_2 += int(flags2.sum().item())

        evaluated += batch_size
        pbar.update(batch_size)
        del imgn, imgp, gt_box, pred1, pred2
        torch.cuda.empty_cache()
    pbar.close()
    if evaluated == 0:
        raise RuntimeError("Evaluation dataloader did not yield any samples.")
    total_eval = evaluated
    metrics = {
        "epoch": e,
        "samples": total_eval,
        "success": success,
        "no_triggered_success": success_1,
        "triggered_success": success_2,
        "ASR": success / total_eval,
        "No_triggered": success_1 / total_eval,
        "Triggered": success_2 / total_eval,
    }
    if not quiet:
        print(
            f"ASR: {metrics['ASR']}; "
            f"No_triggered: {metrics['No_triggered']}; "
            f"Triggered: {metrics['Triggered']}"
        )
    if capture_sample:
        return metrics, visualization_sample
    return metrics
