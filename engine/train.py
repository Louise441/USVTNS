import torch
import torch.nn as nn
import time
from progress.bar import Bar



def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    data_time = AverageMeter()
    start = time.time()
    bar = Bar('Processing', max=len(trainloader))
    total_loss = 0.0
    for batch_idx, data in enumerate(trainloader):
        keys_lst, vals_lst, temporal_lst, keys_dict, vals_dict, tempo_dict = [], [], [], {}, {}, {}
        keys_dict_LL, vals_dict_LL, keys_dict_HH, vals_dict_HH = {}, {}, {}, {}
        batch_loss = 0.0
        batch_frames, batch_masks, objs, batch_detail = data
        data_time.update(time.time() - start)
        B, T, C, H, W = batch_frames.shape
        pred_lst, upsampled_logits_lst = [], []
        for T_idx in range(int(T)):
            if T_idx == 0:
                frame, mask = batch_frames[:, T_idx:T_idx + 1], batch_masks[:, T_idx:T_idx + 1]
                frame = frame.flatten(start_dim=0, end_dim=1).contiguous()
                mask = mask.flatten(start_dim=0, end_dim=1).contiguous()

                temporal_feat, memo_K, memo_V, memo_K_LL, memo_V_LL, memo_K_HH, memo_V_HH = model.memorize_frames(imgs=frame, masks=mask)
                temporal_lst.append(temporal_feat)
                keys_lst.append(memo_K)
                vals_lst.append(memo_V)
                
                tempo_dict[T_idx] = temporal_feat
                keys_dict[T_idx] = memo_K
                vals_dict[T_idx] = memo_V
                keys_dict_LL[T_idx] = memo_K_LL
                vals_dict_LL[T_idx] = memo_V_LL
                keys_dict_HH[T_idx] = memo_K_HH
                vals_dict_HH[T_idx] = memo_V_HH

                logits = model(img=frame, gt_semantic_seg=mask.long(), temporal_feat_lst=temporal_lst,
                               keys_lst=keys_lst, vals_lst=vals_lst,
                               keys_dict=keys_dict, vals_dict=vals_dict,
                               opt=opt,
                               keys_dict_LL=keys_dict_LL, vals_dict_LL=vals_dict_LL, 
                               keys_dict_HH=keys_dict_HH, vals_dict_HH=vals_dict_HH)
                upsampled_logits = nn.functional.interpolate(logits, scale_factor=4.0, mode="bilinear",
                                                             align_corners=False)
                upsampled_logits_lst.append(upsampled_logits)
                pred = torch.sigmoid(upsampled_logits.detach())
                pred_lst.append(pred)

            else:
                frame, mask = batch_frames[:, T_idx:T_idx + 1], batch_masks[:, T_idx:T_idx + 1]
                frame = frame.flatten(start_dim=0, end_dim=1).contiguous()
                mask = mask.flatten(start_dim=0, end_dim=1).contiguous()

                logits = model(img=frame, gt_semantic_seg=mask.long(), temporal_feat_lst=temporal_lst,
                               keys_lst=keys_lst, vals_lst=vals_lst,
                               keys_dict=keys_dict, vals_dict=vals_dict,
                               opt=opt,
                               keys_dict_LL=keys_dict_LL, vals_dict_LL=vals_dict_LL, 
                               keys_dict_HH=keys_dict_HH, vals_dict_HH=vals_dict_HH)
                upsampled_logits = nn.functional.interpolate(logits, scale_factor=4.0, mode="bilinear",
                                                             align_corners=False)
                upsampled_logits_lst.append(upsampled_logits)
                pred = torch.sigmoid(upsampled_logits.detach())
                pred_lst.append(pred)

                mask = pred
                temporal_feat, memo_K, memo_V, memo_K_LL, memo_V_LL, memo_K_HH, memo_V_HH = model.memorize_frames(imgs=frame, masks=mask)
                temporal_lst.append(temporal_feat)
                keys_lst.append(memo_K)
                vals_lst.append(memo_V)
                
                tempo_dict[T_idx] = temporal_feat
                keys_dict[T_idx] = memo_K
                vals_dict[T_idx] = memo_V
                keys_dict_LL[T_idx] = memo_K_LL
                vals_dict_LL[T_idx] = memo_V_LL
                keys_dict_HH[T_idx] = memo_K_HH
                vals_dict_HH[T_idx] = memo_V_HH

        upsampled_logits = torch.cat(upsampled_logits_lst, dim=1).unsqueeze(2).flatten(start_dim=0, end_dim=1)
        batch_loss = batch_loss + criterion(upsampled_logits, batch_masks.flatten(start_dim=0, end_dim=1))
        total_loss = total_loss + batch_loss

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    bar.finish()

    return total_loss.item()

