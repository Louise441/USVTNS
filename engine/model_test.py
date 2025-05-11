import torch
import torch.nn as nn
import os
import time

from progress.bar import Bar

def test_adaptive_memory(testloader, model, use_cuda, model_name, opt):
    data_time = AverageMeter()
    bar = Bar('Processing model testing', max=len(testloader))

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            keys_lst, vals_lst, temporal_lst, keys_dict, vals_dict, tempo_dict = [], [], [], {}, {}, {}
            keys_dict_LL, vals_dict_LL, keys_dict_HH, vals_dict_HH = {}, {}, {}, {}
            frames, masks, objs, infos = data
            t1 = time.time()
            T, _, H, W = frames.shape
            pred_lst, temp_msk_lst, ret_msk_lst = [], [], []

            for T_idx in range(int(T)):
                if T_idx == 0:
                    frame, mask = frames[T_idx:T_idx + 1], masks[T_idx:T_idx + 1]
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
                    logits, ret_tempo_confidence_map, tempo_confidence_map, ret_segmsk, temp_segmsk, wo_sc_logits = model.ddim_sample(
                        img=frame,
                        temporal_feat_lst=temporal_lst,
                        keys_lst=keys_lst,
                        vals_lst=vals_lst,
                        keys_dict=keys_dict,
                        vals_dict=vals_dict,
                        opt=opt,
                        keys_dict_LL=keys_dict_LL,
                        vals_dict_LL=vals_dict_LL,
                        keys_dict_HH=keys_dict_HH,
                        vals_dict_HH=vals_dict_HH)
                    upsampled_logits = nn.functional.interpolate(logits, scale_factor=4.0, mode="bilinear",
                                                                 align_corners=False)
                    pred = torch.sigmoid(upsampled_logits)
                    pred_lst.append(pred)


                else:
                    frame = frames[T_idx:T_idx + 1]
                    logits, ret_tempo_confidence_map, tempo_confidence_map, ret_segmsk, temp_segmsk, wo_sc_logits = model.ddim_sample(
                        img=frame,
                        temporal_feat_lst=temporal_lst,
                        keys_lst=keys_lst,
                        vals_lst=vals_lst,
                        keys_dict=keys_dict,
                        vals_dict=vals_dict,
                        opt=opt,
                        keys_dict_LL=keys_dict_LL,
                        vals_dict_LL=vals_dict_LL,
                        keys_dict_HH=keys_dict_HH,
                        vals_dict_HH=vals_dict_HH)
                    upsampled_logits = nn.functional.interpolate(logits, scale_factor=4.0, mode="bilinear",
                                                                 align_corners=False)
                    pred = torch.sigmoid(upsampled_logits)
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
            pred = torch.cat(pred_lst, dim=0)
            pred = pred.detach().cpu().numpy()

            toc = time.time() - t1
            data_time.update(toc, 1)

            bar.suffix = '({batch}/{size}) Time: {data:.3f}s'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.sum
            )
            bar.next()
        bar.finish()

    return data_time.sum


