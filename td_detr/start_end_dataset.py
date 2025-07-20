from math import ceil, floor
from typing import Iterator, List
import torch
import torch.utils
from torch.utils.data import Dataset, BatchSampler, Sampler, ConcatDataset
import bisect
import numpy as np
import torch.utils.data
import torch.utils.data.dataloader
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from td_detr.span_utils import span_xx_to_cxw
import os
from torch import nn

logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 dset_domain=None):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        self.pooling = nn.AdaptiveAvgPool1d(max_v_l)
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()
        
        # load specific domain data for tvsum dataset
        if self.dset_name == 'tvsum':
            target_domain = dset_domain
            assert target_domain in ["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"]

            new_data = []
            for d in self.data:
                if target_domain == d['domain']:
                    new_data.append(d)
            self.data = new_data

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)
        if self.use_video:
            model_inputs["video_feat"], clip_len = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        model_inputs['idxs'] = torch.LongTensor([index % len(self.data)])
        if self.load_labels:
            if self.dset_name == 'tvsum': 
                model_inputs["span_labels"] = torch.tensor([[0., 0.]])
                meta_label = meta['label']
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                            self.get_saliency_labels_all_tvsum(meta_label, ctx_l)
            else:
                model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l, clip_len)  # (#windows, 2)
                
                # model_inputs["video_feat"] = self._get_video_feat_fake_by_vid(meta["vid"], meta["relevant_windows"])
                # if self.use_tef:
                #     tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
                #     tef_ed = tef_st + 1.0 / ctx_l
                #     tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
                #     if self.use_video:
                #         model_inputs["video_feat"] = torch.cat(
                #             [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
                #     else:
                #         model_inputs["video_feat"] = tef

                if "subs_train" not in self.data_path and self.dset_name == 'hl':
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
                    # model_inputs["saliency_all_labels"] = model_inputs["saliency_all_labels"][:ctx_l]
                else:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l, clip_len)  # only one gt
                    
        return meta, model_inputs

    # def _get_video_feat_fake_by_vid(self, vid, windows):
    #     v_feat_list = []
    #     for _feat_dir in self.v_feat_dirs:
    #         try:
    #             _feat_path = join(_feat_dir, f"{vid}.npz")
    #             _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
    #         except:
    #             _feat_path = join(_feat_dir, f"{vid}.npy")
    #             _feat = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
    #         max_v = _feat.max()
    #         min_v = _feat.min()
    #         for w in windows:
    #             st, ed = w
    #             st = ceil(max(0, st - 3) // 2)
    #             ed = floor(min(len(_feat), (ed + 3)  // 2))
    #             _feat[st:ed] = np.random.rand(*(_feat[st:ed].shape)) * (max_v - min_v) + min_v
    #         if self.normalize_v:
    #             _feat = l2_normalize_np_array(_feat)
    #         v_feat_list.append(_feat)
    #     # some features are slightly longer than the others
    #     min_len = min([len(e) for e in v_feat_list])
    #     v_feat_list = [e[:min_len] for e in v_feat_list]
    #     v_feat = np.concatenate(v_feat_list, axis=1)
    #     return torch.from_numpy(v_feat)  # (Lv, D)

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, clip_len, max_n=2):
        gt_st = int(gt_window[0] / clip_len)
        gt_ed = max(0, min(int(gt_window[1] / clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed+1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))
        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except:
            neg_clip_indices = pos_clip_indices
        # return pos_clip_indices, neg_clip_indices
        
        score_array = np.zeros(ctx_l)
        score_array[gt_st:gt_ed+1] = 1

        return pos_clip_indices, neg_clip_indices, score_array
        

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # score_array = [min(agg_scores[idx], ctx_l-1) for idx in range(ctx_l)]
        score_array = np.zeros(ctx_l)
        for idx in range(len(rel_clip_ids)):
            if rel_clip_ids[idx] >= ctx_l:
                score_array_new = np.zeros(ctx_l + 1)
                score_array_new[:ctx_l] = score_array
                score_array = score_array_new
            # if rel_clip_ids[idx] == ctx_l:
            #     print(rel_clip_ids[idx], ctx_l)
            score_array[rel_clip_ids[idx]] = agg_scores[idx]

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_tvsum(self, labels, ctx_l, max_n=1, add_easy_negative=False):
        
        agg_scores = np.sum(labels - np.ones_like(labels), axis=-1)[:ctx_l] # start from 1, so minus 1
        score_array = agg_scores / 80 * 12
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array

    def get_span_labels(self, windows, ctx_l, clip_len):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / clip_len), min(int(w[1] / clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        if (not isinstance(qid, int)) and '_' in qid and self.dset_name != 'tacos':
            qid = int(qid.split('_')[0])

        if self.dset_name == 'tvsum':
            q_feat = np.load(join(self.q_feat_dir, "{}.npz".format(qid))) # 'token', 'text'
            return torch.from_numpy(q_feat['token'])
        elif self.dset_name in ['hl', 'charades'] and 'intern' in self.q_feat_dir:
            try:
                q_feat = torch.load(join(self.q_feat_dir, f"qid{qid}.pt")).numpy().astype(np.float32)
            except:
                print(f'qid: {qid}')
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
            if self.txt_drop_ratio > 0:
                q_feat = self.random_drop_rows(q_feat)
        else:
            # QVhighlight dataset
            if self.dset_name in ['tacos', 'ego4d']:
                q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            else:
                q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
            if self.txt_drop_ratio > 0:
                q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid, down_sample=True):
        if self.dset_name == 'tvsum':
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                _feat_path = join(_feat_dir, f"{vid}_rgb.npy")
                _feat_rgb = np.load(_feat_path).astype(np.float32)

                _feat_path = join(_feat_dir, f"{vid}_opt.npy")
                _feat_opt = np.load(_feat_path).astype(np.float32)
                
                _feat = np.concatenate([_feat_rgb, _feat_opt], axis=-1)
                # _feat = _feat_rgb
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
        elif self.dset_name == 'charades' and ('i3d' or 'vgg') in self.v_feat_dirs[0]:
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                _feat_path = join(_feat_dir, f"{vid}.npy")
                _feat = np.load(_feat_path).astype(np.float32)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
        elif self.dset_name in ['hl', 'charades'] and 'intern' in self.q_feat_dir:
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                _feat_path = join(_feat_dir, f"{vid}.pt")
                try:
                    _feat = torch.load(_feat_path).numpy().astype(np.float32)
                except FileNotFoundError:
                    _feat = torch.zeros((self.max_v_l, 768), dtype=torch.float32)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
        else:
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                _feat_path = join(_feat_dir, f"{vid}.npz")
                _feat = np.load(_feat_path)["features"].astype(np.float32)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
        
        v_feat = torch.from_numpy(v_feat)
        clip_len = self.clip_len
        if v_feat.shape[0] > self.max_v_l and down_sample and self.dset_name != 'hl':
            # print(f'before v_feat.shape: {v_feat.shape}')
            v_feat, clip_len = self.down_sampling(v_feat, clip_len)
            # print(f'after v_feat.shape: {v_feat.shape}')
        return v_feat, clip_len  # (Lv, D)

    def down_sampling(self, v_feat, clip_len_ori):
        clip_len = clip_len_ori * v_feat.shape[0] / self.max_v_l
        v_feat = v_feat.transpose(0, 1).unsqueeze(0)
        v_feat = self.pooling(v_feat)
        v_feat = v_feat.squeeze(0).transpose(0, 1)
        return v_feat, clip_len


class SimSampler(BatchSampler):
    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool, sim_matrix, sample_strategy) -> None:
        super().__init__(sampler, batch_size, drop_last)
        sim = torch.load(sim_matrix)
        self.dset_name = sampler.data_source.datasets[0].dset_name
        self.sim = torch.zeros(len(sampler), len(sampler))
        self.sim[:sim.size(0), :sim.size(1)] = sim
        self.sample_strategy = sample_strategy

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                indices = torch.LongTensor(batch)
                sim_idx = self.get_sim_indices(indices)
                # import pdb; pdb.set_trace()
                batch = indices[sim_idx].tolist()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            indices = torch.LongTensor(batch)
            sim_idx = self.get_sim_indices(indices)
            # batch = torch.cat([indices.unsqueeze(1), indices[sim_idx].unsqueeze(1)], dim=1).tolist()
            batch = indices[sim_idx].tolist()
            yield batch

    # def __iter__(self):
    #     n = len(self.data_source)
    #     if self.generator is None:
    #         generator = torch.Generator()
    #         generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    #     else:
    #         generator = self.generator
    #     indices = torch.randperm(n, generator=generator).unsqueeze(1)
    #     sim_idx = self.get_sim_indices(indices)
    #     idx = torch.stack([indices, sim_idx], axis=1)
    #     yield from idx.tolist()

    def get_sim_indices(self, idx):
        bsz = len(idx)
        batch_sim = self.sim[idx, :][:, idx]

        # batch_sim = torch.tril(batch_sim, diagonal=-1)
        # flat_matrix = batch_sim.view(-1)
        mask = torch.diag(torch.ones(bsz))
        batch_sim = batch_sim * (~mask.bool()).float()

        if self.sample_strategy == 'sim':
            if self.dset_name != 'hl':
                batch_sim[batch_sim>0.9] = 0
            # batch_sim[batch_sim>0.95] = 0
            selected_indices = torch.argmax(batch_sim, dim=-1)
        elif self.sample_strategy == 'random':
            selected_indices = torch.randint(0, bsz, (bsz,))
        elif self.sample_strategy == 'dissim':
            # batch_sim[batch_sim<0.05] = 0
            selected_indices = torch.argmin(batch_sim, dim=-1)
        elif self.sample_strategy == 'mix':
            topk_indices = torch.argmax(batch_sim, dim=-1)
            bottomk_indices = torch.argmin(batch_sim, dim=-1)
            random_mask = torch.rand_like(topk_indices, dtype=torch.float) > 0.5
            selected_indices = torch.where(random_mask, topk_indices, bottomk_indices)
        else:
            raise NotImplementedError(f"Unknown sample strategy: {self.sample_strategy}")
        # topk_indices = list(range(bsz))
        # random.shuffle(topk_indices)
        # topk_indices = torch.LongTensor(topk_indices)
        # batch_sim[batch_sim<=0.01] = 1e10
        # bottomk_indices = torch.argmin(batch_sim, dim=-1)
        # indices = torch.cat([topk_indices, bottomk_indices], dim=0)
        # weights = torch.ones(len(indices))
        # weights[bsz:] = 0.5
        # selected_indices = torch.multinomial(weights.float(), bsz)
        # indices_2d = indices[selected_indices]
        # indices_2d = topk_indices

        return torch.cat([torch.arange(len(selected_indices)).unsqueeze(1), selected_indices.unsqueeze(1)], dim=-1)


class SimilarityConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(SimilarityConcatDataset, self).__init__(datasets)

    def __getitem__(self, index):
        if not isinstance(index, int):
            idx = index[0]
        else:
            logger.warning("index: {}".format(index))
            idx = index
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class SyntheticDataset(StartEndDataset):
    # 修改
    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir, 
                 q_feat_type="last_hidden_state", 
                 max_q_l=32, max_v_l=75, data_ratio=1, 
                 ctx_mode="video", normalize_v=True, normalize_t=True, 
                 load_labels=True, clip_len=2, max_windows=5, span_loss_type="l1", 
                 txt_drop_ratio=0, dset_domain=None, alpha=0.5):
        super().__init__(dset_name, data_path, v_feat_dirs, q_feat_dir, q_feat_type, max_q_l, max_v_l, data_ratio, ctx_mode, normalize_v, normalize_t, load_labels, clip_len, max_windows, span_loss_type, txt_drop_ratio, dset_domain) 
        self.alpha = alpha

        video_feature = []
        if not os.path.exists(f'similarity_matrix/{self.dset_name}_similarity_matrix_top50.pt'):
            vid = [d['vid'] for d in self.data]
            for i, v in enumerate(tqdm(vid)):
                v_feat, _ = self._get_video_feat_by_vid(v)
                video_feature.append(v_feat)
            # import pdb; pdb.set_trace()
            video_feature, _ = pad_sequences_1d(video_feature, dtype=torch.float32)
            
            print("Computing similarity matrix...")
            batch_size = 35  # 你可以根据需要调整这个值
            num_videos = video_feature.shape[0]

            sim_matrix = torch.zeros((num_videos, num_videos))
            for i in tqdm(range(0, num_videos, batch_size)):
                end_i = min(i + batch_size, num_videos)
                video_feature_batch_i = video_feature[i:end_i].unsqueeze(1)  # Shape: (batch_size, 1, Lv, D)
                # Compute cosine similarity
                # memory_size = video_feature_batch_i.element_size() * video_feature_batch_i.numel()
                # print(f'memory_size: {memory_size / 1024 / 1024 / 1024} GB')
                for j in range(0, num_videos, batch_size):
                    end_j = min(j + batch_size, num_videos)
                    video_feature_batch_j = video_feature[j:end_j].transpose(1, 2)  # Shape: (num_videos, D, Lv)
                    sim = torch.matmul(video_feature_batch_i.cuda(), video_feature_batch_j.cuda()) / 2  # Shape: (batch_size, batch_size, Lv, Lv)
                    
                    # Average over the last dimension
                    top = 50
                    non_zero_mask = sim != 0
                    non_zero_count = non_zero_mask.sum(dim=(2, 3))
                    sim = sim.flatten(start_dim=2)
                    sim = torch.topk(sim, top, dim=2).values
                    non_zero_count[non_zero_count>top] = top
                    sim = sim.sum(dim=2) / top
                    sim_matrix[i:end_i, j:end_j] = sim.cpu()  # Shape: (batch_size, num_videos)
            
            torch.save(sim_matrix, f'similarity_matrix/{self.dset_name}_similarity_matrix_top50.pt')
            del video_feature, video_feature_batch_i, video_feature_batch_j, sim
            print("Done. Similarity matrix shape:", sim_matrix.shape)
            del sim_matrix

    def __getitem__(self, index):
        
        batched_model_inputs = {}
        max_len = len(self.data) - 1
        idxs = [min(index[0], max_len), min(index[1], max_len)]

        video_feat = []
        all_indices = []
        gt_indices = []
        # if self.data[int(idxs[0])]["duration"] > 410 and self.data[int(idxs[1])]["duration"] < 390:
        #     import pdb; pdb.set_trace()
        for i in range(0, 2):
            # 1. new video feature
            new_meta = self.data[int(idxs[i])]
            # import pdb; pdb.set_trace()
            new_video_feature, _ = self._get_video_feat_by_vid(new_meta["vid"], down_sample=False)
            relevant_windows = new_meta["relevant_windows"]
            indices = []
            c = len(new_video_feature)
            for r in relevant_windows:
                indices.extend(range(min(max(0, int(floor(r[0]) / self.clip_len)), c-1), min(int(ceil(r[1]) / self.clip_len), c)))
            if len(indices) == 0:
                # print(f"No relevant windows found: {relevant_windows}, {new_meta['vid']}, {max(0, floor(r[0])//self.clip_len)}, {min(ceil(r[1])//self.clip_len, c)}, {c}, {new_video_feature.shape}")
                indices = [min(max(0, int(floor(r[0]) / self.clip_len)), c-1)]
                # raise ValueError
            gt = list(range(min(indices), max(indices) + 1))
            gt_indices.append(gt)
            all_indices.append(list(set(range(c)) - set(gt)))
            video_feat.append(new_video_feature)
        
        # if len(set(range(c)) - set(indices)) == 0:
        #     print(f'{new_meta["relevant_windows"]}')
        
        concat_v_feat = []
        pos_bias = []
        clip = []
        for i in range(0, 2):
            another_i = 1 - i
            another_len = video_feat[another_i].shape[0]
            if int(len(all_indices[i]) * self.alpha) > 2:
                idx_now_video = random.sample(all_indices[i], int(len(all_indices[i]) * self.alpha))
            else:
                idx_now_video = []
            if int(another_len * (1 - self.alpha)) > 2:
                idx_another_video = random.sample(list(range(another_len)), int(another_len * (1 - self.alpha)))
            else:
                idx_another_video = []

            idx_now_video = sorted(idx_now_video)
            idx_another_video = sorted(idx_another_video)

            #########################
            # idx_now_video = idx_now_video + gt_indices[i]
            # idx_now_video = sorted(idx_now_video)
            # idx_another_video = sorted(idx_another_video)
            #########################

            drop_rate = 1.2 - random.random() * 0.4
            
            now_sample = len(idx_now_video) * drop_rate
            if len(idx_now_video) < now_sample:
                while len(idx_now_video) < now_sample:
                    now_sample -= len(idx_now_video)
                    idx_now_video.extend(random.sample(idx_now_video, int(now_sample)))
            else:
                idx_now_video = random.sample(idx_now_video, int(len(idx_now_video) * drop_rate))
            idx_now_video = idx_now_video + gt_indices[i]
            idx_now_video = sorted(idx_now_video)

            drop_rate = 1.2 - random.random() * 0.4
            another_sample = len(idx_another_video) * drop_rate
            if len(idx_another_video) < another_sample:
                while len(idx_another_video) < another_sample:
                    another_sample -= len(idx_another_video)
                    idx_another_video.extend(random.sample(idx_another_video, int(another_sample)))
            else:
                idx_another_video = random.sample(idx_another_video, int(len(idx_another_video) * drop_rate))        
            idx_another_video = sorted(idx_another_video)

            st_idxs = min(gt_indices[i])
            position = idx_now_video.index(st_idxs)
            pos_bias.append(st_idxs - position)

            if i == 0:
                concat_v_feat.append(torch.cat([video_feat[i][idx_now_video], video_feat[another_i][idx_another_video]], dim=0))
            else:
                concat_v_feat.append(torch.cat([video_feat[another_i][idx_another_video], video_feat[i][idx_now_video]], dim=0))

            clip.append(len(video_feat[another_i][idx_another_video]))

        ctx_l = [len(v) for v in concat_v_feat]

        # import pdb; pdb.set_trace()
        for i in range(0, 2):
            c = ctx_l[i]
            if self.use_tef:
                tef_st = torch.arange(0, c, 1.0) / c
                tef_ed = tef_st + 1.0 / c
                tef = torch.stack([tef_st, tef_ed], dim=1)
                if self.use_video:
                    concat_v_feat[i] = torch.cat(
                        [concat_v_feat[i], tef], dim=1)  # (Lv, Dv+2)
                else:
                    concat_v_feat[i] = tef 

        c = 0
        meta = {}
        for i in range(0, 2):
            if i == 1:
                c += clip[i]
            tmp = {}
            new_meta = self.data[int(idxs[i])]
            
            tmp_meta = {
                "qid": new_meta["qid"],
                "query": new_meta["query"],
                "ori_duration": new_meta["duration"],
                "duration": ctx_l[i] * self.clip_len,
                "vid": new_meta["vid"],
                "ori_windows": new_meta["relevant_windows"],
                "relevant_windows": (np.array(new_meta["relevant_windows"]) + (c - pos_bias[i]) * self.clip_len).tolist(),
                # "relevant_clip_ids": (torch.tensor(new_meta["relevant_clip_ids"]) + c - pos_bias[i]).tolist(),
                # "saliency_scores": new_meta["saliency_scores"]
            }

            if self.dset_name == 'hl':
                tmp_meta["relevant_clip_ids"] = (torch.tensor(new_meta["relevant_clip_ids"]) + c - pos_bias[i]).tolist()
                tmp_meta["saliency_scores"] = new_meta["saliency_scores"]

            clip_len = self.clip_len
            if concat_v_feat[i].shape[0] > self.max_v_l and self.dset_name != 'hl':
                concat_v_feat[i], clip_len = self.down_sampling(concat_v_feat[i], self.clip_len)
                ctx_l[i] = len(concat_v_feat[i])

            # load labels
            if self.load_labels:
                if self.dset_name == 'tvsum':
                    tmp["span_labels"] = torch.tensor([[0., 0.]])
                    l = torch.tensor(new_meta['label']) + c - pos_bias[i]
                    tmp["saliency_pos_labels"], tmp["saliency_neg_labels"], tmp["saliency_all_labels"] = \
                          self.get_saliency_labels_all_tvsum(l.tolist())
                else:

                    # for s in tmp_meta["relevant_windows"]:
                    #     if s[1] - tmp_meta["duration"] > 3:
                    #         import pdb; pdb.set_trace()
                    tmp["span_labels"] = self.get_span_labels(tmp_meta["relevant_windows"], ctx_l[i], clip_len)  # (#windows, 2)
                    if "subs_train" not in self.data_path and self.dset_name == 'hl': 
                        ss = torch.tensor(new_meta["saliency_scores"])
                        tmp["saliency_pos_labels"], tmp["saliency_neg_labels"], tmp["saliency_all_labels"] = \
                            self.get_saliency_labels_all(tmp_meta["relevant_clip_ids"], ss, ctx_l[i])
                        tmp["saliency_all_labels"] = tmp["saliency_all_labels"][:ctx_l[i]]
                    else:
                        tmp["saliency_pos_labels"], tmp["saliency_neg_labels"], tmp["saliency_all_labels"] = \
                            self.get_saliency_labels_sub_as_query(tmp_meta["relevant_windows"][0], ctx_l[i], clip_len)  # only one gt
                        tmp["saliency_all_labels"] = tmp["saliency_all_labels"][:ctx_l[i]]
            
            # load query feature
            tmp["query_feat"] = self._get_query_feat_by_qid(new_meta["qid"])  # (Dq, ) or (Lq, Dq)
            tmp["video_feat"] = concat_v_feat[i]  # (Lv, Dv)
            tmp['idxs'] = torch.LongTensor([idxs[i] % len(self.data)])
            # Add batch
            batched_model_inputs[f'batch_{i}'] = tmp
            meta[f'batch_{i}'] = tmp_meta
        return meta, batched_model_inputs


def start_end_collate(batch):
    batch_meta = [e[0] for e in batch]

    new_meta = []
    # import pdb; pdb.set_trace()
    for e in batch_meta:
        if 'batch_0' in e:
            for k in e.keys():
                new_meta.append(e[k])
        else:
            new_meta.append(e)


    batch = [e[1] for e in batch]

    new_batch = []
    for e in batch:
        if 'batch_0' in e:
            for k in e.keys():
                new_batch.append(e[k])
        else:
            new_batch.append(e)
    batched_data = start_end_collate_helper(new_batch)
    # import pdb; pdb.set_trace()
    return new_meta, batched_data


def start_end_collate_helper(batch):
    model_inputs_keys = batch[0].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e[k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e[k] for e in batch], dtype=np.float32, fixed_length=None)
            # print(pad_data, mask_data)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue
        if k == 'idxs':
            batched_data[k] = torch.LongTensor([e[k] for e in batch])
            continue

        batched_data[k] = pad_sequences_1d(
            [e[k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
    )
    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)

    if 'idxs' in batched_model_inputs:
        targets['idxs'] = batched_model_inputs['idxs'].to(device, non_blocking=non_blocking)

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
