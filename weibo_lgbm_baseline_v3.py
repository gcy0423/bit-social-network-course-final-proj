#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import hashlib
import json
import os
import re
import warnings
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("lightgbm 未安装。请先执行: pip install lightgbm") from exc

ENCODINGS = ["utf-8", "utf-8-sig", "gb18030", "latin1"]
TARGETS = ["forward_count", "comment_count", "like_count"]
TRAIN_COLUMNS = ["uid", "mid", "time", "forward_count", "comment_count", "like_count", "content"]
TEST_COLUMNS = ["uid", "mid", "time", "content"]
DEFAULT_BLEND = {"forward_count": 0.85, "comment_count": 0.85, "like_count": 0.85}
DEFAULT_FULL_ITERS = {"forward_count": 900, "comment_count": 900, "like_count": 900}

TRAIN_LINE_PATTERN = re.compile(
    r"^(\S+)\s+(\S+)\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*(.*)$"
)
TEST_LINE_PATTERN = re.compile(
    r"^(\S+)\s+(\S+)\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s*(.*)$"
)


@dataclass
class FeatureBundle:
    numeric_df: pd.DataFrame
    feature_cols: List[str]
    fill_values: Dict[str, float]
    history_cache: Dict[str, dict]
    global_defaults: Dict[str, float]
    recent_ks: List[int]
    recent_days: List[int]


class SentenceEmbeddingEncoder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 128,
        normalize_embeddings: bool = True,
        max_chars: int = 256,
        cache_dir: Optional[str] = None,
        text_prefix: str = "",
        max_seq_length: Optional[int] = None,
    ):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "使用预训练句向量需要安装 sentence-transformers 和 torch。\n"
                "请先执行: pip install sentence-transformers torch"
            ) from exc

        if device in (None, "", "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.max_chars = int(max_chars)
        self.cache_dir = cache_dir
        self.text_prefix = text_prefix or ""
        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length is not None and hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = int(max_seq_length)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _normalize_text(text: str, max_chars: int, text_prefix: str = "") -> str:
        text = str(text)
        text = re.sub(r"http[s]?://\S+|t\.cn/\S+", " [URL] ", text)
        text = re.sub(r"@[A-Za-z0-9_\-\u4e00-\u9fff]+", " @用户 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if max_chars > 0:
            text = text[:max_chars]
        if not text:
            text = "[EMPTY]"
        if text_prefix:
            text = f"{text_prefix}{text}"
        return text

    def _cache_path(self, texts: pd.Series, split_name: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        md5 = hashlib.md5()
        md5.update(self.model_name.encode("utf-8", errors="ignore"))
        md5.update(str(self.normalize_embeddings).encode())
        md5.update(str(self.max_chars).encode())
        md5.update(str(len(texts)).encode())
        for x in texts.astype(str).tolist():
            norm_x = self._normalize_text(x, self.max_chars, self.text_prefix)
            md5.update(norm_x.encode("utf-8", errors="ignore"))
            md5.update(b"\0")
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_name)
        fname = f"{split_name}_{safe_model}_{md5.hexdigest()[:16]}.npy"
        return os.path.join(self.cache_dir, fname)

    def encode(self, texts: pd.Series, split_name: str = "data") -> np.ndarray:
        cache_path = self._cache_path(texts, split_name)
        if cache_path and os.path.exists(cache_path):
            emb = np.load(cache_path)
            return emb.astype(np.float32)

        text_list = [self._normalize_text(x, self.max_chars, self.text_prefix) for x in texts.astype(str).tolist()]
        emb = self.model.encode(
            text_list,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        emb = np.asarray(emb, dtype=np.float32)
        if cache_path:
            np.save(cache_path, emb)
        return emb


# ------------------------- 基础工具 -------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def detect_tab_separated(path: str) -> bool:
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    return "\t" in line
        except Exception:
            continue
    return False


def read_with_pandas_tsv(path: str, is_train: bool) -> pd.DataFrame:
    names = TRAIN_COLUMNS if is_train else TEST_COLUMNS
    last_err = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                header=None,
                names=names,
                engine="python",
                encoding=enc,
                quoting=csv.QUOTE_NONE,
                na_filter=False,
                keep_default_na=False,
            )
            if is_train and df.shape[1] >= 7:
                for col in TARGETS:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int32)
            return df
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"无法以 TSV 方式读取文件: {path}. 最后错误: {last_err}")


def read_with_fallback_parser(path: str, is_train: bool) -> pd.DataFrame:
    last_err = None
    rows = []
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    if is_train:
                        m = TRAIN_LINE_PATTERN.match(line)
                        if not m:
                            raise ValueError(f"第 {lineno} 行无法解析: {line[:120]}")
                        uid, mid, d, t, fwd, cmt, like, content = m.groups()
                        rows.append({
                            "uid": uid,
                            "mid": mid,
                            "time": f"{d} {t}",
                            "forward_count": int(fwd),
                            "comment_count": int(cmt),
                            "like_count": int(like),
                            "content": content or "",
                        })
                    else:
                        m = TEST_LINE_PATTERN.match(line)
                        if not m:
                            raise ValueError(f"第 {lineno} 行无法解析: {line[:120]}")
                        uid, mid, d, t, content = m.groups()
                        rows.append({
                            "uid": uid,
                            "mid": mid,
                            "time": f"{d} {t}",
                            "content": content or "",
                        })
            return pd.DataFrame(rows)
        except Exception as exc:
            last_err = exc
            rows = []
    raise RuntimeError(f"无法按空白分隔回退解析文件: {path}. 最后错误: {last_err}")


def read_data(path: str, is_train: bool) -> pd.DataFrame:
    if detect_tab_separated(path):
        df = read_with_pandas_tsv(path, is_train=is_train)
    else:
        df = read_with_fallback_parser(path, is_train=is_train)

    df["uid"] = df["uid"].astype(str)
    df["mid"] = df["mid"].astype(str)
    df["content"] = df["content"].astype(str)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError(f"time 字段有 {int(df['time'].isna().sum())} 行无法解析，请检查原始数据格式。")
    if is_train:
        for col in TARGETS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).astype(np.int32)
    return df


# ------------------------- 基础特征 -------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    dt = df["time"]
    out["hour"] = dt.dt.hour.astype(np.int16)
    out["weekday"] = dt.dt.weekday.astype(np.int16)
    out["day"] = dt.dt.day.astype(np.int16)
    out["month"] = dt.dt.month.astype(np.int16)
    out["is_weekend"] = (dt.dt.weekday >= 5).astype(np.int8)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0).astype(np.float32)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0).astype(np.float32)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7.0).astype(np.float32)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7.0).astype(np.float32)
    return out


def add_content_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    texts = df["content"].fillna("").astype(str)
    out = pd.DataFrame(index=df.index)
    out["text_len"] = texts.str.len().astype(np.int32)
    out["digit_cnt"] = texts.str.count(r"\d").astype(np.int16)
    out["url_cnt"] = texts.str.count(r"http[s]?://|t\.cn/").astype(np.int16)
    out["mention_cnt"] = texts.str.count(r"@").astype(np.int16)
    out["hashtag_cnt"] = texts.str.count(r"#").floordiv(2).astype(np.int16)
    out["exclam_cnt"] = texts.str.count(r"[!！]").astype(np.int16)
    out["question_cnt"] = texts.str.count(r"[?？]").astype(np.int16)
    out["weibo_emoticon_cnt"] = texts.str.count(r"\[[^\]]+\]").astype(np.int16)
    out["punct_cnt"] = texts.str.count(r"[,.，。!！?？:：;；…~～-]").astype(np.int16)
    out["cjk_char_cnt"] = texts.str.count(r"[\u4e00-\u9fff]").astype(np.int32)
    out["alpha_cnt"] = texts.str.count(r"[A-Za-z]").astype(np.int32)
    out["space_cnt"] = texts.str.count(r"\s").astype(np.int16)

    unique_char = texts.apply(lambda s: len(set(s)) if s else 0).astype(np.int32)
    text_len = out["text_len"].replace(0, 1)
    out["unique_char_ratio"] = (unique_char / text_len).astype(np.float32)
    out["url_ratio"] = (out["url_cnt"] / text_len).astype(np.float32)
    out["mention_ratio"] = (out["mention_cnt"] / text_len).astype(np.float32)

    keyword_map = {
        "has_link": r"http[s]?://|t\.cn/",
        "has_redpacket": r"红包",
        "has_lottery": r"抽奖|转发有奖",
        "has_share": r"分享|分享自|发表了博文|我分享了",
        "has_zhihu": r"知乎",
        "has_music": r"网易云音乐|唱吧|单曲|歌单",
        "has_finance": r"股票|财经|理财|投资",
        "has_tech": r"机器学习|Python|Java|Linux|Systemd|开源|大数据|Azure|Emacs|算法|模型",
        "has_photo": r"图片|美拍|拍摄",
        "has_topic": r"#.+?#",
    }
    for feat, patt in keyword_map.items():
        out[feat] = texts.str.contains(patt, regex=True).astype(np.int8)
    return out


# ------------------------- 历史统计 v2 -------------------------

def _recent_stats_from_items(items: List[Tuple[int, int, int, int, int, int]], recent_ks: List[int]) -> Dict[str, float]:
    feats = {}
    n = len(items)
    for k in recent_ks:
        sub = items[-k:] if n > 0 else []
        if not sub:
            feats[f"forward_count_recent{k}_mean"] = 0.0
            feats[f"comment_count_recent{k}_mean"] = 0.0
            feats[f"like_count_recent{k}_mean"] = 0.0
            feats[f"total_count_recent{k}_mean"] = 0.0
            feats[f"total_count_recent{k}_max"] = 0.0
            feats[f"zero_ratio_recent{k}"] = 0.0
            continue
        arr = np.asarray(sub, dtype=np.float32)
        feats[f"forward_count_recent{k}_mean"] = float(arr[:, 1].mean())
        feats[f"comment_count_recent{k}_mean"] = float(arr[:, 2].mean())
        feats[f"like_count_recent{k}_mean"] = float(arr[:, 3].mean())
        feats[f"total_count_recent{k}_mean"] = float(arr[:, 4].mean())
        feats[f"total_count_recent{k}_max"] = float(arr[:, 4].max())
        feats[f"zero_ratio_recent{k}"] = float(arr[:, 5].mean())
    return feats


def _window_stats_from_items(items: List[Tuple[int, int, int, int, int, int]], current_ts: int, recent_days: List[int]) -> Dict[str, float]:
    feats = {}
    if not items:
        for d in recent_days:
            feats[f"user_posts_last{d}d"] = 0.0
            feats[f"total_count_last{d}d_mean"] = 0.0
            feats[f"total_count_last{d}d_max"] = 0.0
            feats[f"zero_ratio_last{d}d"] = 0.0
        return feats

    for d in recent_days:
        lower = current_ts - d * 86400
        sub = [x for x in items if x[0] >= lower]
        if not sub:
            feats[f"user_posts_last{d}d"] = 0.0
            feats[f"total_count_last{d}d_mean"] = 0.0
            feats[f"total_count_last{d}d_max"] = 0.0
            feats[f"zero_ratio_last{d}d"] = 0.0
            continue
        arr = np.asarray(sub, dtype=np.float32)
        feats[f"user_posts_last{d}d"] = float(len(sub))
        feats[f"total_count_last{d}d_mean"] = float(arr[:, 4].mean())
        feats[f"total_count_last{d}d_max"] = float(arr[:, 4].max())
        feats[f"zero_ratio_last{d}d"] = float(arr[:, 5].mean())
    return feats


def add_cumulative_user_features_v2(train_df: pd.DataFrame, recent_ks: List[int], recent_days: List[int]) -> pd.DataFrame:
    df = train_df.sort_values(["time", "uid", "mid"]).reset_index(drop=True).copy()
    max_recent_k = max(recent_ks)
    max_recent_day = max(recent_days)

    time_feat = add_time_features(df)
    text_feat = add_content_surface_features(df)

    total_arr = (df["forward_count"] + df["comment_count"] + df["like_count"]).astype(np.int32).values
    ts_arr = (df["time"].astype("int64") // 10**9).values.astype(np.int64)
    f_arr = df["forward_count"].values.astype(np.int32)
    c_arr = df["comment_count"].values.astype(np.int32)
    l_arr = df["like_count"].values.astype(np.int32)

    feat_rows = []
    state_map: Dict[str, dict] = {}

    for i, row in df.iterrows():
        uid = row["uid"]
        cur_ts = int(ts_arr[i])
        cur_f = int(f_arr[i])
        cur_c = int(c_arr[i])
        cur_l = int(l_arr[i])
        cur_total = int(total_arr[i])
        cur_zero = int(cur_total == 0)

        if uid not in state_map:
            state_map[uid] = {
                "count": 0,
                "zero_count": 0,
                "sum_forward": 0.0,
                "sum_comment": 0.0,
                "sum_like": 0.0,
                "sum_total": 0.0,
                "max_forward": 0.0,
                "max_comment": 0.0,
                "max_like": 0.0,
                "max_total": 0.0,
                "first_ts": cur_ts,
                "last_ts": None,
                "last_f": 0.0,
                "last_c": 0.0,
                "last_l": 0.0,
                "last_total": 0.0,
                "last2_f": 0.0,
                "last2_c": 0.0,
                "last2_l": 0.0,
                "last2_total": 0.0,
                "recent": deque(maxlen=max_recent_k),
                "window": deque(),
            }
        st = state_map[uid]

        while st["window"] and st["window"][0][0] < cur_ts - max_recent_day * 86400:
            st["window"].popleft()

        cnt = st["count"]
        if cnt == 0:
            base = {
                "user_prev_post_count": 0.0,
                "user_days_since_prev_post": -1.0,
                "user_days_since_first_post": 0.0,
                "user_prev_zero_ratio": 0.0,
                "forward_count_prev_mean": 0.0,
                "forward_count_prev_max": 0.0,
                "forward_count_prev_last1": 0.0,
                "forward_count_prev_last2": 0.0,
                "comment_count_prev_mean": 0.0,
                "comment_count_prev_max": 0.0,
                "comment_count_prev_last1": 0.0,
                "comment_count_prev_last2": 0.0,
                "like_count_prev_mean": 0.0,
                "like_count_prev_max": 0.0,
                "like_count_prev_last1": 0.0,
                "like_count_prev_last2": 0.0,
                "total_count_prev_mean": 0.0,
                "total_count_prev_max": 0.0,
                "total_count_prev_last1": 0.0,
                "total_count_prev_last2": 0.0,
                "forward_share_prev_mean": 0.0,
                "comment_share_prev_mean": 0.0,
                "like_share_prev_mean": 0.0,
                "user_prev_posts_per_day": 0.0,
            }
        else:
            days_since_prev = (cur_ts - st["last_ts"]) / 86400.0 if st["last_ts"] is not None else -1.0
            days_since_first = (cur_ts - st["first_ts"]) / 86400.0
            total_mean = st["sum_total"] / cnt
            base = {
                "user_prev_post_count": float(cnt),
                "user_days_since_prev_post": float(days_since_prev),
                "user_days_since_first_post": float(days_since_first),
                "user_prev_zero_ratio": float(st["zero_count"] / cnt),
                "forward_count_prev_mean": float(st["sum_forward"] / cnt),
                "forward_count_prev_max": float(st["max_forward"]),
                "forward_count_prev_last1": float(st["last_f"]),
                "forward_count_prev_last2": float(st["last2_f"]),
                "comment_count_prev_mean": float(st["sum_comment"] / cnt),
                "comment_count_prev_max": float(st["max_comment"]),
                "comment_count_prev_last1": float(st["last_c"]),
                "comment_count_prev_last2": float(st["last2_c"]),
                "like_count_prev_mean": float(st["sum_like"] / cnt),
                "like_count_prev_max": float(st["max_like"]),
                "like_count_prev_last1": float(st["last_l"]),
                "like_count_prev_last2": float(st["last2_l"]),
                "total_count_prev_mean": float(total_mean),
                "total_count_prev_max": float(st["max_total"]),
                "total_count_prev_last1": float(st["last_total"]),
                "total_count_prev_last2": float(st["last2_total"]),
                "forward_share_prev_mean": float((st["sum_forward"] / cnt) / (total_mean + 1e-3)),
                "comment_share_prev_mean": float((st["sum_comment"] / cnt) / (total_mean + 1e-3)),
                "like_share_prev_mean": float((st["sum_like"] / cnt) / (total_mean + 1e-3)),
                "user_prev_posts_per_day": float(cnt / (max(days_since_first, 0.0) + 1.0)),
            }

        recent_feats = _recent_stats_from_items(list(st["recent"]), recent_ks)
        window_feats = _window_stats_from_items(list(st["window"]), cur_ts, recent_days)
        feat_rows.append({**base, **recent_feats, **window_feats})

        st["count"] += 1
        st["zero_count"] += cur_zero
        st["sum_forward"] += cur_f
        st["sum_comment"] += cur_c
        st["sum_like"] += cur_l
        st["sum_total"] += cur_total
        st["max_forward"] = max(st["max_forward"], cur_f)
        st["max_comment"] = max(st["max_comment"], cur_c)
        st["max_like"] = max(st["max_like"], cur_l)
        st["max_total"] = max(st["max_total"], cur_total)
        st["last2_f"] = st["last_f"]
        st["last2_c"] = st["last_c"]
        st["last2_l"] = st["last_l"]
        st["last2_total"] = st["last_total"]
        st["last_f"] = float(cur_f)
        st["last_c"] = float(cur_c)
        st["last_l"] = float(cur_l)
        st["last_total"] = float(cur_total)
        st["last_ts"] = cur_ts
        st["recent"].append((cur_ts, cur_f, cur_c, cur_l, cur_total, cur_zero))
        st["window"].append((cur_ts, cur_f, cur_c, cur_l, cur_total, cur_zero))

        if (i + 1) % 100000 == 0:
            print(f"  已处理训练历史特征: {i + 1:,} 行")

    hist_feat = pd.DataFrame(feat_rows, index=df.index)
    return pd.concat([df[["uid", "mid", "time", "content"]], time_feat, text_feat, hist_feat], axis=1)


def build_history_cache(train_df: pd.DataFrame) -> Tuple[Dict[str, dict], Dict[str, float]]:
    df = train_df.sort_values(["time", "uid", "mid"]).reset_index(drop=True).copy()
    df["total_count"] = (df["forward_count"] + df["comment_count"] + df["like_count"]).astype(np.int32)
    df["is_zero_total"] = (df["total_count"] == 0).astype(np.int8)

    cache: Dict[str, dict] = {}
    for uid, g in df.groupby("uid", sort=False):
        ts = (g["time"].astype("int64") // 10**9).values.astype(np.int64)
        f = g["forward_count"].values.astype(np.float32)
        c = g["comment_count"].values.astype(np.float32)
        l = g["like_count"].values.astype(np.float32)
        total = g["total_count"].values.astype(np.float32)
        zero = g["is_zero_total"].values.astype(np.float32)

        cache[uid] = {
            "ts": ts,
            "f": f,
            "c": c,
            "l": l,
            "total": total,
            "zero": zero,
            "cum_f": np.cumsum(f),
            "cum_c": np.cumsum(c),
            "cum_l": np.cumsum(l),
            "cum_total": np.cumsum(total),
            "cum_zero": np.cumsum(zero),
            "cummax_f": np.maximum.accumulate(f),
            "cummax_c": np.maximum.accumulate(c),
            "cummax_l": np.maximum.accumulate(l),
            "cummax_total": np.maximum.accumulate(total),
            "first_ts": int(ts[0]),
        }

    global_defaults = {
        "global_forward_mean": float(df["forward_count"].mean()),
        "global_comment_mean": float(df["comment_count"].mean()),
        "global_like_mean": float(df["like_count"].mean()),
        "global_total_mean": float(df["total_count"].mean()),
        "global_forward_max": float(df["forward_count"].max()),
        "global_comment_max": float(df["comment_count"].max()),
        "global_like_max": float(df["like_count"].max()),
        "global_total_max": float(df["total_count"].max()),
        "global_zero_ratio": float(df["is_zero_total"].mean()),
    }
    return cache, global_defaults


def make_infer_features_v2(
    df: pd.DataFrame,
    history_cache: Dict[str, dict],
    global_defaults: Dict[str, float],
    recent_ks: List[int],
    recent_days: List[int],
) -> pd.DataFrame:
    base = pd.concat([df[["uid", "mid", "time", "content"]].copy(), add_time_features(df), add_content_surface_features(df)], axis=1)
    rows = []
    max_recent_k = max(recent_ks)

    ts_queries = (df["time"].astype("int64") // 10**9).values.astype(np.int64)
    for i, row in df.iterrows():
        uid = row["uid"]
        qts = int(ts_queries[i])
        hist = history_cache.get(uid)
        end = 0 if hist is None else int(np.searchsorted(hist["ts"], qts, side="left"))

        if hist is None or end <= 0:
            feat = {
                "user_prev_post_count": 0.0,
                "user_days_since_prev_post": -1.0,
                "user_days_since_first_post": 0.0,
                "user_prev_zero_ratio": 0.0,
                "forward_count_prev_mean": global_defaults["global_forward_mean"],
                "forward_count_prev_max": global_defaults["global_forward_max"],
                "forward_count_prev_last1": global_defaults["global_forward_mean"],
                "forward_count_prev_last2": global_defaults["global_forward_mean"],
                "comment_count_prev_mean": global_defaults["global_comment_mean"],
                "comment_count_prev_max": global_defaults["global_comment_max"],
                "comment_count_prev_last1": global_defaults["global_comment_mean"],
                "comment_count_prev_last2": global_defaults["global_comment_mean"],
                "like_count_prev_mean": global_defaults["global_like_mean"],
                "like_count_prev_max": global_defaults["global_like_max"],
                "like_count_prev_last1": global_defaults["global_like_mean"],
                "like_count_prev_last2": global_defaults["global_like_mean"],
                "total_count_prev_mean": global_defaults["global_total_mean"],
                "total_count_prev_max": global_defaults["global_total_max"],
                "total_count_prev_last1": global_defaults["global_total_mean"],
                "total_count_prev_last2": global_defaults["global_total_mean"],
                "forward_share_prev_mean": global_defaults["global_forward_mean"] / (global_defaults["global_total_mean"] + 1e-3),
                "comment_share_prev_mean": global_defaults["global_comment_mean"] / (global_defaults["global_total_mean"] + 1e-3),
                "like_share_prev_mean": global_defaults["global_like_mean"] / (global_defaults["global_total_mean"] + 1e-3),
                "user_prev_posts_per_day": 0.0,
            }
            for k in recent_ks:
                feat[f"forward_count_recent{k}_mean"] = global_defaults["global_forward_mean"]
                feat[f"comment_count_recent{k}_mean"] = global_defaults["global_comment_mean"]
                feat[f"like_count_recent{k}_mean"] = global_defaults["global_like_mean"]
                feat[f"total_count_recent{k}_mean"] = global_defaults["global_total_mean"]
                feat[f"total_count_recent{k}_max"] = global_defaults["global_total_max"]
                feat[f"zero_ratio_recent{k}"] = global_defaults["global_zero_ratio"]
            for d in recent_days:
                feat[f"user_posts_last{d}d"] = 0.0
                feat[f"total_count_last{d}d_mean"] = 0.0
                feat[f"total_count_last{d}d_max"] = 0.0
                feat[f"zero_ratio_last{d}d"] = 0.0
            rows.append(feat)
            continue

        cnt = end
        sum_f = float(hist["cum_f"][end - 1])
        sum_c = float(hist["cum_c"][end - 1])
        sum_l = float(hist["cum_l"][end - 1])
        sum_total = float(hist["cum_total"][end - 1])
        zero_sum = float(hist["cum_zero"][end - 1])
        total_mean = sum_total / cnt
        days_since_prev = (qts - int(hist["ts"][end - 1])) / 86400.0
        days_since_first = (qts - int(hist["first_ts"])) / 86400.0

        feat = {
            "user_prev_post_count": float(cnt),
            "user_days_since_prev_post": float(days_since_prev),
            "user_days_since_first_post": float(days_since_first),
            "user_prev_zero_ratio": float(zero_sum / cnt),
            "forward_count_prev_mean": float(sum_f / cnt),
            "forward_count_prev_max": float(hist["cummax_f"][end - 1]),
            "forward_count_prev_last1": float(hist["f"][end - 1]),
            "forward_count_prev_last2": float(hist["f"][end - 2] if end >= 2 else 0.0),
            "comment_count_prev_mean": float(sum_c / cnt),
            "comment_count_prev_max": float(hist["cummax_c"][end - 1]),
            "comment_count_prev_last1": float(hist["c"][end - 1]),
            "comment_count_prev_last2": float(hist["c"][end - 2] if end >= 2 else 0.0),
            "like_count_prev_mean": float(sum_l / cnt),
            "like_count_prev_max": float(hist["cummax_l"][end - 1]),
            "like_count_prev_last1": float(hist["l"][end - 1]),
            "like_count_prev_last2": float(hist["l"][end - 2] if end >= 2 else 0.0),
            "total_count_prev_mean": float(total_mean),
            "total_count_prev_max": float(hist["cummax_total"][end - 1]),
            "total_count_prev_last1": float(hist["total"][end - 1]),
            "total_count_prev_last2": float(hist["total"][end - 2] if end >= 2 else 0.0),
            "forward_share_prev_mean": float((sum_f / cnt) / (total_mean + 1e-3)),
            "comment_share_prev_mean": float((sum_c / cnt) / (total_mean + 1e-3)),
            "like_share_prev_mean": float((sum_l / cnt) / (total_mean + 1e-3)),
            "user_prev_posts_per_day": float(cnt / (max(days_since_first, 0.0) + 1.0)),
        }

        start_k = max(0, end - max_recent_k)
        items = list(
            zip(
                hist["ts"][start_k:end].tolist(),
                hist["f"][start_k:end].astype(int).tolist(),
                hist["c"][start_k:end].astype(int).tolist(),
                hist["l"][start_k:end].astype(int).tolist(),
                hist["total"][start_k:end].astype(int).tolist(),
                hist["zero"][start_k:end].astype(int).tolist(),
            )
        )
        feat.update(_recent_stats_from_items(items, recent_ks))

        for d in recent_days:
            lower = qts - d * 86400
            st = int(np.searchsorted(hist["ts"], lower, side="left"))
            if st >= end:
                feat[f"user_posts_last{d}d"] = 0.0
                feat[f"total_count_last{d}d_mean"] = 0.0
                feat[f"total_count_last{d}d_max"] = 0.0
                feat[f"zero_ratio_last{d}d"] = 0.0
            else:
                total_slice = hist["total"][st:end]
                zero_slice = hist["zero"][st:end]
                feat[f"user_posts_last{d}d"] = float(end - st)
                feat[f"total_count_last{d}d_mean"] = float(total_slice.mean())
                feat[f"total_count_last{d}d_max"] = float(total_slice.max())
                feat[f"zero_ratio_last{d}d"] = float(zero_slice.mean())

        rows.append(feat)
        if (i + 1) % 100000 == 0:
            print(f"  已处理推理历史特征: {i + 1:,} 行")

    hist_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([base, hist_df], axis=1)


def build_feature_bundle(train_df: pd.DataFrame, recent_ks: List[int], recent_days: List[int]) -> FeatureBundle:
    train_feat_df = add_cumulative_user_features_v2(train_df, recent_ks=recent_ks, recent_days=recent_days)
    cache, global_defaults = build_history_cache(train_df)
    exclude_cols = {"uid", "mid", "time", "content"}
    feature_cols = [c for c in train_feat_df.columns if c not in exclude_cols]
    fill_values = train_feat_df[feature_cols].median(numeric_only=True).astype(float).to_dict()
    train_feat_df[feature_cols] = train_feat_df[feature_cols].fillna(fill_values).astype(np.float32)
    return FeatureBundle(
        numeric_df=train_feat_df,
        feature_cols=feature_cols,
        fill_values=fill_values,
        history_cache=cache,
        global_defaults=global_defaults,
        recent_ks=recent_ks,
        recent_days=recent_days,
    )


def prepare_infer_bundle(df: pd.DataFrame, feature_bundle: FeatureBundle) -> pd.DataFrame:
    feat_df = make_infer_features_v2(
        df,
        history_cache=feature_bundle.history_cache,
        global_defaults=feature_bundle.global_defaults,
        recent_ks=feature_bundle.recent_ks,
        recent_days=feature_bundle.recent_days,
    )
    feat_df[feature_bundle.feature_cols] = feat_df[feature_bundle.feature_cols].fillna(feature_bundle.fill_values).astype(np.float32)
    return feat_df


# ------------------------- 文本与模型 -------------------------

def fit_text_vectorizer(train_texts: pd.Series, max_features: int, min_df: int) -> TfidfVectorizer:
    texts = train_texts.fillna("").astype(str).tolist()
    trial_settings = [
        {"ngram_range": (2, 4), "min_df": max(1, min(int(min_df), max(1, len(texts) - 1)))},
        {"ngram_range": (2, 3), "min_df": 1},
        {"ngram_range": (1, 3), "min_df": 1},
    ]
    last_err = None
    for setting in trial_settings:
        try:
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=setting["ngram_range"],
                lowercase=False,
                sublinear_tf=True,
                min_df=setting["min_df"],
                max_df=1.0,
                max_features=max_features,
            )
            vectorizer.fit(texts)
            return vectorizer
        except ValueError as exc:
            last_err = exc
    raise ValueError(f"文本向量器构建失败: {last_err}")


def make_sparse_text_features(
    vectorizer: Optional[TfidfVectorizer],
    texts: pd.Series,
    numeric_df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[sparse.csr_matrix, Optional[StandardScaler]]:
    X_num = numeric_df.values.astype(np.float32)
    if fit_scaler:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
    else:
        X_num = scaler.transform(X_num)
    X_num_sparse = csr_matrix(X_num)
    if vectorizer is None:
        return X_num_sparse, scaler
    X_text = vectorizer.transform(texts.fillna("").astype(str).tolist())
    X = hstack([X_num_sparse, X_text], format="csr")
    return X, scaler


def append_sentence_embeddings(numeric_df: pd.DataFrame, embeddings: np.ndarray, prefix: str = "sentemb") -> Tuple[pd.DataFrame, List[str]]:
    emb_dim = int(embeddings.shape[1])
    cols = [f"{prefix}_{i:03d}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(embeddings.astype(np.float32), index=numeric_df.index, columns=cols)
    out_df = pd.concat([numeric_df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
    return out_df, cols


def official_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    dev_f = np.abs(y_pred[:, 0] - y_true[:, 0]) / (y_true[:, 0] + 5.0)
    dev_c = np.abs(y_pred[:, 1] - y_true[:, 1]) / (y_true[:, 1] + 3.0)
    dev_l = np.abs(y_pred[:, 2] - y_true[:, 2]) / (y_true[:, 2] + 3.0)
    precision_i = 1.0 - 0.5 * dev_f - 0.25 * dev_c - 0.25 * dev_l
    post_weight = np.minimum(y_true.sum(axis=1), 100.0) + 1.0
    hit = (precision_i > 0.8).astype(np.float32)
    return float((post_weight * hit).sum() / post_weight.sum())


def lightgbm_params(seed: int, num_threads: int) -> Dict[str, object]:
    return {
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "n_estimators": 3000,
        "random_state": seed,
        "n_jobs": int(num_threads),
        "verbosity": -1,
    }


def fit_models_validate(
    train_num: pd.DataFrame,
    valid_num: pd.DataFrame,
    train_sparse: csr_matrix,
    valid_sparse: csr_matrix,
    y_train: pd.DataFrame,
    y_valid: pd.DataFrame,
    seed: int,
    num_threads: int,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    lgb_models, ridge_models, lgb_preds, ridge_preds, best_iters = {}, {}, {}, {}, {}
    params = lightgbm_params(seed, num_threads=num_threads)

    for target in TARGETS:
        y_tr = np.log1p(y_train[target].values.astype(np.float32))
        y_va = np.log1p(y_valid[target].values.astype(np.float32))

        model = lgb.LGBMRegressor(**params)
        model.fit(
            train_num.values,
            y_tr,
            eval_set=[(valid_num.values, y_va)],
            eval_metric="l1",
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )
        lgb_models[target] = model
        lgb_preds[target] = model.predict(valid_num.values, num_iteration=model.best_iteration_)
        best_iters[target] = int(model.best_iteration_ if model.best_iteration_ is not None else params["n_estimators"])

        ridge = Ridge(alpha=2.0, solver="lsqr", random_state=seed)
        ridge.fit(train_sparse, y_tr)
        ridge_models[target] = ridge
        ridge_preds[target] = ridge.predict(valid_sparse)

    return lgb_models, ridge_models, lgb_preds, ridge_preds, best_iters


def fit_models_full(
    train_num: pd.DataFrame,
    train_sparse: csr_matrix,
    y_train: pd.DataFrame,
    seed: int,
    lgb_rounds: Dict[str, int],
    num_threads: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    lgb_models, ridge_models = {}, {}
    base_params = lightgbm_params(seed, num_threads=num_threads)
    for target in TARGETS:
        params = dict(base_params)
        params["n_estimators"] = int(lgb_rounds.get(target, DEFAULT_FULL_ITERS[target]))
        y_tr = np.log1p(y_train[target].values.astype(np.float32))
        model = lgb.LGBMRegressor(**params)
        model.fit(train_num.values, y_tr)
        lgb_models[target] = model

        ridge = Ridge(alpha=2.0, solver="lsqr", random_state=seed)
        ridge.fit(train_sparse, y_tr)
        ridge_models[target] = ridge
    return lgb_models, ridge_models


def blend_predictions(lgb_preds: Dict[str, np.ndarray], ridge_preds: Dict[str, np.ndarray], blend_weights: Dict[str, float]) -> np.ndarray:
    pred = np.zeros((len(next(iter(lgb_preds.values()))), 3), dtype=np.float32)
    for idx, target in enumerate(TARGETS):
        w = float(blend_weights.get(target, DEFAULT_BLEND[target]))
        raw = w * lgb_preds[target] + (1.0 - w) * ridge_preds[target]
        pred[:, idx] = np.clip(np.expm1(raw), 0.0, None)
    return pred


def apply_postprocess(pred: np.ndarray, config: Dict[str, Dict[str, float]], int_mode: str = "round") -> np.ndarray:
    out = pred.copy().astype(np.float32)
    for idx, target in enumerate(TARGETS):
        cfg = config[target]
        scale = float(cfg.get("scale", 1.0))
        zero_thr = float(cfg.get("zero_thr", 0.5))
        max_clip = float(cfg.get("max_clip", 500.0))
        out[:, idx] = np.clip(out[:, idx] * scale, 0.0, max_clip)
        out[:, idx][out[:, idx] < zero_thr] = 0.0
    if int_mode == "floor":
        out = np.floor(out)
    else:
        out = np.rint(out)
    return np.clip(out, 0, None).astype(np.int32)


def tune_blend_weights(y_true: np.ndarray, lgb_preds: Dict[str, np.ndarray], ridge_preds: Dict[str, np.ndarray]) -> Dict[str, float]:
    weights = dict(DEFAULT_BLEND)
    grid = [0.65, 0.75, 0.85, 0.95, 1.0]
    base_pp = {t: {"scale": 1.0, "zero_thr": 0.5, "max_clip": 500.0} for t in TARGETS}
    for _ in range(2):
        for target in TARGETS:
            best_w, best_score = weights[target], -1.0
            for w in grid:
                trial = dict(weights)
                trial[target] = w
                pred = blend_predictions(lgb_preds, ridge_preds, trial)
                pred_int = apply_postprocess(pred, base_pp, int_mode="round")
                score = official_score(y_true, pred_int)
                if score > best_score:
                    best_score = score
                    best_w = w
            weights[target] = best_w
    return weights


def tune_postprocess(y_true: np.ndarray, pred_cont: np.ndarray) -> Tuple[Dict[str, Dict[str, float]], str, float]:
    config = {t: {"scale": 1.0, "zero_thr": 0.5, "max_clip": 500.0} for t in TARGETS}
    scale_grid = {t: [0.75, 0.85, 0.95, 1.00, 1.05] for t in TARGETS}
    thr_grid = {t: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0] for t in TARGETS}
    clip_grid = {t: [200.0, 300.0, 500.0] for t in TARGETS}

    best_mode = "round"
    best_score = official_score(y_true, apply_postprocess(pred_cont, config, int_mode=best_mode))
    for _ in range(2):
        for target in TARGETS:
            best_local = (config[target]["scale"], config[target]["zero_thr"], config[target]["max_clip"], best_mode, best_score)
            for s in scale_grid[target]:
                for thr in thr_grid[target]:
                    for clip in clip_grid[target]:
                        for mode in ["round", "floor"]:
                            trial = json.loads(json.dumps(config))
                            trial[target]["scale"] = s
                            trial[target]["zero_thr"] = thr
                            trial[target]["max_clip"] = clip
                            pred_int = apply_postprocess(pred_cont, trial, int_mode=mode)
                            score = official_score(y_true, pred_int)
                            if score > best_local[4]:
                                best_local = (s, thr, clip, mode, score)
            config[target]["scale"] = best_local[0]
            config[target]["zero_thr"] = best_local[1]
            config[target]["max_clip"] = best_local[2]
            best_mode = best_local[3]
            best_score = best_local[4]
    return config, best_mode, best_score


# ------------------------- 训练/验证/提交 -------------------------

def save_validation_predictions(path: str, valid_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = valid_df[["uid", "mid", "time"]].copy()
    out["true_forward"] = y_true[:, 0]
    out["true_comment"] = y_true[:, 1]
    out["true_like"] = y_true[:, 2]
    out["pred_forward"] = y_pred[:, 0]
    out["pred_comment"] = y_pred[:, 1]
    out["pred_like"] = y_pred[:, 2]
    out.to_csv(path, index=False)


def write_submission(path: str, test_df: pd.DataFrame, pred_int: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, row in test_df.iterrows():
            f.write(f"{row['uid']}\t{row['mid']}\t{int(pred_int[i, 0])},{int(pred_int[i, 1])},{int(pred_int[i, 2])}\n")


def summarize_split(train_df: pd.DataFrame, valid_df: Optional[pd.DataFrame] = None) -> None:
    print("=" * 80)
    print(f"训练样本数: {len(train_df):,}")
    print(f"训练时间范围: {train_df['time'].min()} -> {train_df['time'].max()}")
    if valid_df is not None:
        print(f"验证样本数: {len(valid_df):,}")
        print(f"验证时间范围: {valid_df['time'].min()} -> {valid_df['time'].max()}")
    print("=" * 80)


def _build_sentence_encoder(args) -> Optional[SentenceEmbeddingEncoder]:
    if not getattr(args, "use_sentence_embeddings", False):
        return None
    cache_dir = getattr(args, "embedding_cache_dir", None)
    if cache_dir in (None, ""):
        cache_dir = os.path.join(args.output_dir, "embedding_cache")
    encoder = SentenceEmbeddingEncoder(
        model_name=args.sentence_model_name,
        device=getattr(args, "embedding_device", "auto"),
        batch_size=int(getattr(args, "embedding_batch_size", 128)),
        normalize_embeddings=bool(getattr(args, "embedding_normalize", True)),
        max_chars=int(getattr(args, "embedding_max_chars", 256)),
        cache_dir=cache_dir,
        text_prefix=str(getattr(args, "embedding_text_prefix", "")),
        max_seq_length=getattr(args, "embedding_max_seq_length", None),
    )
    print(f"句向量模型: {encoder.model_name}")
    print(f"句向量设备: {encoder.device}")
    print(f"句向量缓存目录: {cache_dir}")
    return encoder


def _validate_core(args, train_df: pd.DataFrame, valid_df: pd.DataFrame, output_dir: str, save_artifacts: bool = True) -> Dict[str, object]:
    summarize_split(train_df, valid_df)

    print("[2/8] 构造基础数值特征...")
    train_bundle = build_feature_bundle(train_df, recent_ks=args.recent_ks, recent_days=args.recent_days)
    valid_feat_df = prepare_infer_bundle(valid_df, train_bundle)
    X_train_num = train_bundle.numeric_df[train_bundle.feature_cols].copy()
    X_valid_num = valid_feat_df[train_bundle.feature_cols].copy()
    sentence_feature_cols: List[str] = []

    sentence_encoder = _build_sentence_encoder(args)
    if sentence_encoder is not None:
        print("[3/8] 生成预训练句向量...")
        train_emb = sentence_encoder.encode(train_df["content"], split_name="train")
        valid_emb = sentence_encoder.encode(valid_df["content"], split_name="valid")
        print(f"句向量维度: {train_emb.shape[1]}")
        X_train_num, sentence_feature_cols = append_sentence_embeddings(X_train_num, train_emb, prefix="sentemb")
        X_valid_num, _ = append_sentence_embeddings(X_valid_num, valid_emb, prefix="sentemb")
    else:
        print("[3/8] 跳过预训练句向量分支...")

    vectorizer = None
    if getattr(args, "use_tfidf", True):
        print("[4/8] 构造字符 TF-IDF 特征...")
        vectorizer = fit_text_vectorizer(train_df["content"], args.tfidf_max_features, args.tfidf_min_df)
    else:
        print("[4/8] 跳过字符 TF-IDF 特征...")

    print("[5/8] 构造稀疏训练矩阵...")
    X_train_sparse, scaler = make_sparse_text_features(vectorizer, train_df["content"], X_train_num, fit_scaler=True)
    X_valid_sparse, _ = make_sparse_text_features(vectorizer, valid_df["content"], X_valid_num, scaler=scaler, fit_scaler=False)

    y_train = train_df[TARGETS].copy()
    y_valid = valid_df[TARGETS].copy()

    print("[6/8] 训练模型...")
    lgb_models, ridge_models, lgb_preds, ridge_preds, best_iters = fit_models_validate(
        X_train_num, X_valid_num, X_train_sparse, X_valid_sparse, y_train, y_valid, args.seed, args.num_threads
    )

    print("[7/8] 融合与后处理调优...")
    y_valid_np = y_valid.values.astype(np.int32)
    blend_weights = tune_blend_weights(y_valid_np, lgb_preds, ridge_preds)
    pred_valid_cont = blend_predictions(lgb_preds, ridge_preds, blend_weights)

    base_pp = {t: {"scale": 1.0, "zero_thr": 0.5, "max_clip": 500.0} for t in TARGETS}
    base_pred_int = apply_postprocess(pred_valid_cont, base_pp, int_mode="round")
    base_score = official_score(y_valid_np, base_pred_int)

    tuned_pp, int_mode, tuned_score = tune_postprocess(y_valid_np, pred_valid_cont)
    tuned_pred_int = apply_postprocess(pred_valid_cont, tuned_pp, int_mode=int_mode)

    config = {
        "seed": args.seed,
        "tfidf_max_features": args.tfidf_max_features,
        "tfidf_min_df": args.tfidf_min_df,
        "use_tfidf": bool(getattr(args, "use_tfidf", True)),
        "use_sentence_embeddings": bool(getattr(args, "use_sentence_embeddings", False)),
        "sentence_model_name": getattr(args, "sentence_model_name", None),
        "embedding_device": getattr(args, "embedding_device", "auto"),
        "embedding_batch_size": int(getattr(args, "embedding_batch_size", 128)),
        "embedding_normalize": bool(getattr(args, "embedding_normalize", True)),
        "embedding_max_chars": int(getattr(args, "embedding_max_chars", 256)),
        "embedding_text_prefix": str(getattr(args, "embedding_text_prefix", "")),
        "sentence_embedding_dim": int(len(sentence_feature_cols)),
        "recent_ks": args.recent_ks,
        "recent_days": args.recent_days,
        "base_feature_cols": train_bundle.feature_cols,
        "sentence_feature_cols": sentence_feature_cols,
        "final_feature_cols": list(X_train_num.columns),
        "best_iterations": best_iters,
        "blend_weights": blend_weights,
        "postprocess": tuned_pp,
        "int_mode": int_mode,
        "base_score": base_score,
        "tuned_score": tuned_score,
        "train_min_time": str(train_df['time'].min()),
        "train_max_time": str(train_df['time'].max()),
        "valid_min_time": str(valid_df['time'].min()),
        "valid_max_time": str(valid_df['time'].max()),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
    }

    if save_artifacts:
        print("[8/8] 输出结果...")
        os.makedirs(output_dir, exist_ok=True)
        save_validation_predictions(os.path.join(output_dir, "valid_predictions.csv"), valid_df, y_valid_np, tuned_pred_int)
        with open(os.path.join(output_dir, "validate_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        bundle = {
            "vectorizer": vectorizer,
            "scaler": scaler,
            "fill_values": train_bundle.fill_values,
            "base_feature_cols": train_bundle.feature_cols,
            "sentence_feature_cols": sentence_feature_cols,
            "final_feature_cols": list(X_train_num.columns),
            "history_cache": train_bundle.history_cache,
            "global_defaults": train_bundle.global_defaults,
            "recent_ks": train_bundle.recent_ks,
            "recent_days": train_bundle.recent_days,
            "lgb_models": lgb_models,
            "ridge_models": ridge_models,
            "blend_weights": blend_weights,
            "postprocess": tuned_pp,
            "int_mode": int_mode,
            "best_iterations": best_iters,
            "sentence_model_name": getattr(args, "sentence_model_name", None),
            "use_tfidf": bool(getattr(args, "use_tfidf", True)),
            "use_sentence_embeddings": bool(getattr(args, "use_sentence_embeddings", False)),
            "embedding_normalize": bool(getattr(args, "embedding_normalize", True)),
            "embedding_max_chars": int(getattr(args, "embedding_max_chars", 256)),
        }
        joblib.dump(bundle, os.path.join(output_dir, "validate_artifacts.joblib"))
    else:
        print("[8/8] 跳过单折模型落盘，仅保留评分摘要...")

    print("验证完成")
    print(f"基础分数(默认融合+默认后处理): {base_score:.6f}")
    print(f"调优后分数: {tuned_score:.6f}")
    print(f"blend_weights: {blend_weights}")
    print(f"int_mode: {int_mode}")
    print(f"postprocess: {json.dumps(tuned_pp, ensure_ascii=False)}")
    print(f"best_iterations: {best_iters}")
    print(f"结果目录: {output_dir}")
    return config


def run_validate(args) -> None:
    print("[1/8] 读取训练数据...")
    df = read_data(args.train_path, is_train=True)
    df = df.sort_values(["time", "uid", "mid"]).reset_index(drop=True)

    valid_start = pd.to_datetime(args.valid_start)
    train_df = df[df["time"] < valid_start].copy().reset_index(drop=True)
    valid_df = df[df["time"] >= valid_start].copy().reset_index(drop=True)
    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("按当前 valid_start 划分后，训练集或验证集为空，请调整日期。")

    config = _validate_core(args, train_df, valid_df, output_dir=args.output_dir, save_artifacts=True)
    config["valid_start"] = args.valid_start
    with open(os.path.join(args.output_dir, "validate_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def run_cv(args) -> None:
    print("[1/3] 读取训练数据...")
    df = read_data(args.train_path, is_train=True)
    df = df.sort_values(["time", "uid", "mid"]).reset_index(drop=True)
    os.makedirs(args.output_dir, exist_ok=True)

    fold_results = []
    for fold_id, valid_start in enumerate(args.cv_valid_starts, start=1):
        print("\n" + "#" * 80)
        print(f"Fold {fold_id}: valid_start = {valid_start}")
        valid_start_dt = pd.to_datetime(valid_start)
        train_df = df[df["time"] < valid_start_dt].copy().reset_index(drop=True)
        valid_df = df[df["time"] >= valid_start_dt].copy().reset_index(drop=True)
        if len(train_df) == 0 or len(valid_df) == 0:
            print(f"Fold {fold_id} 跳过: 划分后训练集或验证集为空")
            continue
        fold_dir = os.path.join(args.output_dir, f"fold_{fold_id}_{valid_start}")
        cfg = _validate_core(args, train_df, valid_df, output_dir=fold_dir, save_artifacts=False)
        cfg["valid_start"] = valid_start
        fold_results.append(cfg)
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, "fold_summary.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    if not fold_results:
        raise ValueError("没有成功完成任何一个 CV fold。")

    summary = pd.DataFrame([
        {
            "valid_start": x["valid_start"],
            "base_score": x["base_score"],
            "tuned_score": x["tuned_score"],
            "n_train": x["n_train"],
            "n_valid": x["n_valid"],
        }
        for x in fold_results
    ])
    summary.to_csv(os.path.join(args.output_dir, "cv_summary.csv"), index=False)
    mean_tuned = float(summary["tuned_score"].mean())
    std_tuned = float(summary["tuned_score"].std(ddof=0))

    agg = {
        "mean_tuned_score": mean_tuned,
        "std_tuned_score": std_tuned,
        "folds": fold_results,
    }
    with open(os.path.join(args.output_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    print("\n[2/3] CV 完成")
    print(summary)
    print(f"mean tuned score = {mean_tuned:.6f}")
    print(f"std tuned score  = {std_tuned:.6f}")
    print(f"结果目录: {args.output_dir}")
    print("[3/3] 你现在可以根据不同月份的稳定性，再决定 full 模式要复用哪一折的 validate_config。")


def load_validate_config(path: Optional[str]) -> Dict[str, object]:
    if path is None or not os.path.exists(path):
        return {
            "best_iterations": DEFAULT_FULL_ITERS,
            "blend_weights": DEFAULT_BLEND,
            "postprocess": {t: {"scale": 1.0, "zero_thr": 0.5, "max_clip": 500.0} for t in TARGETS},
            "int_mode": "round",
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_full(args) -> None:
    print("[1/7] 读取训练/测试数据...")
    train_df = read_data(args.train_path, is_train=True)
    train_df = train_df.sort_values(["time", "uid", "mid"]).reset_index(drop=True)
    test_df = read_data(args.test_path, is_train=False)
    test_df = test_df.sort_values(["time", "uid", "mid"]).reset_index(drop=True)

    summarize_split(train_df)
    print(f"测试样本数: {len(test_df):,}")
    print(f"测试时间范围: {test_df['time'].min()} -> {test_df['time'].max()}")

    print("[2/7] 构造基础数值特征...")
    train_bundle = build_feature_bundle(train_df, recent_ks=args.recent_ks, recent_days=args.recent_days)
    test_feat_df = prepare_infer_bundle(test_df, train_bundle)
    X_train_num = train_bundle.numeric_df[train_bundle.feature_cols].copy()
    X_test_num = test_feat_df[train_bundle.feature_cols].copy()
    sentence_feature_cols: List[str] = []

    sentence_encoder = _build_sentence_encoder(args)
    if sentence_encoder is not None:
        print("[3/7] 生成预训练句向量...")
        train_emb = sentence_encoder.encode(train_df["content"], split_name="train_full")
        test_emb = sentence_encoder.encode(test_df["content"], split_name="test")
        print(f"句向量维度: {train_emb.shape[1]}")
        X_train_num, sentence_feature_cols = append_sentence_embeddings(X_train_num, train_emb, prefix="sentemb")
        X_test_num, _ = append_sentence_embeddings(X_test_num, test_emb, prefix="sentemb")
    else:
        print("[3/7] 跳过预训练句向量分支...")

    vectorizer = None
    if getattr(args, "use_tfidf", True):
        print("[4/7] 构造字符 TF-IDF 特征...")
        vectorizer = fit_text_vectorizer(train_df["content"], args.tfidf_max_features, args.tfidf_min_df)
    else:
        print("[4/7] 跳过字符 TF-IDF 特征...")

    print("[5/7] 构造稀疏训练矩阵并训练全量模型...")
    X_train_sparse, scaler = make_sparse_text_features(vectorizer, train_df["content"], X_train_num, fit_scaler=True)
    X_test_sparse, _ = make_sparse_text_features(vectorizer, test_df["content"], X_test_num, scaler=scaler, fit_scaler=False)

    cfg = load_validate_config(args.validate_config)
    best_iterations = cfg.get("best_iterations", DEFAULT_FULL_ITERS)
    blend_weights = cfg.get("blend_weights", DEFAULT_BLEND)
    postprocess = cfg.get("postprocess", {t: {"scale": 1.0, "zero_thr": 0.5, "max_clip": 500.0} for t in TARGETS})
    int_mode = cfg.get("int_mode", "round")

    lgb_models, ridge_models = fit_models_full(X_train_num, X_train_sparse, train_df[TARGETS].copy(), args.seed, best_iterations, args.num_threads)

    print("[6/7] 预测测试集...")
    lgb_preds, ridge_preds = {}, {}
    for target in TARGETS:
        lgb_preds[target] = lgb_models[target].predict(X_test_num.values)
        ridge_preds[target] = ridge_models[target].predict(X_test_sparse)
    pred_test_cont = blend_predictions(lgb_preds, ridge_preds, blend_weights)
    pred_test_int = apply_postprocess(pred_test_cont, postprocess, int_mode=int_mode)

    print("[7/7] 写出提交文件...")
    os.makedirs(args.output_dir, exist_ok=True)
    submission_path = os.path.join(args.output_dir, args.submission_name)
    write_submission(submission_path, test_df, pred_test_int)

    bundle = {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "fill_values": train_bundle.fill_values,
        "base_feature_cols": train_bundle.feature_cols,
        "sentence_feature_cols": sentence_feature_cols,
        "final_feature_cols": list(X_train_num.columns),
        "history_cache": train_bundle.history_cache,
        "global_defaults": train_bundle.global_defaults,
        "recent_ks": train_bundle.recent_ks,
        "recent_days": train_bundle.recent_days,
        "lgb_models": lgb_models,
        "ridge_models": ridge_models,
        "blend_weights": blend_weights,
        "postprocess": postprocess,
        "int_mode": int_mode,
        "best_iterations": best_iterations,
        "sentence_model_name": getattr(args, "sentence_model_name", None),
        "use_tfidf": bool(getattr(args, "use_tfidf", True)),
        "use_sentence_embeddings": bool(getattr(args, "use_sentence_embeddings", False)),
        "embedding_normalize": bool(getattr(args, "embedding_normalize", True)),
        "embedding_max_chars": int(getattr(args, "embedding_max_chars", 256)),
    }
    joblib.dump(bundle, os.path.join(args.output_dir, "full_artifacts.joblib"))
    print(f"提交文件已保存: {submission_path}")
    print(f"模型与特征已保存: {os.path.join(args.output_dir, 'full_artifacts.joblib')}")


# ------------------------- 配置入口 -------------------------
CONFIG = {
    # 运行模式: "validate" / "cv" / "full"
    "mode": "validate",

    # 数据路径
    "train_path": "data/weibo_train_data.txt",
    "test_path": "data/weibo_predict_data.txt",

    # 单折验证起点
    "valid_start": "2015-07-01",

    # 滚动时间验证月份
    "cv_valid_starts": ["2015-06-01", "2015-07-01"],

    # 输出目录
    "output_dir": "./outputs/weibo_baseline_v3",
    "submission_name": "submission.txt",

    # 文本特征：v3 支持两条文本支路
    "use_tfidf": True,
    "tfidf_max_features": 50000,
    "tfidf_min_df": 5,

    # 预训练句向量分支（v3 新增）
    "use_sentence_embeddings": True,
    "sentence_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_device": "auto",          # auto / cuda / cpu
    "embedding_batch_size": 128,
    "embedding_normalize": True,
    "embedding_max_chars": 256,
    "embedding_max_seq_length": 256,
    "embedding_text_prefix": "",
    "embedding_cache_dir": "./outputs/weibo_baseline_v3/embedding_cache",

    # v2 的近期历史窗口
    "recent_ks": [3, 5, 10],
    "recent_days": [7, 14, 30],

    # 训练参数
    "seed": 42,
    "num_threads": 8,

    # full 模式可复用 validate 产生的配置；没有就填 None
    "validate_config": None,
}


def dict_to_namespace(config: Dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**config)


def run_from_config(config: Dict[str, object]) -> None:
    args = dict_to_namespace(config)
    set_seed(args.seed)
    if args.mode == "validate":
        run_validate(args)
    elif args.mode == "cv":
        run_cv(args)
    elif args.mode == "full":
        if not getattr(args, "test_path", None):
            raise ValueError("full 模式下必须提供 test_path")
        run_full(args)
    else:
        raise ValueError(f"未知 mode: {args.mode}")


def main() -> None:
    run_from_config(CONFIG)


if __name__ == "__main__":
    main()
