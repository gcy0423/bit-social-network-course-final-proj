import csv
import json
import math
import os
import re
import warnings
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
    raise ImportError(
        "lightgbm 未安装。请先执行: pip install lightgbm"
    ) from exc

ENCODINGS = ["utf-8", "utf-8-sig", "gb18030", "latin1"]
TARGETS = ["forward_count", "comment_count", "like_count"]
TRAIN_COLUMNS = ["uid", "mid", "time", "forward_count", "comment_count", "like_count", "content"]
TEST_COLUMNS = ["uid", "mid", "time", "content"]
TARGET_BIAS = {"forward_count": 5.0, "comment_count": 3.0, "like_count": 3.0}
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
    user_stats: Dict[str, pd.DataFrame]
    fill_values: Dict[str, float]



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
        bad = int(df["time"].isna().sum())
        raise ValueError(f"time 字段有 {bad} 行无法解析，请检查原始数据格式。")
    if is_train:
        for col in TARGETS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).astype(np.int32)
    return df



def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float32)
    mask = b > 0
    out[mask] = (a[mask] / b[mask]).astype(np.float32)
    return out



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
        "has_tech": r"机器学习|Python|Java|Linux|Systemd|开源|大数据|Azure|Emacs",
        "has_photo": r"图片|美拍|拍摄",
        "has_topic": r"#.+?#",
    }
    for feat, patt in keyword_map.items():
        out[feat] = texts.str.contains(patt, regex=True).astype(np.int8)

    return out



def add_cumulative_user_features(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    df = train_df.sort_values(["time", "uid", "mid"]).reset_index(drop=True).copy()
    feat = pd.DataFrame(index=df.index)
    g_uid = df.groupby("uid", sort=False)

    total = (df["forward_count"] + df["comment_count"] + df["like_count"]).astype(np.int32)
    df["total_count"] = total
    df["is_zero_total"] = (total == 0).astype(np.int8)

    prev_post_count = g_uid.cumcount().astype(np.int32)
    feat["user_prev_post_count"] = prev_post_count

    feat["user_days_since_prev_post"] = (
        g_uid["time"].diff().dt.total_seconds().fillna(-1) / 86400.0
    ).astype(np.float32)
    first_time = g_uid["time"].transform("min")
    feat["user_days_since_first_post"] = (
        (df["time"] - first_time).dt.total_seconds() / 86400.0
    ).astype(np.float32)

    zero_prev_cumsum = g_uid["is_zero_total"].cumsum() - df["is_zero_total"]
    cnt_arr = prev_post_count.values.astype(np.float32)
    feat["user_prev_zero_ratio"] = safe_divide(zero_prev_cumsum.values.astype(np.float32), np.maximum(cnt_arr, 0))

    for col in TARGETS + ["total_count"]:
        prev_sum = g_uid[col].cumsum() - df[col]
        prev_max = g_uid[col].cummax().groupby(df["uid"], sort=False).shift(1)
        prev_last1 = g_uid[col].shift(1)
        prev_last2 = g_uid[col].shift(2)

        feat[f"{col}_prev_mean"] = safe_divide(prev_sum.values.astype(np.float32), np.maximum(cnt_arr, 0))
        feat[f"{col}_prev_max"] = prev_max.fillna(0).astype(np.float32)
        feat[f"{col}_prev_last1"] = prev_last1.fillna(0).astype(np.float32)
        feat[f"{col}_prev_last2"] = prev_last2.fillna(0).astype(np.float32)

    denom = feat["total_count_prev_mean"].values.astype(np.float32) + 1e-3
    feat["forward_share_prev_mean"] = (feat["forward_count_prev_mean"].values / denom).astype(np.float32)
    feat["comment_share_prev_mean"] = (feat["comment_count_prev_mean"].values / denom).astype(np.float32)
    feat["like_share_prev_mean"] = (feat["like_count_prev_mean"].values / denom).astype(np.float32)

    age = np.maximum(feat["user_days_since_first_post"].values.astype(np.float32), 0.0) + 1.0
    feat["user_prev_posts_per_day"] = (feat["user_prev_post_count"].values.astype(np.float32) / age).astype(np.float32)

    base_time_feat = add_time_features(df)
    base_text_feat = add_content_surface_features(df)
    feat = pd.concat([base_time_feat, base_text_feat, feat], axis=1)

    user_agg = g_uid.agg(
        hist_post_count=("mid", "count"),
        hist_forward_mean=("forward_count", "mean"),
        hist_comment_mean=("comment_count", "mean"),
        hist_like_mean=("like_count", "mean"),
        hist_total_mean=("total_count", "mean"),
        hist_forward_max=("forward_count", "max"),
        hist_comment_max=("comment_count", "max"),
        hist_like_max=("like_count", "max"),
        hist_total_max=("total_count", "max"),
        hist_zero_ratio=("is_zero_total", "mean"),
        last_post_time=("time", "max"),
        first_post_time=("time", "min"),
    )

    last1 = g_uid[["forward_count", "comment_count", "like_count", "total_count"]].nth(-1)
    last1.columns = [f"{c}_hist_last1" for c in last1.columns]

    last2 = g_uid[["forward_count", "comment_count", "like_count", "total_count"]].nth(-2)
    last2.columns = [f"{c}_hist_last2" for c in last2.columns]

    user_agg = user_agg.join(last1, how="left").join(last2, how="left")
    user_agg = user_agg.fillna(0)

    global_stats = {
        "global_forward_mean": float(df["forward_count"].mean()),
        "global_comment_mean": float(df["comment_count"].mean()),
        "global_like_mean": float(df["like_count"].mean()),
        "global_total_mean": float(df["total_count"].mean()),
        "global_forward_max": float(df["forward_count"].max()),
        "global_comment_max": float(df["comment_count"].max()),
        "global_like_max": float(df["like_count"].max()),
        "global_total_max": float(df["total_count"].max()),
        "global_zero_ratio": float(df["is_zero_total"].mean()),
        "global_days_since_prev_post": float(np.median(feat["user_days_since_prev_post"].replace(-1, np.nan).dropna()))
        if (feat["user_days_since_prev_post"] >= 0).any() else 7.0,
    }

    return pd.concat([df[["uid", "mid", "time", "content"]], feat], axis=1), {
        "user_agg": user_agg,
        "global_stats": pd.DataFrame([global_stats]),
    }



def make_infer_features(df: pd.DataFrame, user_stats: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat([df[["uid", "mid", "time", "content"]].copy(), add_time_features(df), add_content_surface_features(df)], axis=1)

    user_agg = user_stats["user_agg"]
    global_stats = user_stats["global_stats"].iloc[0].to_dict()
    joined = df[["uid", "time"]].merge(user_agg, how="left", left_on="uid", right_index=True)

    fill_map = {
        "hist_post_count": 0.0,
        "hist_forward_mean": global_stats["global_forward_mean"],
        "hist_comment_mean": global_stats["global_comment_mean"],
        "hist_like_mean": global_stats["global_like_mean"],
        "hist_total_mean": global_stats["global_total_mean"],
        "hist_forward_max": global_stats["global_forward_max"],
        "hist_comment_max": global_stats["global_comment_max"],
        "hist_like_max": global_stats["global_like_max"],
        "hist_total_max": global_stats["global_total_max"],
        "hist_zero_ratio": global_stats["global_zero_ratio"],
        "forward_count_hist_last1": global_stats["global_forward_mean"],
        "comment_count_hist_last1": global_stats["global_comment_mean"],
        "like_count_hist_last1": global_stats["global_like_mean"],
        "total_count_hist_last1": global_stats["global_total_mean"],
        "forward_count_hist_last2": global_stats["global_forward_mean"],
        "comment_count_hist_last2": global_stats["global_comment_mean"],
        "like_count_hist_last2": global_stats["global_like_mean"],
        "total_count_hist_last2": global_stats["global_total_mean"],
    }

    for col, val in fill_map.items():
        joined[col] = joined[col].fillna(val)

    last_post = pd.to_datetime(joined["last_post_time"], errors="coerce")
    first_post = pd.to_datetime(joined["first_post_time"], errors="coerce")

    days_since_last = ((joined["time"] - last_post).dt.total_seconds() / 86400.0)
    days_since_first = ((joined["time"] - first_post).dt.total_seconds() / 86400.0)

    joined["user_prev_post_count"] = joined["hist_post_count"].fillna(0).astype(np.float32)
    joined["user_days_since_prev_post"] = days_since_last.fillna(global_stats["global_days_since_prev_post"]).astype(np.float32)
    joined["user_days_since_first_post"] = days_since_first.fillna(0).astype(np.float32)
    joined["user_prev_zero_ratio"] = joined["hist_zero_ratio"].astype(np.float32)
    joined["forward_count_prev_mean"] = joined["hist_forward_mean"].astype(np.float32)
    joined["comment_count_prev_mean"] = joined["hist_comment_mean"].astype(np.float32)
    joined["like_count_prev_mean"] = joined["hist_like_mean"].astype(np.float32)
    joined["total_count_prev_mean"] = joined["hist_total_mean"].astype(np.float32)
    joined["forward_count_prev_max"] = joined["hist_forward_max"].astype(np.float32)
    joined["comment_count_prev_max"] = joined["hist_comment_max"].astype(np.float32)
    joined["like_count_prev_max"] = joined["hist_like_max"].astype(np.float32)
    joined["total_count_prev_max"] = joined["hist_total_max"].astype(np.float32)
    joined["forward_count_prev_last1"] = joined["forward_count_hist_last1"].astype(np.float32)
    joined["comment_count_prev_last1"] = joined["comment_count_hist_last1"].astype(np.float32)
    joined["like_count_prev_last1"] = joined["like_count_hist_last1"].astype(np.float32)
    joined["total_count_prev_last1"] = joined["total_count_hist_last1"].astype(np.float32)
    joined["forward_count_prev_last2"] = joined["forward_count_hist_last2"].astype(np.float32)
    joined["comment_count_prev_last2"] = joined["comment_count_hist_last2"].astype(np.float32)
    joined["like_count_prev_last2"] = joined["like_count_hist_last2"].astype(np.float32)
    joined["total_count_prev_last2"] = joined["total_count_hist_last2"].astype(np.float32)

    denom = joined["total_count_prev_mean"].values.astype(np.float32) + 1e-3
    joined["forward_share_prev_mean"] = (joined["forward_count_prev_mean"].values / denom).astype(np.float32)
    joined["comment_share_prev_mean"] = (joined["comment_count_prev_mean"].values / denom).astype(np.float32)
    joined["like_share_prev_mean"] = (joined["like_count_prev_mean"].values / denom).astype(np.float32)

    age = np.maximum(joined["user_days_since_first_post"].values.astype(np.float32), 0.0) + 1.0
    joined["user_prev_posts_per_day"] = (joined["user_prev_post_count"].values.astype(np.float32) / age).astype(np.float32)

    hist_cols = [
        "user_prev_post_count",
        "user_days_since_prev_post",
        "user_days_since_first_post",
        "user_prev_zero_ratio",
        "forward_count_prev_mean",
        "comment_count_prev_mean",
        "like_count_prev_mean",
        "total_count_prev_mean",
        "forward_count_prev_max",
        "comment_count_prev_max",
        "like_count_prev_max",
        "total_count_prev_max",
        "forward_count_prev_last1",
        "comment_count_prev_last1",
        "like_count_prev_last1",
        "total_count_prev_last1",
        "forward_count_prev_last2",
        "comment_count_prev_last2",
        "like_count_prev_last2",
        "total_count_prev_last2",
        "forward_share_prev_mean",
        "comment_share_prev_mean",
        "like_share_prev_mean",
        "user_prev_posts_per_day",
    ]

    out = pd.concat([out, joined[hist_cols].reset_index(drop=True)], axis=1)
    return out



def build_feature_bundle(train_df: pd.DataFrame) -> FeatureBundle:
    train_feat_df, user_stats = add_cumulative_user_features(train_df)
    exclude_cols = {"uid", "mid", "time", "content"}
    feature_cols = [c for c in train_feat_df.columns if c not in exclude_cols]
    fill_values = train_feat_df[feature_cols].median(numeric_only=True).astype(float).to_dict()
    train_feat_df[feature_cols] = train_feat_df[feature_cols].fillna(fill_values).astype(np.float32)
    return FeatureBundle(
        numeric_df=train_feat_df,
        feature_cols=feature_cols,
        user_stats=user_stats,
        fill_values=fill_values,
    )



def prepare_infer_bundle(df: pd.DataFrame, feature_bundle: FeatureBundle) -> pd.DataFrame:
    feat_df = make_infer_features(df, feature_bundle.user_stats)
    feat_df[feature_bundle.feature_cols] = feat_df[feature_bundle.feature_cols].fillna(feature_bundle.fill_values).astype(np.float32)
    return feat_df



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
            continue
    raise ValueError(f"文本向量器构建失败: {last_err}")



def make_sparse_text_features(
    vectorizer: TfidfVectorizer,
    texts: pd.Series,
    numeric_df: pd.DataFrame,
    feature_cols: List[str],
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[sparse.csr_matrix, Optional[StandardScaler]]:
    X_text = vectorizer.transform(texts.fillna("").astype(str).tolist())
    X_num = numeric_df[feature_cols].values.astype(np.float32)
    if fit_scaler:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
    else:
        X_num = scaler.transform(X_num)
    X_num_sparse = csr_matrix(X_num)
    X = hstack([X_num_sparse, X_text], format="csr")
    return X, scaler



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
    lgb_models = {}
    ridge_models = {}
    lgb_preds = {}
    ridge_preds = {}
    best_iters = {}

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
    lgb_models = {}
    ridge_models = {}
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



def blend_predictions(
    lgb_preds: Dict[str, np.ndarray],
    ridge_preds: Dict[str, np.ndarray],
    blend_weights: Dict[str, float],
) -> np.ndarray:
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
    out = np.clip(out, 0, None).astype(np.int32)
    return out



def tune_blend_weights(
    y_true: np.ndarray,
    lgb_preds: Dict[str, np.ndarray],
    ridge_preds: Dict[str, np.ndarray],
) -> Dict[str, float]:
    weights = dict(DEFAULT_BLEND)
    grid = [0.65, 0.75, 0.85, 0.95, 1.0]
    base_pp = {t: {"scale": 1.0, "zero_thr": 0.5, "max_clip": 500.0} for t in TARGETS}

    for _ in range(2):
        for target in TARGETS:
            best_w = weights[target]
            best_score = -1.0
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
    scale_grid = {
        "forward_count": [0.75, 0.85, 0.95, 1.00, 1.05],
        "comment_count": [0.75, 0.85, 0.95, 1.00, 1.05],
        "like_count": [0.75, 0.85, 0.95, 1.00, 1.05],
    }
    thr_grid = {
        "forward_count": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "comment_count": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "like_count": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    }
    clip_grid = {
        "forward_count": [200.0, 300.0, 500.0],
        "comment_count": [200.0, 300.0, 500.0],
        "like_count": [200.0, 300.0, 500.0],
    }

    best_mode = "round"
    best_score = official_score(y_true, apply_postprocess(pred_cont, config, int_mode=best_mode))

    for _ in range(2):
        for target in TARGETS:
            idx = TARGETS.index(target)
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
            f.write(
                f"{row['uid']}\t{row['mid']}\t"
                f"{int(pred_int[i, 0])},{int(pred_int[i, 1])},{int(pred_int[i, 2])}\n"
            )



def summarize_split(train_df: pd.DataFrame, valid_df: Optional[pd.DataFrame] = None) -> None:
    print("=" * 80)
    print(f"训练样本数: {len(train_df):,}")
    print(f"训练时间范围: {train_df['time'].min()} -> {train_df['time'].max()}")
    if valid_df is not None:
        print(f"验证样本数: {len(valid_df):,}")
        print(f"验证时间范围: {valid_df['time'].min()} -> {valid_df['time'].max()}")
    print("=" * 80)



def run_validate(args) -> None:
    print("[1/7] 读取训练数据...")
    df = read_data(args.train_path, is_train=True)
    df = df.sort_values(["time", "uid", "mid"]).reset_index(drop=True)

    valid_start = pd.to_datetime(args.valid_start)
    train_df = df[df["time"] < valid_start].copy().reset_index(drop=True)
    valid_df = df[df["time"] >= valid_start].copy().reset_index(drop=True)

    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("按当前 valid_start 划分后，训练集或验证集为空，请调整日期。")

    summarize_split(train_df, valid_df)

    print("[2/7] 构造数值特征...")
    train_bundle = build_feature_bundle(train_df)
    valid_feat_df = prepare_infer_bundle(valid_df, train_bundle)

    X_train_num = train_bundle.numeric_df[train_bundle.feature_cols].copy()
    X_valid_num = valid_feat_df[train_bundle.feature_cols].copy()

    print("[3/7] 构造文本特征...")
    vectorizer = fit_text_vectorizer(train_df["content"], args.tfidf_max_features, args.tfidf_min_df)
    X_train_sparse, scaler = make_sparse_text_features(
        vectorizer, train_df["content"], train_bundle.numeric_df, train_bundle.feature_cols, fit_scaler=True
    )
    X_valid_sparse, _ = make_sparse_text_features(
        vectorizer, valid_df["content"], valid_feat_df, train_bundle.feature_cols, scaler=scaler, fit_scaler=False
    )

    y_train = train_df[TARGETS].copy()
    y_valid = valid_df[TARGETS].copy()

    print("[4/7] 训练模型...")
    lgb_models, ridge_models, lgb_preds, ridge_preds, best_iters = fit_models_validate(
        X_train_num, X_valid_num, X_train_sparse, X_valid_sparse, y_train, y_valid, args.seed, args.num_threads
    )

    print("[5/7] 融合与后处理调优...")
    y_valid_np = y_valid.values.astype(np.int32)
    blend_weights = tune_blend_weights(y_valid_np, lgb_preds, ridge_preds)
    pred_valid_cont = blend_predictions(lgb_preds, ridge_preds, blend_weights)

    base_pp = {t: {"scale": 1.0, "zero_thr": 0.5, "max_clip": 500.0} for t in TARGETS}
    base_pred_int = apply_postprocess(pred_valid_cont, base_pp, int_mode="round")
    base_score = official_score(y_valid_np, base_pred_int)

    tuned_pp, int_mode, tuned_score = tune_postprocess(y_valid_np, pred_valid_cont)
    tuned_pred_int = apply_postprocess(pred_valid_cont, tuned_pp, int_mode=int_mode)

    print("[6/7] 输出结果...")
    os.makedirs(args.output_dir, exist_ok=True)
    save_validation_predictions(
        os.path.join(args.output_dir, "valid_predictions.csv"),
        valid_df,
        y_valid_np,
        tuned_pred_int,
    )

    config = {
        "valid_start": args.valid_start,
        "seed": args.seed,
        "tfidf_max_features": args.tfidf_max_features,
        "tfidf_min_df": args.tfidf_min_df,
        "feature_cols": train_bundle.feature_cols,
        "best_iterations": best_iters,
        "blend_weights": blend_weights,
        "postprocess": tuned_pp,
        "int_mode": int_mode,
        "base_score": base_score,
        "tuned_score": tuned_score,
    }
    with open(os.path.join(args.output_dir, "validate_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    bundle = {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "fill_values": train_bundle.fill_values,
        "feature_cols": train_bundle.feature_cols,
        "user_stats": train_bundle.user_stats,
        "lgb_models": lgb_models,
        "ridge_models": ridge_models,
        "blend_weights": blend_weights,
        "postprocess": tuned_pp,
        "int_mode": int_mode,
        "best_iterations": best_iters,
    }
    joblib.dump(bundle, os.path.join(args.output_dir, "validate_artifacts.joblib"))

    print("[7/7] 验证完成")
    print(f"基础分数(默认融合+默认后处理): {base_score:.6f}")
    print(f"调优后分数: {tuned_score:.6f}")
    print(f"blend_weights: {blend_weights}")
    print(f"int_mode: {int_mode}")
    print(f"postprocess: {json.dumps(tuned_pp, ensure_ascii=False)}")
    print(f"best_iterations: {best_iters}")
    print(f"结果目录: {args.output_dir}")



def load_validate_config(path: Optional[str]) -> Dict[str, object]:
    if path is None or not os.path.exists(path):
        return {
            "best_iterations": DEFAULT_FULL_ITERS,
            "blend_weights": DEFAULT_BLEND,
            "postprocess": {t: {"scale": 1.0, "zero_thr": 0.5, "max_clip": 500.0} for t in TARGETS},
            "int_mode": "round",
        }
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg



def run_full(args) -> None:
    print("[1/6] 读取训练/测试数据...")
    train_df = read_data(args.train_path, is_train=True)
    train_df = train_df.sort_values(["time", "uid", "mid"]).reset_index(drop=True)
    test_df = read_data(args.test_path, is_train=False)
    test_df = test_df.sort_values(["time", "uid", "mid"]).reset_index(drop=True)

    summarize_split(train_df)
    print(f"测试样本数: {len(test_df):,}")
    print(f"测试时间范围: {test_df['time'].min()} -> {test_df['time'].max()}")

    print("[2/6] 构造数值特征...")
    train_bundle = build_feature_bundle(train_df)
    test_feat_df = prepare_infer_bundle(test_df, train_bundle)

    X_train_num = train_bundle.numeric_df[train_bundle.feature_cols].copy()
    X_test_num = test_feat_df[train_bundle.feature_cols].copy()

    print("[3/6] 构造文本特征...")
    vectorizer = fit_text_vectorizer(train_df["content"], args.tfidf_max_features, args.tfidf_min_df)
    X_train_sparse, scaler = make_sparse_text_features(
        vectorizer, train_df["content"], train_bundle.numeric_df, train_bundle.feature_cols, fit_scaler=True
    )
    X_test_sparse, _ = make_sparse_text_features(
        vectorizer, test_df["content"], test_feat_df, train_bundle.feature_cols, scaler=scaler, fit_scaler=False
    )

    print("[4/6] 训练全量模型...")
    cfg = load_validate_config(args.validate_config)
    best_iterations = cfg.get("best_iterations", DEFAULT_FULL_ITERS)
    blend_weights = cfg.get("blend_weights", DEFAULT_BLEND)
    postprocess = cfg.get("postprocess", {t: {"scale": 1.0, "zero_thr": 0.5, "max_clip": 500.0} for t in TARGETS})
    int_mode = cfg.get("int_mode", "round")

    lgb_models, ridge_models = fit_models_full(
        X_train_num, X_train_sparse, train_df[TARGETS].copy(), args.seed, best_iterations, args.num_threads
    )

    print("[5/6] 预测测试集...")
    lgb_preds = {}
    ridge_preds = {}
    for target in TARGETS:
        lgb_preds[target] = lgb_models[target].predict(X_test_num.values)
        ridge_preds[target] = ridge_models[target].predict(X_test_sparse)
    pred_test_cont = blend_predictions(lgb_preds, ridge_preds, blend_weights)
    pred_test_int = apply_postprocess(pred_test_cont, postprocess, int_mode=int_mode)

    print("[6/6] 写出提交文件...")
    os.makedirs(args.output_dir, exist_ok=True)
    submission_path = os.path.join(args.output_dir, args.submission_name)
    write_submission(submission_path, test_df, pred_test_int)

    bundle = {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "fill_values": train_bundle.fill_values,
        "feature_cols": train_bundle.feature_cols,
        "user_stats": train_bundle.user_stats,
        "lgb_models": lgb_models,
        "ridge_models": ridge_models,
        "blend_weights": blend_weights,
        "postprocess": postprocess,
        "int_mode": int_mode,
        "best_iterations": best_iterations,
    }
    joblib.dump(bundle, os.path.join(args.output_dir, "full_artifacts.joblib"))
    print(f"提交文件已保存: {submission_path}")
    print(f"模型与特征已保存: {os.path.join(args.output_dir, 'full_artifacts.joblib')}")



CONFIG = {
    # 运行模式: "validate" 或 "full"
    "mode": "full",

    # 数据路径
    "train_path": "/mnt/h/Datasets/WeiboData/weibo_train_data(new)/weibo_train_data.txt",
    "test_path": "/mnt/h/Datasets/WeiboData/weibo_predict_data(new)/weibo_predict_data.txt",  # validate 模式下可忽略

    # 时间切分: 训练集 < valid_start, 验证集 >= valid_start
    "valid_start": "2015-07-01",

    # 输出目录与提交文件名
    "output_dir": "./outputs/weibo_baseline",
    "submission_name": "submission.txt",

    # 文本特征参数
    "tfidf_max_features": 50000,
    "tfidf_min_df": 5,

    # 训练参数
    "seed": 42,
    "num_threads": 8,

    # full 模式下可复用 validate 产生的配置；没有就填 None
    "validate_config": None,
}


def dict_to_namespace(config: Dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**config)



def run_from_config(config: Dict[str, object]) -> None:
    args = dict_to_namespace(config)
    set_seed(args.seed)

    if args.mode == "validate":
        run_validate(args)
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
