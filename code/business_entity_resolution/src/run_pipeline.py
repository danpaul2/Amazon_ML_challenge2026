

import os
import sys
import argparse
import time
import gc
import re
import pickle
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import lightgbm as lgb
from unidecode import unidecode
from rapidfuzz import fuzz

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SRC_DIR)
CODE_DIR = os.path.dirname(PKG_DIR)
BASE_DIR = os.path.dirname(CODE_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(PKG_DIR, "models")
PKG_MODEL_DIR = MODEL_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PKG_MODEL_DIR, exist_ok=True)

MAX_POSTING_LIST = 200
MAX_CANDIDATES_PER_S1 = 300
TRAIN_SAMPLE_S1 = 40000
TRAIN_NEG_RATIO = 8
VAL_FRACTION = 0.15
RANDOM_SEED = 42
NUM_BOOST_ROUNDS = 600
DEFAULT_THRESHOLD = 0.90
TUNE_S1 = 12000

LEET_MAP = str.maketrans("01345$", "oieasS")

LEGAL_STOP_FEAT = frozenset({
    "inc", "llc", "corp", "corporation", "pvt", "ltd", "private", "limited",
    "co", "company", "services", "service", "enterprise", "enterprises",
    "group", "holdings", "holding", "sarl", "sasu", "sci", "sas", "sa",
    "eurl", "the", "and", "of", "for", "in", "at", "mr", "mrs", "ms",
    "dr", "smt", "shri", "shree",
})

STOP_ADDR_FEAT = frozenset({
    "road", "street", "avenue", "lane", "drive", "near", "floor", "flat",
    "unit", "opp", "opposite", "behind", "bldg", "building", "null", "c/o",
})

LEGAL_STOP = LEGAL_STOP_FEAT | frozenset({
    "associates", "solutions", "consulting", "management", "technology",
    "international", "national", "construction", "manufacturing",
    "development", "distribution",
})

STOP_ADDR = STOP_ADDR_FEAT | frozenset({
    "plot", "sector", "phase", "block", "tower", "room", "apartment", "suite",
    "rue", "boulevard", "allee", "place", "chemin", "impasse",
})

_RE_DOMAIN = re.compile(r"\.(com|org|net|in|co|io|fr|edu|gov)\b", re.I)
_RE_NON_ALNUM = re.compile(r"[^\w\s]")
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_SQUEEZE = re.compile(r"(\w)\1+")
_RE_DIGITS = re.compile(r"\b(\d+)\b")

FEATURE_NAMES = [
    "n_sort", "n_set", "n_ratio", "n_partial",
    "s_n_sort", "s_n_set", "max_name_sim",
    "shared_name", "jaccard_name", "sub_match",
    "a_sort", "a_set", "a_ratio", "a_partial",
    "has_empty_addr",
    "shared_addr", "jaccard_addr",
    "digits_overlap", "exact_addr",
    "name_addr_conflict",
    "combined_score",
]


def log(msg):
    msg_safe = str(msg).encode("ascii", errors="replace").decode("ascii")
    print(f"[{time.strftime('%H:%M:%S')}] {msg_safe}", flush=True)


def clean_name(name: str) -> str:
    if not name:
        return ""
    n = unidecode(name).lower()
    n = n.translate(LEET_MAP)
    n = _RE_DOMAIN.sub("", n)
    n = _RE_NON_ALNUM.sub(" ", n)
    return _RE_MULTI_SPACE.sub(" ", n).strip()


def clean_addr(addr: str) -> str:
    if not addr:
        return ""
    a = unidecode(addr).lower()
    a = _RE_NON_ALNUM.sub(" ", a)
    return _RE_MULTI_SPACE.sub(" ", a).strip()


def squeeze(text: str) -> str:
    return _RE_SQUEEZE.sub(r"\1", text)


def name_words(n: str):
    return [w for w in n.split() if len(w) >= 3 and w not in LEGAL_STOP]


def addr_words(a: str):
    return [w for w in a.split() if len(w) >= 4 and w not in STOP_ADDR]


def addr_digits(a: str):
    out = []
    for d in _RE_DIGITS.findall(a):
        try:
            val = int(d)
            if 0 < val <= 999999:
                out.append(str(val))
        except ValueError:
            pass
    return out


def get_blocking_keys(name: str, addr: str, country: str) -> set:
    keys = set()
    c = (country or "").lower().strip()
    n = clean_name(name)
    n_words = name_words(n)
    n_squeezed = [squeeze(w) for w in n_words]

    for w in n_words:
        if len(w) >= 4:
            keys.add(f"{c}_nw_{w}")
    for w in n_squeezed:
        if len(w) >= 4:
            keys.add(f"{c}_nw_{w}")

    if n_words:
        comp = "".join(n_words[:3])
        if len(comp) >= 5:
            keys.add(f"{c}_nc_{comp[:12]}")

    if len(n_words) >= 2:
        keys.add(f"{c}_n2_{n_words[0]}_{n_words[1]}")
        if len(n_squeezed) >= 2:
            keys.add(f"{c}_n2_{n_squeezed[0]}_{n_squeezed[1]}")

    a = clean_addr(addr)
    digits = addr_digits(a)
    a_w = addr_words(a)

    for d in digits[:3]:
        for w in a_w[:6]:
            keys.add(f"{c}_num_{d}_{w}")
    for i in range(min(len(a_w) - 1, 5)):
        keys.add(f"{c}_a2_{a_w[i]}_{a_w[i + 1]}")
    return keys


def extract_features(s1_name, s1_addr, sx_name, sx_addr):
    n1 = clean_name(s1_name)
    nx = clean_name(sx_name)
    a1 = clean_addr(s1_addr)
    ax = clean_addr(sx_addr)

    n_sort = fuzz.token_sort_ratio(n1, nx) / 100.0
    n_set = fuzz.token_set_ratio(n1, nx) / 100.0
    n_ratio = fuzz.ratio(n1, nx) / 100.0
    n_partial = fuzz.partial_ratio(n1, nx) / 100.0

    s_n1 = squeeze(n1)
    s_nx = squeeze(nx)
    s_n_sort = fuzz.token_sort_ratio(s_n1, s_nx) / 100.0
    s_n_set = fuzz.token_set_ratio(s_n1, s_nx) / 100.0
    max_name_sim = max(n_sort, n_set, s_n_sort, s_n_set)

    w1_n = set(w for w in n1.split() if len(w) >= 3 and w not in LEGAL_STOP_FEAT)
    wx_n = set(w for w in nx.split() if len(w) >= 3 and w not in LEGAL_STOP_FEAT)
    shared_name = len(w1_n & wx_n)
    union_n = len(w1_n | wx_n)
    jaccard_name = shared_name / union_n if union_n > 0 else 0.0

    nx_comp = "".join(nx.split())
    n1_comp = "".join(n1.split())
    sub_match = 1.0 if (
        any(len(w) >= 4 and w in nx_comp for w in w1_n)
        or any(len(w) >= 4 and w in n1_comp for w in wx_n)
    ) else 0.0

    has_empty_addr = 1.0 if (not a1 or not ax) else 0.0
    if not has_empty_addr:
        a_sort = fuzz.token_sort_ratio(a1, ax) / 100.0
        a_set = fuzz.token_set_ratio(a1, ax) / 100.0
        a_ratio = fuzz.ratio(a1, ax) / 100.0
        a_partial = fuzz.partial_ratio(a1, ax) / 100.0
    else:
        a_sort = a_set = a_ratio = a_partial = 0.0

    w1_a = set(w for w in a1.split() if len(w) >= 4 and w not in STOP_ADDR_FEAT)
    wx_a = set(w for w in ax.split() if len(w) >= 4 and w not in STOP_ADDR_FEAT)
    shared_addr = len(w1_a & wx_a)
    union_a = len(w1_a | wx_a)
    jaccard_addr = shared_addr / union_a if union_a > 0 else 0.0

    d1 = set(addr_digits(a1))
    dx = set(addr_digits(ax))
    digits_overlap = 1.0 if (d1 and dx and bool(d1 & dx)) else 0.0
    exact_addr = 1.0 if (digits_overlap and shared_addr >= 2) else 0.0
    name_addr_conflict = 1.0 if (not has_empty_addr and max_name_sim < 0.40 and not exact_addr) else 0.0
    combined_score = 0.6 * max_name_sim + 0.4 * a_set if not has_empty_addr else max_name_sim

    return [
        n_sort, n_set, n_ratio, n_partial,
        s_n_sort, s_n_set, max_name_sim,
        float(shared_name), jaccard_name, sub_match,
        a_sort, a_set, a_ratio, a_partial,
        has_empty_addr,
        float(shared_addr), jaccard_addr,
        digits_overlap, exact_addr,
        name_addr_conflict,
        combined_score,
    ]


def precision_gate(s1_name, s1_addr, sx_name, sx_addr, score, threshold):
    n1 = clean_name(s1_name)
    nx = clean_name(sx_name)
    a1 = clean_addr(s1_addr)
    ax = clean_addr(sx_addr)

    w1_n = set(w for w in n1.split() if len(w) >= 3 and w not in LEGAL_STOP_FEAT)
    wx_n = set(w for w in nx.split() if len(w) >= 3 and w not in LEGAL_STOP_FEAT)
    sq1 = set(squeeze(w) for w in w1_n)
    sqx = set(squeeze(w) for w in wx_n)
    nx_comp = "".join(nx.split())
    n1_comp = "".join(n1.split())
    substring_match = any(len(w) >= 4 and w in nx_comp for w in w1_n) or any(
        len(w) >= 4 and w in n1_comp for w in wx_n
    )
    has_name_signal = bool(w1_n & wx_n) or bool(sq1 & sqx) or substring_match

    d1 = set(addr_digits(a1))
    dx = set(addr_digits(ax))
    digits_match = bool(d1 & dx) if (d1 and dx) else False
    w1_a = set(w for w in a1.split() if len(w) >= 4 and w not in STOP_ADDR_FEAT)
    wx_a = set(w for w in ax.split() if len(w) >= 4 and w not in STOP_ADDR_FEAT)
    addr_overlap = len(w1_a & wx_a)
    exact_address_match = digits_match and addr_overlap >= 2

    rescue = (
        score >= (threshold - 0.06)
        and score < threshold
        and max(fuzz.token_set_ratio(squeeze(n1), squeeze(nx)) / 100.0, fuzz.token_sort_ratio(n1, nx) / 100.0) >= 0.96
        and (exact_address_match or not a1 or not ax)
    )
    if score < threshold and not rescue:
        return False

    if not has_name_signal and not exact_address_match:
        return False
    if not has_name_signal and exact_address_match:
        return digits_match and addr_overlap >= 3
    if a1 and ax and w1_a and wx_a and addr_overlap == 0 and not digits_match:
        if fuzz.token_sort_ratio(n1, nx) < 85:
            return False
    return True


def stream_load_tsv(filepath: str, keep_ids=None, keep_countries=None) -> dict:
    log(f"Loading {os.path.basename(filepath)}...")
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 1:
                continue
            eid = parts[0]
            if keep_ids is not None and eid not in keep_ids:
                continue
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""
            if keep_countries is not None and country not in keep_countries:
                continue
            data[eid] = (name, addr, country)
    log(f"  -> {len(data):,} records")
    return data


def load_ground_truth(filepath: str) -> dict:
    log(f"Loading {os.path.basename(filepath)}...")
    gt = {}
    with open(filepath, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if len(parts) > 1 and parts[1]:
                gt[s1_id] = set(m.strip() for m in parts[1].split(",") if m.strip())
            else:
                gt[s1_id] = set()
    log(f"  -> {len(gt):,} records")
    return gt


def build_inverted_index(sx_data: dict) -> dict:
    log("Building inverted index...")
    t0 = time.time()
    inv_index = defaultdict(list)
    count = 0
    for eid, (name, addr, country) in sx_data.items():
        for k in get_blocking_keys(name, addr, country):
            inv_index[k].append(eid)
        count += 1
        if count % 1_000_000 == 0:
            log(f"  Indexed {count:,} records...")
    pruned = 0
    for k in [k for k, v in inv_index.items() if len(v) > MAX_POSTING_LIST]:
        del inv_index[k]
        pruned += 1
    log(f"  -> Index built in {time.time() - t0:.1f}s: {len(inv_index):,} keys ({pruned:,} pruned)")
    return inv_index


def find_candidates(s1_record: tuple, inv_index: dict) -> list:
    name, addr, country = s1_record
    keys = get_blocking_keys(name, addr, country)
    scores = defaultdict(float)
    for k in keys:
        posting = inv_index.get(k)
        if not posting:
            continue
        w = 1.0 / len(posting)
        for eid in posting:
            scores[eid] += w
    if not scores:
        return []
    if len(scores) <= MAX_CANDIDATES_PER_S1:
        return list(scores.keys())
    return [eid for eid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:MAX_CANDIDATES_PER_S1]]


def model_paths():
    model_path = os.path.join(MODEL_DIR, "lgbm_model.txt")
    th_path = os.path.join(MODEL_DIR, "threshold.pkl")
    return model_path, th_path


def save_threshold(threshold: float):
    _, th_path = model_paths()
    with open(th_path, "wb") as f:
        pickle.dump(float(threshold), f)
    with open(os.path.join(PKG_MODEL_DIR, "threshold.pkl"), "wb") as f:
        pickle.dump(float(threshold), f)
    log(f"  Threshold {threshold:.4f} saved")


def load_model_and_threshold():
    model_path, th_path = model_paths()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = lgb.Booster(model_file=model_path)
    if os.path.exists(th_path):
        with open(th_path, "rb") as f:
            threshold = float(pickle.load(f))
    else:
        threshold = DEFAULT_THRESHOLD
        log(f"No threshold.pkl; using default {threshold:.3f}")
    log(f"Loaded model ({model.num_trees()} trees) threshold={threshold:.4f}")
    names = model.feature_name()
    if names != FEATURE_NAMES:
        log(f"WARNING: model feature names differ: {names}")
    return model, threshold


def entity_f05(true_m, pred_m):
    if not true_m and not pred_m:
        return 1.0
    if not pred_m or not true_m:
        return 0.0
    tp = len(pred_m & true_m)
    prec = tp / len(pred_m)
    rec = tp / len(true_m)
    if prec + rec == 0:
        return 0.0
    return (1.25 * prec * rec) / (0.25 * prec + rec)


def score_pairs(model, s1_data, sx_data, pairs, batch_size=150000):
    """pairs: list of (s1_id, m_id). Returns list of scores aligned with pairs."""
    scores = np.empty(len(pairs), dtype=np.float32)
    i = 0
    n = len(pairs)
    while i < n:
        j = min(i + batch_size, n)
        feats = np.array(
            [
                extract_features(s1_data[s][0], s1_data[s][1], sx_data[m][0], sx_data[m][1])
                for s, m in pairs[i:j]
                if s in s1_data and m in sx_data
            ],
            dtype=np.float32,
        )
        if feats.shape[0] != (j - i):
            chunk = []
            for s, m in pairs[i:j]:
                if s in s1_data and m in sx_data:
                    chunk.append(extract_features(s1_data[s][0], s1_data[s][1], sx_data[m][0], sx_data[m][1]))
                else:
                    chunk.append([0.0] * 21)
            feats = np.array(chunk, dtype=np.float32)
        scores[i:j] = model.predict(feats)
        i = j
    return scores


def decide_matches(s1_id, s1_data, sx_data, cand_ids, cand_scores, threshold):
    s1_name, s1_addr, _ = s1_data[s1_id]
    out = []
    seen = set()
    for m_id, sc in zip(cand_ids, cand_scores):
        if m_id in seen or m_id not in sx_data:
            continue
        seen.add(m_id)
        sx_name, sx_addr, _ = sx_data[m_id]
        if precision_gate(s1_name, s1_addr, sx_name, sx_addr, float(sc), threshold):
            out.append(m_id)
    return out


def run_train_pipeline(sample_fraction=None):
    log("=" * 70)
    log("TRAINING PIPELINE")
    log("=" * 70)
    total_start = time.time()
    gt = load_ground_truth(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))

    if sample_fraction and sample_fraction < 1.0:
        n_train = max(1000, int(len(gt) * sample_fraction))
    else:
        n_train = TRAIN_SAMPLE_S1

    np.random.seed(RANDOM_SEED)
    all_s1_ids = list(gt.keys())
    np.random.shuffle(all_s1_ids)
    train_s1_ids = all_s1_ids[:n_train]
    log(f"Using {n_train:,} S1 entities")

    s1_data = stream_load_tsv(os.path.join(TRAIN_DIR, "train_source1.tsv"), keep_ids=set(train_s1_ids))
    s2_data = stream_load_tsv(os.path.join(TRAIN_DIR, "train_source2.tsv"))
    s3_data = stream_load_tsv(os.path.join(TRAIN_DIR, "train_source3.tsv"))
    sx_data = {}
    sx_data.update(s2_data)
    sx_data.update(s3_data)
    del s2_data, s3_data
    gc.collect()

    inv_index = build_inverted_index(sx_data)
    log("Generating candidate pairs...")
    candidate_pairs = {}
    for s1_id in train_s1_ids:
        if s1_id in s1_data:
            candidate_pairs[s1_id] = set(find_candidates(s1_data[s1_id], inv_index))

    total_true = caught = 0
    for s1_id in train_s1_ids:
        true_matches = gt.get(s1_id, set())
        cands = candidate_pairs.get(s1_id, set())
        total_true += len(true_matches)
        caught += len(true_matches & cands)
    blocking_recall = caught / max(total_true, 1) * 100
    avg_cands = sum(len(v) for v in candidate_pairs.values()) / max(len(candidate_pairs), 1)
    log(f"  Blocking recall: {caught:,}/{total_true:,} = {blocking_recall:.2f}%")
    log(f"  Avg candidates per S1: {avg_cands:.1f}")
    del inv_index
    gc.collect()

    val_size = int(len(train_s1_ids) * VAL_FRACTION)
    val_s1_ids = set(train_s1_ids[:val_size])
    tr_s1_ids = set(train_s1_ids[val_size:])

    log("Extracting training features...")
    X_train, y_train = [], []
    for s1_id in tr_s1_ids:
        if s1_id not in s1_data:
            continue
        s1_name, s1_addr, _ = s1_data[s1_id]
        true_matches = gt.get(s1_id, set())
        cands = candidate_pairs.get(s1_id, set())
        for m_id in true_matches:
            if m_id in sx_data:
                X_train.append(extract_features(s1_name, s1_addr, sx_data[m_id][0], sx_data[m_id][1]))
                y_train.append(1)
        neg_cands = list(cands - true_matches)
        np.random.shuffle(neg_cands)
        n_neg = min(len(neg_cands), TRAIN_NEG_RATIO * max(1, len(true_matches)))
        for m_id in neg_cands[:n_neg]:
            if m_id in sx_data:
                X_train.append(extract_features(s1_name, s1_addr, sx_data[m_id][0], sx_data[m_id][1]))
                y_train.append(0)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    log(f"  Training data: {len(X_train):,} pairs ({int(y_train.sum()):,} pos)")

    train_dataset = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.04,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "verbose": -1,
        "seed": RANDOM_SEED,
        "n_jobs": -1,
    }
    model = lgb.train(params, train_dataset, num_boost_round=NUM_BOOST_ROUNDS)
    model_path, _ = model_paths()
    model.save_model(model_path)
    model.save_model(os.path.join(PKG_MODEL_DIR, "lgbm_model.txt"))
    log(f"  Model saved to {model_path}")
    del X_train, y_train, train_dataset
    gc.collect()

    best_th, best_f05 = optimize_threshold(model, s1_data, sx_data, gt, val_s1_ids, candidate_pairs)
    save_threshold(best_th)
    log(f"TRAINING COMPLETE Macro F_0.5={best_f05:.6f} th={best_th:.3f} blocking={blocking_recall:.2f}% time={(time.time()-total_start)/60:.1f}m")
    return model, best_th


def optimize_threshold(model, s1_data, sx_data, gt, val_s1_ids, candidate_pairs):
    log(f"Scoring {len(val_s1_ids):,} validation S1 entities...")
    s1_val_scores = defaultdict(list)
    batch_features, batch_ids = [], []
    for s1_id in val_s1_ids:
        if s1_id not in s1_data:
            continue
        s1_name, s1_addr, _ = s1_data[s1_id]
        for m_id in candidate_pairs.get(s1_id, []):
            if m_id not in sx_data:
                continue
            feat = extract_features(s1_name, s1_addr, sx_data[m_id][0], sx_data[m_id][1])
            batch_features.append(feat)
            batch_ids.append((s1_id, m_id))
            if len(batch_features) >= 100_000:
                scores = model.predict(np.array(batch_features, dtype=np.float32))
                for (sid, mid), sc in zip(batch_ids, scores):
                    s1_val_scores[sid].append((mid, float(sc)))
                batch_features.clear()
                batch_ids.clear()
    if batch_features:
        scores = model.predict(np.array(batch_features, dtype=np.float32))
        for (sid, mid), sc in zip(batch_ids, scores):
            s1_val_scores[sid].append((mid, float(sc)))

    log("Optimizing threshold for Macro F_0.5 (with precision gate)...")
    best_th, best_f05 = 0.90, -1.0
    for th in np.arange(0.70, 0.995, 0.005):
        f05_list = []
        for s1_id in val_s1_ids:
            true_m = gt.get(s1_id, set())
            pred_m = set()
            if s1_id in s1_data:
                s1_name, s1_addr, _ = s1_data[s1_id]
                for m_id, sc in s1_val_scores.get(s1_id, []):
                    if m_id not in sx_data:
                        continue
                    if precision_gate(s1_name, s1_addr, sx_data[m_id][0], sx_data[m_id][1], sc, th):
                        pred_m.add(m_id)
            f05_list.append(entity_f05(true_m, pred_m))
        macro = float(np.mean(f05_list)) if f05_list else 0.0
        if macro > best_f05:
            best_f05 = macro
            best_th = float(th)
            log(f"  new best th={best_th:.3f} F_0.5={best_f05:.6f}")
    log(f"  Best threshold: {best_th:.3f}  Macro F_0.5: {best_f05:.6f}")
    return best_th, best_f05


def run_tune_pipeline(n_s1=TUNE_S1):
    """Tune threshold on a random train holdout using the existing booster."""
    log("=" * 70)
    log("THRESHOLD TUNE (existing LightGBM)")
    log("=" * 70)
    model, _ = load_model_and_threshold()
    gt = load_ground_truth(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))
    np.random.seed(RANDOM_SEED)
    all_s1_ids = list(gt.keys())
    np.random.shuffle(all_s1_ids)
    val_ids = all_s1_ids[:n_s1]
    val_set = set(val_ids)
    s1_data = stream_load_tsv(os.path.join(TRAIN_DIR, "train_source1.tsv"), keep_ids=val_set)
    countries = sorted({s1_data[i][2] for i in s1_data})
    log(f"Tune countries: {countries}")

    candidate_pairs = defaultdict(set)
    sx_keep = {}
    for country in countries:
        log(f"--- blocking country={country!r} ---")
        s2 = stream_load_tsv(os.path.join(TRAIN_DIR, "train_source2.tsv"), keep_countries={country})
        s3 = stream_load_tsv(os.path.join(TRAIN_DIR, "train_source3.tsv"), keep_countries={country})
        sx = {}
        sx.update(s2)
        sx.update(s3)
        del s2, s3
        gc.collect()
        inv = build_inverted_index(sx)
        needed = set()
        n_c = 0
        for s1_id in val_ids:
            rec = s1_data.get(s1_id)
            if not rec or rec[2] != country:
                continue
            cands = find_candidates(rec, inv)
            candidate_pairs[s1_id].update(cands)
            needed.update(cands)
            n_c += 1
        log(f"  blocked {n_c:,} S1 in {country}; keeping {len(needed):,} SX cands")
        for eid in needed:
            rec = sx.get(eid)
            if rec is not None:
                sx_keep[eid] = rec
        del inv, sx, needed
        gc.collect()

    total_true = caught = 0
    for s1_id in val_ids:
        tm = gt.get(s1_id, set())
        total_true += len(tm)
        caught += len(tm & candidate_pairs[s1_id])
    log(f"Blocking recall on tune set: {caught:,}/{total_true:,} = {caught / max(total_true, 1) * 100:.2f}%")

    best_th, best_f05 = optimize_threshold(model, s1_data, sx_keep, gt, val_set, candidate_pairs)
    save_threshold(best_th)
    log(f"TUNE COMPLETE F_0.5={best_f05:.6f} th={best_th:.3f}")
    return best_th, best_f05


def load_s1_id_countries(filepath: str):
    """IDs + country only, so predict does not keep all S1 strings in RAM."""
    log(f"Indexing IDs from {os.path.basename(filepath)}...")
    ids = []
    countries_of = {}
    with open(filepath, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            eid = parts[0]
            country = parts[3] if len(parts) > 3 else ""
            ids.append(eid)
            countries_of[eid] = country
    log(f"  -> {len(ids):,} S1 IDs")
    return ids, countries_of


def process_split(model, threshold, s1_path, s2_path, s3_path, matching_path, candidate_path):
    all_s1_ids, s1_countries = load_s1_id_countries(s1_path)
    countries = sorted(set(s1_countries.values()))
    log(f"S1={len(all_s1_ids):,} countries={countries}")

    matching = {sid: [] for sid in all_s1_ids}

    cand_tmp_dir = os.path.join(OUTPUT_DIR, "_cand_parts")
    os.makedirs(cand_tmp_dir, exist_ok=True)
    part_files = []

    for country in countries:
        log(f"=== country {country!r} ===")
        t0 = time.time()
        s2 = stream_load_tsv(s2_path, keep_countries={country})
        s3 = stream_load_tsv(s3_path, keep_countries={country})
        sx = {}
        sx.update(s2)
        sx.update(s3)
        del s2, s3
        gc.collect()
        inv = build_inverted_index(sx)

        s1_data = stream_load_tsv(s1_path, keep_countries={country})
        country_s1 = [sid for sid in all_s1_ids if s1_countries[sid] == country]
        part_path = os.path.join(cand_tmp_dir, f"cand_{len(part_files)}.tsv")
        part_files.append(part_path)
        processed = 0
        batch_features, batch_meta = [], []

        def flush_batch():
            nonlocal batch_features, batch_meta
            if not batch_features:
                return
            scores = model.predict(np.array(batch_features, dtype=np.float32))
            grouped = defaultdict(list)
            for (sid, mid), sc in zip(batch_meta, scores):
                grouped[sid].append((mid, float(sc)))
            for sid, items in grouped.items():
                s1_name, s1_addr, _ = s1_data[sid]
                chosen = []
                seen = set()
                for mid, sc in items:
                    if mid in seen or mid not in sx:
                        continue
                    seen.add(mid)
                    if precision_gate(s1_name, s1_addr, sx[mid][0], sx[mid][1], sc, threshold):
                        chosen.append(mid)
                matching[sid].extend(chosen)
            batch_features = []
            batch_meta = []

        with open(part_path, "w", encoding="utf-8") as pf:
            for s1_id in country_s1:
                cands = find_candidates(s1_data[s1_id], inv)
                uniq = []
                seen = set()
                for c in cands:
                    if c not in seen and c in sx:
                        seen.add(c)
                        uniq.append(c)
                pf.write(f"{s1_id}\t{','.join(uniq)}\n")
                s1_name, s1_addr, _ = s1_data[s1_id]
                for m_id in uniq:
                    batch_features.append(extract_features(s1_name, s1_addr, sx[m_id][0], sx[m_id][1]))
                    batch_meta.append((s1_id, m_id))
                if len(batch_features) >= 200_000:
                    flush_batch()
                    gc.collect()
                processed += 1
                if processed % 100_000 == 0:
                    log(f"  {processed:,}/{len(country_s1):,} {country}")
            flush_batch()

        log(f"  {country} done in {(time.time() - t0) / 60:.1f} min")
        del inv, sx, s1_data
        gc.collect()

    log("Writing matching_results.tsv...")
    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            ids = []
            seen = set()
            for m in matching[s1_id]:
                if m not in seen:
                    seen.add(m)
                    ids.append(m)
            f.write(f"{s1_id}\t{','.join(ids)}\n")

    log("Writing candidate_pairs.tsv...")
    cand_map = {}
    for pp in part_files:
        with open(pp, "r", encoding="utf-8") as pf:
            for line in pf:
                sid, rest = line.rstrip("\n").split("\t", 1)
                cand_map[sid] = rest
    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            f.write(f"{s1_id}\t{cand_map.get(s1_id, '')}\n")

    for pp in part_files:
        try:
            os.remove(pp)
        except OSError:
            pass
    try:
        os.rmdir(cand_tmp_dir)
    except OSError:
        pass

    matched_count = sum(1 for v in matching.values() if v)
    total_matches = sum(len(set(v)) for v in matching.values())
    log(f"  {matched_count:,} S1 with >=1 match, {total_matches:,} match IDs")


def run_predict_pipeline():
    log("=" * 70)
    log("PREDICTION PIPELINE")
    log("=" * 70)
    t0 = time.time()
    model, threshold = load_model_and_threshold()
    process_split(
        model,
        threshold,
        os.path.join(TEST_DIR, "test_source1.tsv"),
        os.path.join(TEST_DIR, "test_source2.tsv"),
        os.path.join(TEST_DIR, "test_source3.tsv"),
        os.path.join(OUTPUT_DIR, "matching_results.tsv"),
        os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
    )
    log(f"PREDICTION COMPLETE in {(time.time() - t0) / 60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["train", "predict", "full", "tune"], default="predict")
    parser.add_argument("--sample", type=float, default=None)
    parser.add_argument("--tune-s1", type=int, default=TUNE_S1)
    args = parser.parse_args()

    if args.mode == "tune":
        run_tune_pipeline(n_s1=args.tune_s1)
    if args.mode in ("train", "full"):
        run_train_pipeline(sample_fraction=args.sample)
    if args.mode in ("predict", "full"):
        run_predict_pipeline()


if __name__ == "__main__":
    main()
