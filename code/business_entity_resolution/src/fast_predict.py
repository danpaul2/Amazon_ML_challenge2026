
import os
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
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
# The shipped booster and tuned threshold live beside ``src`` in this package.
MODEL_DIR = os.path.join(PKG_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_POSTING_LIST = 200
MAX_CANDIDATES_PER_S1 = 300
# Process this many S1 entities at once → limits peak pair RAM
# 5000 entities * 143 avg cands = ~715k pairs → 715k * 21 * 4 bytes ≈ 60 MB feats
BATCH_ENTITIES = 5000

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
    n = unidecode(name).lower().translate(LEET_MAP)
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


def addr_digits(a: str):
    out = []
    for d in _RE_DIGITS.findall(a):
        val = int(d)
        if 0 < val <= 999999:
            out.append(str(val))
    return out


def get_blocking_keys(cn: str, ca: str, country: str) -> set:
    keys = set()
    c = (country or "").lower().strip()
    n_words = [w for w in cn.split() if len(w) >= 3 and w not in LEGAL_STOP]
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
    digits = addr_digits(ca)
    a_w = [w for w in ca.split() if len(w) >= 4 and w not in STOP_ADDR]
    for d in digits[:3]:
        for w in a_w[:6]:
            keys.add(f"{c}_num_{d}_{w}")
    for i in range(min(len(a_w) - 1, 5)):
        keys.add(f"{c}_a2_{a_w[i]}_{a_w[i + 1]}")
    return keys


def build_inverted_index(sx_clean: dict) -> dict:
    """sx_clean: eid -> (cn, ca, country)"""
    log("Building inverted index...")
    t0 = time.time()
    inv_index = defaultdict(list)
    count = 0
    for eid, (cn, ca, country) in sx_clean.items():
        for k in get_blocking_keys(cn, ca, country):
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


def find_candidates_fast(cn: str, ca: str, country: str, inv_index: dict) -> list:
    keys = get_blocking_keys(cn, ca, country)
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


def extract_features_batch(s1n_list, s1a_list, sxn_list, sxa_list):
    """
    Compute 21 features for N pairs. Pre-cleaned strings.
    Returns np.ndarray shape (N, 21).
    """
    N = len(s1n_list)
    if N == 0:
        return np.empty((0, 21), dtype=np.float32)

    s1sq = [squeeze(n) for n in s1n_list]
    sxsq = [squeeze(n) for n in sxn_list]

    n_sort    = np.array([fuzz.token_sort_ratio(a, b) for a, b in zip(s1n_list, sxn_list)], np.float32) / 100
    n_set     = np.array([fuzz.token_set_ratio(a, b)  for a, b in zip(s1n_list, sxn_list)], np.float32) / 100
    n_ratio   = np.array([fuzz.ratio(a, b)             for a, b in zip(s1n_list, sxn_list)], np.float32) / 100
    n_partial = np.array([fuzz.partial_ratio(a, b)     for a, b in zip(s1n_list, sxn_list)], np.float32) / 100
    s_n_sort  = np.array([fuzz.token_sort_ratio(a, b)  for a, b in zip(s1sq, sxsq)], np.float32) / 100
    s_n_set   = np.array([fuzz.token_set_ratio(a, b)   for a, b in zip(s1sq, sxsq)], np.float32) / 100
    max_name_sim = np.maximum(np.maximum(n_sort, n_set), np.maximum(s_n_sort, s_n_set))

    shared_name  = np.empty(N, np.float32)
    jaccard_name = np.empty(N, np.float32)
    sub_match    = np.empty(N, np.float32)
    for i in range(N):
        w1 = set(w for w in s1n_list[i].split() if len(w) >= 3 and w not in LEGAL_STOP_FEAT)
        wx = set(w for w in sxn_list[i].split() if len(w) >= 3 and w not in LEGAL_STOP_FEAT)
        sh = len(w1 & wx)
        un = len(w1 | wx)
        shared_name[i] = sh
        jaccard_name[i] = sh / un if un else 0.0
        nx_c = "".join(sxn_list[i].split())
        n1_c = "".join(s1n_list[i].split())
        sub_match[i] = 1.0 if (
            any(len(w) >= 4 and w in nx_c for w in w1)
            or any(len(w) >= 4 and w in n1_c for w in wx)
        ) else 0.0

    has_empty_addr = np.array([1.0 if (not a or not b) else 0.0
                                for a, b in zip(s1a_list, sxa_list)], np.float32)

    a_sort    = np.zeros(N, np.float32)
    a_set     = np.zeros(N, np.float32)
    a_ratio   = np.zeros(N, np.float32)
    a_partial = np.zeros(N, np.float32)
    for i in range(N):
        if s1a_list[i] and sxa_list[i]:
            a_sort[i]    = fuzz.token_sort_ratio(s1a_list[i], sxa_list[i]) / 100
            a_set[i]     = fuzz.token_set_ratio(s1a_list[i],  sxa_list[i]) / 100
            a_ratio[i]   = fuzz.ratio(s1a_list[i],            sxa_list[i]) / 100
            a_partial[i] = fuzz.partial_ratio(s1a_list[i],    sxa_list[i]) / 100

    shared_addr   = np.empty(N, np.float32)
    jaccard_addr  = np.empty(N, np.float32)
    digits_overlap = np.empty(N, np.float32)
    exact_addr    = np.empty(N, np.float32)
    for i in range(N):
        w1a = set(w for w in s1a_list[i].split() if len(w) >= 4 and w not in STOP_ADDR_FEAT)
        wxa = set(w for w in sxa_list[i].split() if len(w) >= 4 and w not in STOP_ADDR_FEAT)
        sh = len(w1a & wxa)
        un = len(w1a | wxa)
        shared_addr[i] = sh
        jaccard_addr[i] = sh / un if un else 0.0
        d1 = set(addr_digits(s1a_list[i]))
        dx = set(addr_digits(sxa_list[i]))
        do = 1.0 if (d1 and dx and bool(d1 & dx)) else 0.0
        digits_overlap[i] = do
        exact_addr[i] = 1.0 if (do and sh >= 2) else 0.0

    name_addr_conflict = np.where(
        (has_empty_addr == 0) & (max_name_sim < 0.40) & (exact_addr == 0), 1.0, 0.0
    ).astype(np.float32)
    combined_score = np.where(
        has_empty_addr == 0,
        0.6 * max_name_sim + 0.4 * a_set,
        max_name_sim
    ).astype(np.float32)

    return np.column_stack([
        n_sort, n_set, n_ratio, n_partial,
        s_n_sort, s_n_set, max_name_sim,
        shared_name, jaccard_name, sub_match,
        a_sort, a_set, a_ratio, a_partial,
        has_empty_addr,
        shared_addr, jaccard_addr,
        digits_overlap, exact_addr,
        name_addr_conflict,
        combined_score,
    ]).astype(np.float32)


def precision_gate(cn1, ca1, cnx, cax, score, threshold):
    w1_n = set(w for w in cn1.split() if len(w) >= 3 and w not in LEGAL_STOP_FEAT)
    wx_n = set(w for w in cnx.split() if len(w) >= 3 and w not in LEGAL_STOP_FEAT)
    sq1 = set(squeeze(w) for w in w1_n)
    sqx = set(squeeze(w) for w in wx_n)
    nx_comp = "".join(cnx.split())
    n1_comp = "".join(cn1.split())
    substring_match = any(len(w) >= 4 and w in nx_comp for w in w1_n) or any(
        len(w) >= 4 and w in n1_comp for w in wx_n
    )
    has_name_signal = bool(w1_n & wx_n) or bool(sq1 & sqx) or substring_match

    d1 = set(addr_digits(ca1))
    dx = set(addr_digits(cax))
    digits_match = bool(d1 & dx) if (d1 and dx) else False
    w1_a = set(w for w in ca1.split() if len(w) >= 4 and w not in STOP_ADDR_FEAT)
    wx_a = set(w for w in cax.split() if len(w) >= 4 and w not in STOP_ADDR_FEAT)
    addr_overlap = len(w1_a & wx_a)
    exact_address_match = digits_match and addr_overlap >= 2

    rescue = (
        score >= (threshold - 0.06)
        and score < threshold
        and max(
            fuzz.token_set_ratio(squeeze(cn1), squeeze(cnx)) / 100.0,
            fuzz.token_sort_ratio(cn1, cnx) / 100.0
        ) >= 0.96
        and (exact_address_match or not ca1 or not cax)
    )
    if score < threshold and not rescue:
        return False
    if not has_name_signal and not exact_address_match:
        return False
    if not has_name_signal and exact_address_match:
        return digits_match and addr_overlap >= 3
    if ca1 and cax and w1_a and wx_a and addr_overlap == 0 and not digits_match:
        if fuzz.token_sort_ratio(cn1, cnx) < 85:
            return False
    return True


def load_and_clean(filepath: str, keep_countries=None) -> dict:
    """Returns dict: eid -> (clean_name, clean_addr, country)"""
    log(f"Loading & cleaning {os.path.basename(filepath)}...")
    t0 = time.time()
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""
            if keep_countries is not None and country not in keep_countries:
                continue
            data[eid] = (clean_name(name), clean_addr(addr), country)
    log(f"  -> {len(data):,} records in {time.time()-t0:.1f}s")
    return data


def load_s1_meta(filepath: str):
    log(f"Scanning {os.path.basename(filepath)}...")
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


def run_fast_predict():
    log("=" * 70)
    log("FAST PREDICTION PIPELINE v3 (streaming batches)")
    log("=" * 70)
    t_start = time.time()

    model_path = os.path.join(MODEL_DIR, "lgbm_model.txt")
    th_path = os.path.join(MODEL_DIR, "threshold.pkl")
    model = lgb.Booster(model_file=model_path)
    with open(th_path, "rb") as f:
        threshold = float(pickle.load(f))
    log(f"Loaded model ({model.num_trees()} trees) threshold={threshold:.4f}")

    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    s2_path = os.path.join(TEST_DIR, "test_source2.tsv")
    s3_path = os.path.join(TEST_DIR, "test_source3.tsv")
    matching_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    all_s1_ids, s1_countries = load_s1_meta(s1_path)
    countries = sorted(set(s1_countries.values()))
    log(f"S1={len(all_s1_ids):,} countries={countries}")

    matching = {sid: [] for sid in all_s1_ids}
    cand_tmp_dir = os.path.join(OUTPUT_DIR, "_fast_cand_parts")
    os.makedirs(cand_tmp_dir, exist_ok=True)
    part_files = []

    for country in countries:
        log(f"=== country '{country}' ===")
        t0 = time.time()

        s2 = load_and_clean(s2_path, keep_countries={country})
        s3 = load_and_clean(s3_path, keep_countries={country})
        sx = {}
        sx.update(s2)
        sx.update(s3)
        del s2, s3
        gc.collect()

        inv = build_inverted_index(sx)
        s1_clean = load_and_clean(s1_path, keep_countries={country})
        country_s1 = [sid for sid in all_s1_ids if s1_countries[sid] == country]
        log(f"  {len(country_s1):,} S1 entities in {country}")

        part_path = os.path.join(cand_tmp_dir, f"cand_{len(part_files)}.tsv")
        part_files.append(part_path)
        processed = 0

        with open(part_path, "w", encoding="utf-8") as pf:
            for batch_start in range(0, len(country_s1), BATCH_ENTITIES):
                batch_ids = country_s1[batch_start:batch_start + BATCH_ENTITIES]

                batch_cands = {}  # s1_id -> [m_id, ...]
                for s1_id in batch_ids:
                    cn1, ca1, _ = s1_clean[s1_id]
                    cands = find_candidates_fast(cn1, ca1, country, inv)
                    uniq = []
                    seen = set()
                    for c in cands:
                        if c not in seen and c in sx:
                            seen.add(c)
                            uniq.append(c)
                    batch_cands[s1_id] = uniq
                    pf.write(f"{s1_id}\t{','.join(uniq)}\n")

                s1n_list, s1a_list, sxn_list, sxa_list = [], [], [], []
                pair_meta = []  # (s1_id, m_id)
                for s1_id in batch_ids:
                    cn1, ca1, _ = s1_clean[s1_id]
                    for m_id in batch_cands[s1_id]:
                        cnx, cax, _ = sx[m_id]
                        s1n_list.append(cn1)
                        s1a_list.append(ca1)
                        sxn_list.append(cnx)
                        sxa_list.append(cax)
                        pair_meta.append((s1_id, m_id))

                if pair_meta:
                    feats = extract_features_batch(s1n_list, s1a_list, sxn_list, sxa_list)
                    scores = model.predict(feats)

                    for i, (s1_id, m_id) in enumerate(pair_meta):
                        sc = float(scores[i])
                        cn1, ca1, _ = s1_clean[s1_id]
                        cnx, cax, _ = sx[m_id]
                        if precision_gate(cn1, ca1, cnx, cax, sc, threshold):
                            matching[s1_id].append(m_id)

                processed += len(batch_ids)
                if processed % 100_000 == 0 or processed == len(country_s1):
                    elapsed = time.time() - t0
                    rate = processed / elapsed
                    remaining = (len(country_s1) - processed) / rate / 60 if rate > 0 else 0
                    log(f"  {processed:,}/{len(country_s1):,} {country} "
                        f"({rate:.0f} ent/s, ~{remaining:.1f} min remaining)")

        log(f"  {country} done in {(time.time() - t0) / 60:.1f} min")
        del inv, sx, s1_clean
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
    cand_map_all = {}
    for pp in part_files:
        with open(pp, "r", encoding="utf-8") as pf:
            for line in pf:
                if "\t" not in line:
                    continue
                sid, rest = line.rstrip("\n").split("\t", 1)
                cand_map_all[sid] = rest
    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            f.write(f"{s1_id}\t{cand_map_all.get(s1_id, '')}\n")

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
    log(f"FAST PREDICTION COMPLETE in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    run_fast_predict()
