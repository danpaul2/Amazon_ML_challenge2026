

import sys
import os
import re
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from unidecode import unidecode
from rapidfuzz import fuzz
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8')

LEGAL_STOP = {'inc', 'llc', 'corp', 'corporation', 'pvt', 'ltd', 'private', 'limited', 'co', 'company',
              'services', 'service', 'enterprise', 'enterprises', 'group', 'holdings', 'holding',
              'sarl', 'sasu', 'sci', 'sas', 'sa', 'eurl', 'the', 'and', 'of', 'for', 'in', 'at',
              'mr', 'mrs', 'ms', 'dr', 'smt', 'shri', 'shree'}

STOP_ADDR = {'road', 'street', 'avenue', 'lane', 'drive', 'near', 'floor', 'flat', 'unit',
             'opp', 'opposite', 'behind', 'bldg', 'building', 'null'}


def get_keys(name, addr, country):
    keys = set()
    c = country.lower().strip()
    
    n_clean = unidecode(name).lower()
    n_clean = re.sub(r'\.(com|org|net|in|co|io|fr|edu|gov)\b', '', n_clean)
    n_words = [w for w in re.findall(r'[a-z0-9]+', n_clean) if len(w) >= 3 and w not in LEGAL_STOP]
    n_squeezed = [re.sub(r'(\w)\1+', r'\1', w) for w in n_words]
    
    if n_words:
        compressed = ''.join(n_words[:3])
        if len(compressed) >= 5:
            keys.add(f'{c}_nc_{compressed[:12]}')
            
    for w in n_words:
        if len(w) >= 4:
            keys.add(f'{c}_nw_{w}')
    for w in n_squeezed:
        if len(w) >= 4:
            keys.add(f'{c}_nw_{w}')
            
    if len(n_words) >= 2:
        keys.add(f'{c}_n2_{n_words[0]}_{n_words[1]}')
        keys.add(f'{c}_n2_{n_squeezed[0]}_{n_squeezed[1]}')
        
    a_clean = unidecode(addr).lower()
    raw_digits = re.findall(r'\b\d+\b', a_clean)
    digits = [str(int(d)) for d in raw_digits if len(d) <= 6 and int(d) > 0]
    a_words = [w for w in re.findall(r'[a-z]+', a_clean) if len(w) >= 4 and w not in STOP_ADDR]
    
    for d in digits[:3]:
        for w in a_words[:6]:
            keys.add(f'{c}_num_{d}_{w}')
            
    for i in range(min(len(a_words)-1, 5)):
        keys.add(f'{c}_a2_{a_words[i]}_{a_words[i+1]}')
        
    return keys


def extract_features(s1_name, s1_addr, sx_name, sx_addr):
    n1 = unidecode(s1_name).lower()
    nx = unidecode(sx_name).lower()
    a1 = unidecode(s1_addr).lower()
    ax = unidecode(sx_addr).lower()
    
    n_sort = fuzz.token_sort_ratio(n1, nx) / 100.0
    n_set = fuzz.token_set_ratio(n1, nx) / 100.0
    n_ratio = fuzz.ratio(n1, nx) / 100.0
    n_partial = fuzz.partial_ratio(n1, nx) / 100.0
    
    has_empty_addr = 1.0 if (not a1 or not ax) else 0.0
    if not has_empty_addr:
        a_sort = fuzz.token_sort_ratio(a1, ax) / 100.0
        a_set = fuzz.token_set_ratio(a1, ax) / 100.0
        a_ratio = fuzz.ratio(a1, ax) / 100.0
        a_partial = fuzz.partial_ratio(a1, ax) / 100.0
    else:
        a_sort = 0.0
        a_set = 0.0
        a_ratio = 0.0
        a_partial = 0.0
        
    
    w1_n = set(re.findall(r'[a-z0-9]+', n1)) - LEGAL_STOP
    wx_n = set(re.findall(r'[a-z0-9]+', nx)) - LEGAL_STOP
    shared_name = len(w1_n & wx_n)
    jaccard_name = shared_name / len(w1_n | wx_n) if (w1_n or wx_n) else 0.0
    
    w1_a = set(re.findall(r'[a-z0-9]+', a1)) - STOP_ADDR
    wx_a = set(re.findall(r'[a-z0-9]+', ax)) - STOP_ADDR
    shared_addr = len(w1_a & wx_a)
    jaccard_addr = shared_addr / len(w1_a | wx_a) if (w1_a or wx_a) else 0.0
    
    d1 = set(str(int(d)) for d in re.findall(r'\b\d+\b', a1) if len(d) <= 6 and int(d) > 0)
    dx = set(str(int(d)) for d in re.findall(r'\b\d+\b', ax) if len(d) <= 6 and int(d) > 0)
    digits_overlap = 1.0 if (d1 and dx and bool(d1 & dx)) else 0.0
    
    combined_score = 0.6 * n_set + 0.4 * a_set if not has_empty_addr else n_set
    
    return [
        n_sort, n_set, n_ratio, n_partial,
        a_sort, a_set, a_ratio, a_partial,
        has_empty_addr,
        shared_name, jaccard_name,
        shared_addr, jaccard_addr,
        digits_overlap,
        combined_score
    ]

FEATURE_NAMES = [
    'n_sort', 'n_set', 'n_ratio', 'n_partial',
    'a_sort', 'a_set', 'a_ratio', 'a_partial',
    'has_empty_addr',
    'shared_name', 'jaccard_name',
    'shared_addr', 'jaccard_addr',
    'digits_overlap',
    'combined_score'
]

def main():
    print("="*60)
    print("RUNNING BENCHMARK & MODEL EVALUATION")
    print("="*60)
    t0 = time.time()
    
    print("Loading 5,000 S1 entities from Ground Truth...")
    gt = {}
    with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        f.readline()
        for _ in range(5000):
            line = f.readline()
            if not line: break
            p = line.strip().split('\t')
            gt[p[0]] = set(p[1].split(',')) if len(p) > 1 and p[1] else set()
            
    target_s1 = set(gt.keys())
    all_target_sx = set()
    for matches in gt.values():
        all_target_sx.update(matches)
        
    s1_data = {}
    with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if p[0] in target_s1:
                s1_data[p[0]] = p[1:]
                if len(s1_data) == len(target_s1): break
                
    print("Loading SX records into inverted index...")
    sx_data = {}
    sx_index = defaultdict(list)
    
    for src in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
        with open(src, 'r', encoding='utf-8') as f:
            f.readline()
            for i, line in enumerate(f):
                p = line.strip().split('\t')
                if p[0] in all_target_sx or i < 50000:
                    sx_data[p[0]] = p[1:]
                    for k in get_keys(*p[1:]):
                        sx_index[k].append(p[0])
                        
    print(f"Total SX records loaded: {len(sx_data):,}, Index keys: {len(sx_index):,}")
    
    print("Generating candidates for 5,000 S1 entities...")
    candidate_pairs = defaultdict(set)
    for s1_id in target_s1:
        if s1_id not in s1_data: continue
        keys = get_keys(*s1_data[s1_id])
        for k in keys:
            if k in sx_index and len(sx_index[k]) <= 150:
                candidate_pairs[s1_id].update(sx_index[k])
                
    true_pairs = 0
    caught_pairs = 0
    for s1_id, matches in gt.items():
        if s1_id not in s1_data: continue
        true_pairs += len(matches)
        caught_pairs += len(matches & candidate_pairs[s1_id])
        
    print(f"Blocking recall on 5,000 S1: {caught_pairs}/{true_pairs} = {caught_pairs/true_pairs*100:.2f}%")
    
    s1_ids = list(s1_data.keys())
    np.random.seed(42)
    np.random.shuffle(s1_ids)
    split_idx = int(0.8 * len(s1_ids))
    train_s1_ids = set(s1_ids[:split_idx])
    val_s1_ids = set(s1_ids[split_idx:])
    
    def build_pair_dataset(target_ids, max_neg_ratio=3):
        X, y, pair_ids = [], [], []
        for s1_id in target_ids:
            true_m = gt[s1_id]
            cands = candidate_pairs[s1_id]
            s1_info = s1_data[s1_id]
            
            
            for m in true_m:
                if m in sx_data:
                    feat = extract_features(s1_info[0], s1_info[1], sx_data[m][0], sx_data[m][1])
                    X.append(feat)
                    y.append(1)
                    pair_ids.append((s1_id, m))
                    
            neg_cands = list(cands - true_m)
            np.random.shuffle(neg_cands)
            n_neg = min(len(neg_cands), max(1, max_neg_ratio * len(true_m)))
            for m in neg_cands[:n_neg]:
                if m in sx_data:
                    feat = extract_features(s1_info[0], s1_info[1], sx_data[m][0], sx_data[m][1])
                    X.append(feat)
                    y.append(0)
                    pair_ids.append((s1_id, m))
                    
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), pair_ids
        
    print("Building training feature vectors...")
    X_train, y_train, _ = build_pair_dataset(train_s1_ids, max_neg_ratio=3)
    print(f"Train dataset: {len(X_train):,} pairs ({y_train.sum():,} pos, {(y_train==0).sum():,} neg)")
    
    print("Training LightGBM classifier...")
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'verbose': -1,
        'seed': 42,
    }
    model = lgb.train(params, train_data, num_boost_round=200)
    print("Model trained successfully.")
    
    print("Evaluating on 1,000 validation S1 entities...")
    best_threshold = 0.5
    best_f05 = 0.0
    
    val_pairs = []
    val_features = []
    for s1_id in val_s1_ids:
        s1_info = s1_data[s1_id]
        for m in candidate_pairs[s1_id]:
            if m in sx_data:
                feat = extract_features(s1_info[0], s1_info[1], sx_data[m][0], sx_data[m][1])
                val_features.append(feat)
                val_pairs.append((s1_id, m))
                
    if val_features:
        scores = model.predict(np.array(val_features, dtype=np.float32))
        s1_scores = defaultdict(list)
        for (s1_id, m), score in zip(val_pairs, scores):
            s1_scores[s1_id].append((m, score))
            
        print("Tuning threshold for Macro F_0.5...")
        for th in np.arange(0.85, 0.99, 0.01):
            f05_list = []
            for s1_id in val_s1_ids:
                true_m = gt[s1_id]
                pred_m = set(m for m, sc in s1_scores.get(s1_id, []) if sc >= th)
                
                if not true_m and not pred_m:
                    f05_list.append(1.0)
                elif not pred_m:
                    f05_list.append(0.0)
                elif not true_m:
                    f05_list.append(0.0)
                else:
                    tp = len(pred_m & true_m)
                    prec = tp / len(pred_m)
                    rec = tp / len(true_m)
                    if prec + rec > 0:
                        f = (1.25 * prec * rec) / (0.25 * prec + rec)
                    else:
                        f = 0.0
                    f05_list.append(f)
                        
            macro_f05 = np.mean(f05_list)
            print(f"  Threshold {th:.2f} -> Macro F_0.5 = {macro_f05:.4f}")
            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_threshold = th
                
        print("="*60)
        print(f"BEST VALIDATION MACRO F_0.5: {best_f05:.4f} (at threshold {best_threshold:.2f})")
        print(f"Target is > 0.984")
        print(f"Total benchmark elapsed time: {time.time()-t0:.1f}s")
        print("="*60)

if __name__ == '__main__':
    main()
