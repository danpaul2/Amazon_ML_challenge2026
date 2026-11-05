

import sys, re, time
from collections import defaultdict
import numpy as np
from unidecode import unidecode
from rapidfuzz import fuzz
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8')

LEET_MAP = str.maketrans('01345$', 'oieasS')

LEGAL_STOP = {'inc', 'llc', 'corp', 'corporation', 'pvt', 'ltd', 'private', 'limited', 'co', 'company',
              'services', 'service', 'enterprise', 'enterprises', 'group', 'holdings', 'holding',
              'sarl', 'sasu', 'sci', 'sas', 'sa', 'eurl', 'the', 'and', 'of', 'for', 'in', 'at',
              'mr', 'mrs', 'ms', 'dr', 'smt', 'shri', 'shree'}

STOP_ADDR = {'road', 'street', 'avenue', 'lane', 'drive', 'near', 'floor', 'flat', 'unit',
             'opp', 'opposite', 'behind', 'bldg', 'building', 'null', 'c/o'}

def clean_name(name):
    if not name: return ""
    n = unidecode(name).lower()
    n = n.translate(LEET_MAP)
    n = re.sub(r'\.(com|org|net|in|co|io|fr|edu|gov)\b', '', n)
    n = re.sub(r'[^\w\s]', ' ', n)
    return re.sub(r'\s+', ' ', n).strip()

def clean_addr(addr):
    if not addr: return ""
    a = unidecode(addr).lower()
    a = re.sub(r'[^\w\s]', ' ', a)
    return re.sub(r'\s+', ' ', a).strip()

def get_keys(name, addr, country):
    keys = set()
    c = country.lower().strip()
    
    n = clean_name(name)
    n_words = [w for w in n.split() if len(w) >= 3 and w not in LEGAL_STOP]
    n_squeezed = [re.sub(r'(\w)\1+', r'\1', w) for w in n_words]
    
    if n_words:
        comp = ''.join(n_words[:3])
        if len(comp) >= 5: keys.add(f'{c}_nc_{comp[:12]}')
    for w in n_words:
        if len(w) >= 4: keys.add(f'{c}_nw_{w}')
    for w in n_squeezed:
        if len(w) >= 4: keys.add(f'{c}_nw_{w}')
    if len(n_words) >= 2:
        keys.add(f'{c}_n2_{n_words[0]}_{n_words[1]}')
        keys.add(f'{c}_n2_{n_squeezed[0]}_{n_squeezed[1]}')
        
    a = clean_addr(addr)
    raw_digits = re.findall(r'\b\d+\b', a)
    digits = [str(int(d)) for d in raw_digits if len(d) <= 6 and int(d) > 0]
    a_words = [w for w in a.split() if len(w) >= 4 and w not in STOP_ADDR]
    
    for d in digits[:3]:
        for w in a_words[:6]:
            keys.add(f'{c}_num_{d}_{w}')
    for i in range(min(len(a_words)-1, 5)):
        keys.add(f'{c}_a2_{a_words[i]}_{a_words[i+1]}')
        
    return keys

def is_match_pair(s1_name, s1_addr, sx_name, sx_addr, lgb_score):
   
    n1 = clean_name(s1_name)
    nx = clean_name(sx_name)
    a1 = clean_addr(s1_addr)
    ax = clean_addr(sx_addr)
    
   
    w1_n = set(w for w in n1.split() if len(w) >= 3 and w not in LEGAL_STOP)
    wx_n = set(w for w in nx.split() if len(w) >= 3 and w not in LEGAL_STOP)
    
    
    sq1_n = set(re.sub(r'(\w)\1+', r'\1', w) for w in w1_n)
    sqx_n = set(re.sub(r'(\w)\1+', r'\1', w) for w in wx_n)
    
    direct_name_match = bool(w1_n & wx_n) or bool(sq1_n & sqx_n)
    
    nx_comp = ''.join(nx.split())
    n1_comp = ''.join(n1.split())
    substring_match = any((len(w) >= 4 and w in nx_comp) for w in w1_n) or any((len(w) >= 4 and w in n1_comp) for w in wx_n)
    
    has_name_signal = direct_name_match or substring_match
    
    d1 = set(str(int(d)) for d in re.findall(r'\b\d+\b', a1) if len(d) <= 6 and int(d) > 0)
    dx = set(str(int(d)) for d in re.findall(r'\b\d+\b', ax) if len(d) <= 6 and int(d) > 0)
    digits_match = bool(d1 & dx) if (d1 and dx) else False
    
    w1_a = set(w for w in a1.split() if len(w) >= 4 and w not in STOP_ADDR)
    wx_a = set(w for w in ax.split() if len(w) >= 4 and w not in STOP_ADDR)
    addr_words_overlap = len(w1_a & wx_a)
    
    exact_address_match = digits_match and (addr_words_overlap >= 2)
    
    if not has_name_signal and not exact_address_match:
        return False
        
    if not has_name_signal and exact_address_match:
        return digits_match and (addr_words_overlap >= 3)
        
    if a1 and ax and w1_a and wx_a:
        if addr_words_overlap == 0 and not digits_match:
            # Different locations entirely -> require very high name match
            name_sim = fuzz.token_sort_ratio(n1, nx)
            if name_sim < 85:
                return False
                
    return lgb_score >= 0.70

def extract_features(s1_name, s1_addr, sx_name, sx_addr):
    n1 = clean_name(s1_name)
    nx = clean_name(sx_name)
    a1 = clean_addr(s1_addr)
    ax = clean_addr(sx_addr)
    
    n_sort = fuzz.token_sort_ratio(n1, nx) / 100.0
    n_set = fuzz.token_set_ratio(n1, nx) / 100.0
    n_ratio = fuzz.ratio(n1, nx) / 100.0
    n_partial = fuzz.partial_ratio(n1, nx) / 100.0
    
    s_n1 = re.sub(r'(\w)\1+', r'\1', n1)
    s_nx = re.sub(r'(\w)\1+', r'\1', nx)
    s_n_sort = fuzz.token_sort_ratio(s_n1, s_nx) / 100.0
    s_n_set = fuzz.token_set_ratio(s_n1, s_nx) / 100.0
    
    has_empty_addr = 1.0 if (not a1 or not ax) else 0.0
    if not has_empty_addr:
        a_sort = fuzz.token_sort_ratio(a1, ax) / 100.0
        a_set = fuzz.token_set_ratio(a1, ax) / 100.0
        a_ratio = fuzz.ratio(a1, ax) / 100.0
        a_partial = fuzz.partial_ratio(a1, ax) / 100.0
    else:
        a_sort, a_set, a_ratio, a_partial = 0.0, 0.0, 0.0, 0.0
        
    w1_n = set(w for w in n1.split() if len(w) >= 3 and w not in LEGAL_STOP)
    wx_n = set(w for w in nx.split() if len(w) >= 3 and w not in LEGAL_STOP)
    shared_name = len(w1_n & wx_n)
    jaccard_name = shared_name / len(w1_n | wx_n) if (w1_n or wx_n) else 0.0
    
    w1_a = set(w for w in a1.split() if len(w) >= 4 and w not in STOP_ADDR)
    wx_a = set(w for w in ax.split() if len(w) >= 4 and w not in STOP_ADDR)
    shared_addr = len(w1_a & wx_a)
    jaccard_addr = shared_addr / len(w1_a | wx_a) if (w1_a or wx_a) else 0.0
    
    d1 = set(str(int(d)) for d in re.findall(r'\b\d+\b', a1) if len(d) <= 6 and int(d) > 0)
    dx = set(str(int(d)) for d in re.findall(r'\b\d+\b', ax) if len(d) <= 6 and int(d) > 0)
    digits_overlap = 1.0 if (d1 and dx and bool(d1 & dx)) else 0.0
    
    max_name_sim = max(n_sort, n_set, s_n_sort, s_n_set)
    combined_score = 0.6 * max_name_sim + 0.4 * a_set if not has_empty_addr else max_name_sim
    
    return [
        n_sort, n_set, n_ratio, n_partial,
        s_n_sort, s_n_set, max_name_sim,
        a_sort, a_set, a_ratio, a_partial,
        has_empty_addr,
        shared_name, jaccard_name,
        shared_addr, jaccard_addr,
        digits_overlap,
        combined_score
    ]

FEATURE_NAMES = [
    'n_sort', 'n_set', 'n_ratio', 'n_partial',
    's_n_sort', 's_n_set', 'max_name_sim',
    'a_sort', 'a_set', 'a_ratio', 'a_partial',
    'has_empty_addr',
    'shared_name', 'jaccard_name',
    'shared_addr', 'jaccard_addr',
    'digits_overlap',
    'combined_score'
]

def main():
    print("Loading 10,000 S1 records and Ground Truth...")
    gt = {}
    with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        f.readline()
        for _ in range(10000):
            line = f.readline()
            if not line: break
            p = line.strip().split('\t')
            gt[p[0]] = set(p[1].split(',')) if len(p) > 1 and p[1] else set()
            
    target_s1 = set(gt.keys())
    all_target_sx = set()
    for matches in gt.values(): all_target_sx.update(matches)
    
    s1_data = {}
    with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if p[0] in target_s1:
                s1_data[p[0]] = p[1:]
                if len(s1_data) == len(target_s1): break
                
    sx_data = {}
    sx_index = defaultdict(list)
    for src in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
        with open(src, 'r', encoding='utf-8') as f:
            f.readline()
            for i, line in enumerate(f):
                p = line.strip().split('\t')
                if p[0] in all_target_sx or i < 100000:
                    sx_data[p[0]] = p[1:]
                    for k in get_keys(*p[1:]): sx_index[k].append(p[0])
                    
    s1_ids = list(s1_data.keys())
    np.random.seed(42)
    np.random.shuffle(s1_ids)
    train_s1_ids = set(s1_ids[:8000])
    val_s1_ids = set(s1_ids[8000:])
    
    cand_pairs = defaultdict(set)
    for s1_id in s1_ids:
        keys = get_keys(*s1_data[s1_id])
        for k in keys:
            if k in sx_index and len(sx_index[k]) <= 150:
                cand_pairs[s1_id].update(sx_index[k])
                
    print("Training model...")
    X_tr, y_tr = [], []
    for s1_id in train_s1_ids:
        tm = gt[s1_id]
        cands = cand_pairs[s1_id]
        for m in tm:
            if m in sx_data:
                X_tr.append(extract_features(s1_data[s1_id][0], s1_data[s1_id][1], sx_data[m][0], sx_data[m][1]))
                y_tr.append(1)
        neg = list(cands - tm)
        np.random.shuffle(neg)
        for m in neg[:4*len(tm)]:
            if m in sx_data:
                X_tr.append(extract_features(s1_data[s1_id][0], s1_data[s1_id][1], sx_data[m][0], sx_data[m][1]))
                y_tr.append(0)
                
    model = lgb.train({'objective': 'binary', 'num_leaves': 127, 'learning_rate': 0.03, 'verbose': -1, 'seed': 42},
                      lgb.Dataset(np.array(X_tr), label=np.array(y_tr), feature_name=FEATURE_NAMES), num_boost_round=300)
                      
    print("Evaluating precision-gated matching on 2,000 validation entities...")
    val_list = []
    for s1_id in val_s1_ids:
        for m in cand_pairs[s1_id]:
            if m in sx_data:
                feat = extract_features(s1_data[s1_id][0], s1_data[s1_id][1], sx_data[m][0], sx_data[m][1])
                val_list.append((s1_id, m, feat))
                
    scores = model.predict(np.array([x[2] for x in val_list]))
    
    pred_matches = defaultdict(set)
    for (s1_id, m, _), score in zip(val_list, scores):
        s1_name, s1_addr, _ = s1_data[s1_id]
        sx_name, sx_addr, _ = sx_data[m]
        if is_match_pair(s1_name, s1_addr, sx_name, sx_addr, score):
            pred_matches[s1_id].add(m)
            
    f05_scores = []
    tp_tot, fp_tot, fn_tot = 0, 0, 0
    for s1_id in val_s1_ids:
        true_m = gt[s1_id]
        pm = pred_matches[s1_id]
        
        tp = len(pm & true_m)
        fp = len(pm - true_m)
        fn = len(true_m - pm)
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn
        
        if not true_m and not pm:
            f05_scores.append(1.0)
        elif not pm:
            f05_scores.append(0.0)
        elif not true_m:
            f05_scores.append(0.0)
        else:
            prec = tp / len(pm)
            rec = tp / len(true_m)
            if prec + rec > 0:
                f = (1.25 * prec * rec) / (0.25 * prec + rec)
            else:
                f = 0.0
            f05_scores.append(f)
            
    macro_f05 = np.mean(f05_scores)
    print("="*60)
    print(f"PRECISION-GATED RESULTS on 2,000 Validation Entities:")
    print(f"  Macro F_0.5 Score : {macro_f05:.6f}")
    print(f"  Total True Matches: {tp_tot + fn_tot:,}")
    print(f"  True Positives    : {tp_tot:,}")
    print(f"  False Positives   : {fp_tot:,}")
    print(f"  False Negatives   : {fn_tot:,}")
    print(f"  Micro Precision   : {tp_tot / (tp_tot + fp_tot):.4f}")
    print(f"  Micro Recall      : {tp_tot / (tp_tot + fn_tot):.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
