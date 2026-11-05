
import re
import numpy as np
from rapidfuzz import fuzz
from preprocessing import extract_name_tokens, extract_address_tokens, squeeze_text, NAME_STOP_WORDS


_RE_SQUEEZE = re.compile(r'(\w)\1+')
_RE_DIGITS = re.compile(r'\b\d+\b')


def compute_pair_features(
    s1_name_norm: str, s1_addr_norm: str,
    sx_name_norm: str, sx_addr_norm: str,
) -> list:
    
    n_sort = fuzz.token_sort_ratio(s1_name_norm, sx_name_norm) / 100.0
    n_set = fuzz.token_set_ratio(s1_name_norm, sx_name_norm) / 100.0
    n_ratio = fuzz.ratio(s1_name_norm, sx_name_norm) / 100.0
    n_partial = fuzz.partial_ratio(s1_name_norm, sx_name_norm) / 100.0

    s_n1 = _RE_SQUEEZE.sub(r'\1', s1_name_norm)
    s_nx = _RE_SQUEEZE.sub(r'\1', sx_name_norm)
    s_n_sort = fuzz.token_sort_ratio(s_n1, s_nx) / 100.0
    s_n_set = fuzz.token_set_ratio(s_n1, s_nx) / 100.0

    max_name_sim = max(n_sort, n_set, s_n_sort, s_n_set)

    w1_n = set(w for w in s1_name_norm.split() if len(w) >= 3 and w not in NAME_STOP_WORDS)
    wx_n = set(w for w in sx_name_norm.split() if len(w) >= 3 and w not in NAME_STOP_WORDS)
    shared_name = len(w1_n & wx_n)
    union_n = len(w1_n | wx_n)
    jaccard_name = shared_name / union_n if union_n > 0 else 0.0

    nx_comp = ''.join(sx_name_norm.split())
    n1_comp = ''.join(s1_name_norm.split())
    sub_match = 1.0 if (
        any(len(w) >= 4 and w in nx_comp for w in w1_n) or
        any(len(w) >= 4 and w in n1_comp for w in wx_n)
    ) else 0.0

    has_empty_addr = 1.0 if (not s1_addr_norm or not sx_addr_norm) else 0.0
    if not has_empty_addr:
        a_sort = fuzz.token_sort_ratio(s1_addr_norm, sx_addr_norm) / 100.0
        a_set = fuzz.token_set_ratio(s1_addr_norm, sx_addr_norm) / 100.0
        a_ratio = fuzz.ratio(s1_addr_norm, sx_addr_norm) / 100.0
        a_partial = fuzz.partial_ratio(s1_addr_norm, sx_addr_norm) / 100.0
    else:
        a_sort = a_set = a_ratio = a_partial = 0.0

    STOP_ADDR = {'road', 'street', 'avenue', 'lane', 'drive', 'near', 'floor', 'flat',
                 'unit', 'opp', 'opposite', 'behind', 'bldg', 'building', 'null'}
    w1_a = set(w for w in s1_addr_norm.split() if len(w) >= 4 and w not in STOP_ADDR)
    wx_a = set(w for w in sx_addr_norm.split() if len(w) >= 4 and w not in STOP_ADDR)
    shared_addr = len(w1_a & wx_a)
    union_a = len(w1_a | wx_a)
    jaccard_addr = shared_addr / union_a if union_a > 0 else 0.0

    d1 = set(str(int(d)) for d in _RE_DIGITS.findall(s1_addr_norm) if len(d) <= 6 and d != '0')
    dx = set(str(int(d)) for d in _RE_DIGITS.findall(sx_addr_norm) if len(d) <= 6 and d != '0')
    digits_overlap = 1.0 if (d1 and dx and bool(d1 & dx)) else 0.0

    exact_addr = 1.0 if (digits_overlap and shared_addr >= 2) else 0.0

    combined_score = 0.6 * max_name_sim + 0.4 * a_set if not has_empty_addr else max_name_sim

    return [
        n_sort, n_set, n_ratio, n_partial,
        s_n_sort, s_n_set, max_name_sim,
        shared_name, jaccard_name, sub_match,
        a_sort, a_set, a_ratio, a_partial,
        has_empty_addr,
        shared_addr, jaccard_addr,
        digits_overlap, exact_addr,
        combined_score,
        float(len(w1_n)),  
    ]


FEATURE_NAMES = [
    'n_sort', 'n_set', 'n_ratio', 'n_partial',
    's_n_sort', 's_n_set', 'max_name_sim',
    'shared_name', 'jaccard_name', 'sub_match',
    'a_sort', 'a_set', 'a_ratio', 'a_partial',
    'has_empty_addr',
    'shared_addr', 'jaccard_addr',
    'digits_overlap', 'exact_addr',
    'combined_score',
    's1_name_tokens',
]

NUM_FEATURES = len(FEATURE_NAMES)
