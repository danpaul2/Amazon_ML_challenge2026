
import re
import unicodedata
from unidecode import unidecode


_RE_URL = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_RE_DOMAIN_SUFFIX = re.compile(r'\.(com|in|co|org|net|io|biz|info|us|fr|uk|de)\b', re.IGNORECASE)
_RE_AMP = re.compile(r'[&]+')
_RE_HASH = re.compile(r'[#]+')
_RE_NON_ALNUM_SPACE = re.compile(r'[^\w\s]')
_RE_NON_ALNUM_SPACE_COMMA = re.compile(r'[^\w\s,]')
_RE_MULTI_SPACE = re.compile(r'\s+')
_RE_SINGLE_CHAR = re.compile(r'\b[a-z]\b')
_RE_LEADING_ZEROS = re.compile(r'\b0+(\d+)')
_RE_SQUEEZE = re.compile(r'(\w)\1+')
_RE_NUMBERS = re.compile(r'\b(\d+)\b')

_LEGAL_SUFFIX_PATTERNS = [
    (re.compile(r'\bpvt\.?\b', re.I), 'private'),
    (re.compile(r'\bltd\.?\b', re.I), 'limited'),
    (re.compile(r'\bcorp\.?\b', re.I), 'corporation'),
    (re.compile(r'\binc\.?\b', re.I), 'incorporated'),
    (re.compile(r'\bllc\.?\b', re.I), 'llc'),
    (re.compile(r'\bllp\.?\b', re.I), 'llp'),
    (re.compile(r'\bco\.?\b', re.I), 'company'),
    (re.compile(r'\bsvc[s]?\.?\b', re.I), 'services'),
    (re.compile(r'\bmfg\.?\b', re.I), 'manufacturing'),
    (re.compile(r'\bintl\.?\b', re.I), 'international'),
    (re.compile(r'\bnatl\.?\b', re.I), 'national'),
    (re.compile(r'\bassoc\.?\b', re.I), 'associates'),
    (re.compile(r'\bgrp\.?\b', re.I), 'group'),
    (re.compile(r'\benterp\.?\b', re.I), 'enterprises'),
    (re.compile(r'\bhldgs?\.?\b', re.I), 'holdings'),
    (re.compile(r'\bmgmt\.?\b', re.I), 'management'),
    (re.compile(r'\bconsult\.?\b', re.I), 'consulting'),
    (re.compile(r'\bconstr\.?\b', re.I), 'construction'),
    (re.compile(r'\btech\.?\b', re.I), 'technology'),
    (re.compile(r'\bdist\.?\b', re.I), 'distribution'),
    (re.compile(r'\bdev\.?\b', re.I), 'development'),
    (re.compile(r'\bsoln?s?\.?\b', re.I), 'solutions'),
    (re.compile(r'\b(?:p\.?\s*c\.?|pc)\b', re.I), 'pc'),
    (re.compile(r'\bdba\b', re.I), ''),
    (re.compile(r'\bsarl\.?\b', re.I), 'sarl'),
    (re.compile(r'\bsasu\.?\b', re.I), 'sasu'),
    (re.compile(r'\bsci\.?\b', re.I), 'sci'),
    (re.compile(r'\bs\.?a\.?s\.?\b', re.I), 'sas'),
    (re.compile(r'\bs\.?a\.?\b', re.I), 'sa'),
    (re.compile(r'\beurl\.?\b', re.I), 'eurl'),
    (re.compile(r'प्राइवेट'), 'private'),
    (re.compile(r'प्रा\.?'), 'private'),
    (re.compile(r'लिमिटेड'), 'limited'),
    (re.compile(r'लि\.?'), 'limited'),
    (re.compile(r'कंपनी'), 'company'),
    (re.compile(r'कॉर्पोरेशन'), 'corporation'),
    (re.compile(r'एलएलपी'), 'llp'),
    (re.compile(r'एलएलसी'), 'llc'),
]

_ADDR_ABBREV_PATTERNS = [
    (re.compile(r'\brd\.?\b', re.I), 'road'),
    (re.compile(r'\bst\.?\b', re.I), 'street'),
    (re.compile(r'\bave\.?\b', re.I), 'avenue'),
    (re.compile(r'\bblvd\.?\b', re.I), 'boulevard'),
    (re.compile(r'\bdr\.?\b', re.I), 'drive'),
    (re.compile(r'\bln\.?\b', re.I), 'lane'),
    (re.compile(r'\bct\.?\b', re.I), 'court'),
    (re.compile(r'\bpl\.?\b', re.I), 'place'),
    (re.compile(r'\bpkwy\.?\b', re.I), 'parkway'),
    (re.compile(r'\bhwy\.?\b', re.I), 'highway'),
    (re.compile(r'\bapt\.?\b', re.I), 'apartment'),
    (re.compile(r'\bste\.?\b', re.I), 'suite'),
    (re.compile(r'\bfl\.?\b', re.I), 'floor'),
    (re.compile(r'\bbldg\.?\b', re.I), 'building'),
    (re.compile(r'\bdist\.?\b', re.I), 'district'),
    (re.compile(r'\bnr\.?\b', re.I), 'near'),
    (re.compile(r'\bopp\.?\b', re.I), 'opposite'),
    (re.compile(r'\bh\.?\s*no\.?\b', re.I), ''),
    (re.compile(r'\bp\.?\s*o\.?\s*box\b', re.I), 'po box'),
]

NAME_STOP_WORDS = frozenset({
    'private', 'limited', 'pvt', 'ltd', 'llc', 'llp', 'inc',
    'incorporated', 'corporation', 'corp', 'company', 'co',
    'the', 'of', 'and', 'for', 'in', 'at', 'to', 'by',
    'sarl', 'sasu', 'sci', 'sas', 'sa', 'eurl',
    'dba', 'pc', 'pllc', 'enterprises', 'services', 'solutions',
    'group', 'holdings', 'international', 'associates', 'consulting',
    'management', 'technology', 'manufacturing', 'construction',
    'development', 'distribution', 'national',
})

ADDR_STOP_WORDS = frozenset({
    'near', 'opposite', 'behind', 'next', 'to', 'at', 'in',
    'the', 'of', 'and', 'for', 'by', 'no', 'null',
    'floor', 'flat', 'room', 'unit', 'apartment', 'suite',
    'building', 'tower', 'block', 'plot', 'sector', 'phase',
})


def squeeze_word(word: str) -> str:
    return _RE_SQUEEZE.sub(r'\1', word)


def squeeze_text(text: str) -> str:
    """Apply squeeze to every word in text."""
    return ' '.join(squeeze_word(w) for w in text.split())


def normalize_business_name(name: str) -> str:
   
    if not name or not isinstance(name, str):
        return ""

    name = unicodedata.normalize('NFKC', name)

    name = name.lower().strip()

    for pattern, replacement in _LEGAL_SUFFIX_PATTERNS:
        name = pattern.sub(replacement, name)

    name = unidecode(name).lower()

    name = _RE_URL.sub('', name)
    name = _RE_DOMAIN_SUFFIX.sub('', name)

    name = _RE_AMP.sub(' and ', name)
    name = _RE_NON_ALNUM_SPACE.sub(' ', name)

    name = squeeze_text(name)

    name = _RE_MULTI_SPACE.sub(' ', name).strip()

    name = _RE_SINGLE_CHAR.sub('', name)
    name = _RE_MULTI_SPACE.sub(' ', name).strip()

    return name


def normalize_address(address: str) -> str:
   
    if not address or not isinstance(address, str):
        return ""

    address = unicodedata.normalize('NFKC', address)
    address = address.lower().strip()

    address = unidecode(address).lower()

    for pattern, replacement in _ADDR_ABBREV_PATTERNS:
        address = pattern.sub(replacement, address)

    address = _RE_AMP.sub(' and ', address)
    address = _RE_HASH.sub('', address)
    address = _RE_NON_ALNUM_SPACE_COMMA.sub(' ', address)

    address = _RE_LEADING_ZEROS.sub(r'\1', address)

    address = _RE_MULTI_SPACE.sub(' ', address).strip()

    return address


def extract_name_tokens(normalized_name: str) -> set:
    if not normalized_name:
        return set()
    tokens = set(normalized_name.split())
    return tokens - NAME_STOP_WORDS


def extract_address_tokens(normalized_address: str) -> set:
    if not normalized_address:
        return set()
    tokens = set(normalized_address.split())
    return tokens - ADDR_STOP_WORDS


def extract_numbers(text: str) -> set:
    if not text:
        return set()
    return set(_RE_NUMBERS.findall(text))


def get_compressed_name(normalized_name: str) -> str:
 
    tokens = extract_name_tokens(normalized_name)
    if not tokens:
        return ""
    return ''.join(sorted(tokens))
