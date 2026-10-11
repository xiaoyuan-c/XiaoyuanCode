from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]+|[\u3400-\u4dbf\u4e00-\u9fff]+")
_CJK_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


# 将中英文混合文本转换为适合 BM25 的词元
def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        value = match.group(0)
        if not _CJK_RE.fullmatch(value):
            tokens.append(value)
            continue
        tokens.extend(value)
        tokens.extend(value[index:index + 2] for index in range(len(value) - 1))
    return tokens

