"""
文本标准化工具
处理谐音、拼音缩写、特殊符号插入、拆字等规避手段
"""

from __future__ import annotations
import re
import unicodedata


class TextNormalizer:
    """文本标准化处理器"""

    # 常见谐音映射 (可在 data/rules/evasion.json 中扩展)
    HOMOPHONE_MAP: dict[str, str] = {
        # 这里放置常见的谐音/拼音替换映射
        # 例如: "sb" -> "傻逼", "nmsl" -> "你妈死了"
        # 将在初始化时从配置文件加载
    }

    # 特殊符号集 (用于去除干扰字符)
    SPECIAL_SYMBOLS_PATTERN = re.compile(
        r"[。！？，、；：\u201c\u201d\u2018\u2019【】《》（）…—\-–\s"
        r"~`@#$%^&*()_+=\[\]{}|\\;:'\",.<>/?"
        r"0123456789"
        r"☆★●○◎◇◆□■△▽♦♤♣♡♥"
        r"·ˇˊˋˍ↔↕‼⁉⁇"
        r"₀₁₂₃₄₅₆₇₈₉"
        r"①②③④⑤⑥⑦⑧⑨⑩"
        r"]+"
    )

    # 重复字符压缩 (如 "傻傻傻逼逼逼" -> "傻逼")
    REPEAT_PATTERN = re.compile(r"(.)\1{2,}")

    # 全角转半角映射
    FULLWIDTH_MAP = str.maketrans(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
        "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz",
    )

    def __init__(self, homophone_map: dict[str, str] | None = None):
        if homophone_map:
            self.HOMOPHONE_MAP = homophone_map

    def normalize(self, text: str) -> str:
        """
        完整的标准化流程:
        1. 全角转半角
        2. Unicode标准化 (NFKC)
        3. 转小写
        4. 去除特殊符号
        5. 压缩重复字符
        6. 谐音替换
        """
        text = self._fullwidth_to_halfwidth(text)
        text = self._unicode_normalize(text)
        text = text.lower()
        text = self._remove_special_symbols(text)
        text = self._compress_repeats(text)
        text = self._replace_homophones(text)
        return text.strip()

    def normalize_for_display(self, text: str) -> str:
        """保留原文可读性的轻量标准化 (用于展示)"""
        text = self._fullwidth_to_halfwidth(text)
        text = self._unicode_normalize(text)
        text = text.lower()
        return text.strip()

    def extract_pinyin_pattern(self, text: str) -> str:
        """
        提取纯字母部分 (用于拼音缩写匹配)
        例如: "nmsl" -> "nmsl", "你nmsl啊" -> "nmsl"
        """
        letters = re.findall(r"[a-zA-Z]+", text)
        return "".join(letters).lower()

    def _fullwidth_to_halfwidth(self, text: str) -> str:
        return text.translate(self.FULLWIDTH_MAP)

    def _unicode_normalize(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def _remove_special_symbols(self, text: str) -> str:
        return self.SPECIAL_SYMBOLS_PATTERN.sub("", text)

    def _compress_repeats(self, text: str) -> str:
        """压缩重复字符为2个 (保留语气强调效果)"""
        return self.REPEAT_PATTERN.sub(r"\1\1", text)

    def _replace_homophones(self, text: str) -> str:
        """谐音/拼音缩写替换"""
        for key, value in self.HOMOPHONE_MAP.items():
            text = text.replace(key, value)
        return text
