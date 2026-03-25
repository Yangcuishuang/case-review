"""
关键词匹配引擎 - 按业务分类体系匹配
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

try:
    import ahocorasick
    HAS_AC = True
except ImportError:
    try:
        import pyahocorasick as ahocorasick
        HAS_AC = True
    except ImportError:
        HAS_AC = False

from src.models import (
    ViolationSubType, ViolationLevel, DetectionSource,
    DetectionResult, SentenceHit, HitRule,
    SUBTYPE_TO_LEVEL,
)
from src.utils.text_normalizer import TextNormalizer

# 关键词文件 -> 违规子类型 的映射
KEYWORD_FILE_MAP = {
    "swearword": ViolationSubType.SWEARWORD,
    "mild_insult": ViolationSubType.MILD_INSULT,
    "family_curse": ViolationSubType.FAMILY_CURSE,
    "sexual_curse": ViolationSubType.SEXUAL_INSULT,
    "advertisement": ViolationSubType.ADVERTISEMENT,
    "political": ViolationSubType.POLITICAL,
}


class KeywordEngine:
    def __init__(self, keywords_dir: str, normalizer: TextNormalizer):
        self.normalizer = normalizer
        self.keyword_map: dict[str, ViolationSubType] = {}
        self.ac_map: dict[ViolationSubType, object] = {}
        self._load_keywords(keywords_dir)

    def _load_keywords(self, keywords_dir: str):
        base = Path(keywords_dir)
        for filename, sub_type in KEYWORD_FILE_MAP.items():
            filepath = base / f"{filename}.json"
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                words = json.load(f)
            for w in words:
                w = str(w).strip().lower()
                if w:
                    self.keyword_map[w] = sub_type

        # 按子类型构建AC自动机（提高匹配效率）
        type_words = defaultdict(list)
        for word, stype in self.keyword_map.items():
            type_words[stype].append(word)

        if HAS_AC:
            for stype, words in type_words.items():
                ac = ahocorasick.Automaton()
                for w in words:
                    ac.add_word(w, w)
                ac.make_automaton()
                self.ac_map[stype] = ac

    def detect_sentence(self, sentence: str) -> list[HitRule]:
        """检测单句话，返回所有命中的关键词规则"""
        normalized = self.normalizer.normalize(sentence)
        hits = []
        seen = set()

        if HAS_AC:
            for stype, ac in self.ac_map.items():
                for _, matched in ac.iter(normalized):
                    key = (stype.value, matched)
                    if key not in seen:
                        seen.add(key)
                        hits.append(HitRule(
                            source=DetectionSource.KEYWORD,
                            matched_text=matched,
                            rule_id=f"kw_{stype.value}_{matched}",
                            rule_description=f"关键词命中: {matched}",
                            sub_type=stype,
                        ))
        else:
            for word, stype in self.keyword_map.items():
                if word in normalized:
                    key = (stype.value, word)
                    if key not in seen:
                        seen.add(key)
                        hits.append(HitRule(
                            source=DetectionSource.KEYWORD,
                            matched_text=word,
                            rule_id=f"kw_{stype.value}_{word}",
                            rule_description=f"关键词命中: {word}",
                            sub_type=stype,
                        ))
        return hits

    def detect(self, reported_sentences: list[str]) -> DetectionResult:
        """
        检测被举报玩家的所有发言
        
        Args:
            reported_sentences: 被举报玩家的发言列表
            
        Returns:
            DetectionResult，包含每句话的命中详情
        """
        all_sentence_hits: list[SentenceHit] = []
        all_rules: list[HitRule] = []

        for idx, sentence in enumerate(reported_sentences):
            hits = self.detect_sentence(sentence)
            if hits:
                # 找到最严重的子类型
                most_severe = max(hits, key=lambda h: h.sub_type.value)
                all_sentence_hits.append(SentenceHit(
                    sentence=sentence,
                    sentence_index=idx,
                    is_violation=True,
                    sub_type=most_severe.sub_type,
                    confidence=0.95,
                    hit_rules=hits,
                ))
                all_rules.extend(hits)

        is_violation = len(all_sentence_hits) > 0
        sub_type = ViolationSubType.NONE
        if all_sentence_hits:
            sub_type = max(
                all_sentence_hits,
                key=lambda h: h.sub_type.value if h.sub_type != ViolationSubType.NONE else -1
            ).sub_type

        return DetectionResult(
            source=DetectionSource.KEYWORD,
            sentence_hits=all_sentence_hits,
            is_violation=is_violation,
            violation_level=SUBTYPE_TO_LEVEL.get(sub_type, ViolationLevel.NONE),
            violation_sub_type=sub_type,
            confidence=0.95 if is_violation else 0.0,
            reason=f"关键词命中 {len(all_sentence_hits)} 句违规" if is_violation else "关键词未命中",
        )
