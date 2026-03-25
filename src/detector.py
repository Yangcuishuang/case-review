"""
判罚调度器 - 三级级联 + 句数阈值 + 上下文综合判罚
"""
from __future__ import annotations
import json
import logging
import uuid
from pathlib import Path
from datetime import datetime

import yaml

from src.models import (
    ViolationLevel, ViolationSubType, DetectionResult, SentenceHit,
    PenaltyResult, ChatMessage, HitRule,
    SUBTYPE_TO_LEVEL, SENTENCE_THRESHOLDS, SUBTYPE_LABEL_CN,
    DetectionSource,
)
from src.engines.keyword_engine import KeywordEngine
from src.engines.llm_engine import LLMEngine
from src.utils.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)

# 严重程度排序（用于取最严重的违规）
SEVERITY_ORDER = {
    ViolationSubType.NONE: 0,
    ViolationSubType.MINDLESS_HARASS: 1,
    ViolationSubType.NEGATIVE_SPEECH: 1,
    ViolationSubType.SWEARWORD: 1,
    ViolationSubType.REPORT_POSITION: 1,
    ViolationSubType.MILD_INSULT: 2,
    ViolationSubType.FAMILY_CURSE: 4,
    ViolationSubType.FEMALE_CURSE: 4,
    ViolationSubType.SEXUAL_INSULT: 5,
    ViolationSubType.DEATH_CURSE: 4,
    ViolationSubType.REGIONAL_DISCRIM: 6,
    ViolationSubType.RACIAL_DISCRIM: 7,
    ViolationSubType.SEVERE_PORN: 8,
    ViolationSubType.DRUG_GAMBLING: 8,
    ViolationSubType.ADVERTISEMENT: 8,
    ViolationSubType.POLITICAL: 9,
}

# 直接≥1句即成立的严重子类型（关键词命中即判，不需进入LLM）
IMMEDIATE_SEVERE_TYPES = {
    ViolationSubType.RACIAL_DISCRIM,
    ViolationSubType.SEVERE_PORN,
    ViolationSubType.DRUG_GAMBLING,
    ViolationSubType.POLITICAL,
}

# ≥1句即成立的中度子类型
IMMEDIATE_MODERATE_TYPES = {
    ViolationSubType.SEXUAL_INSULT,
}

# ≥1句即成立的轻度子类型
IMMEDIATE_MILD_TYPES = {
    ViolationSubType.MILD_INSULT,
}


class AbuseDetector:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.normalizer = TextNormalizer()
        self._init_engines()

    def _load_config(self, config_path: str) -> dict:
        path = Path(config_path)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _init_engines(self):
        engine_config = self.config.get("engine", {})

        kw_config = engine_config.get("keyword", {})
        if kw_config.get("enabled", True):
            kw_dir = kw_config.get("keywords_dir", "data/keywords")
            self.keyword_engine = KeywordEngine(kw_dir, self.normalizer)
        else:
            self.keyword_engine = None

        llm_config = engine_config.get("llm", {})
        if llm_config.get("enabled", True):
            fewshot_file = llm_config.get("fewshot_file", "data/cases/all_examples.json")
            self.llm_engine = LLMEngine(llm_config, self.normalizer, fewshot_file)
        else:
            self.llm_engine = None

    def detect(
        self,
        chat_messages: list[ChatMessage],
        reported_id: str,
        reporter_id: str,
        case_id: str = "",
    ) -> PenaltyResult:
        if not case_id:
            case_id = f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # 提取被举报玩家的发言
        reported_messages = [m for m in chat_messages if m.sender_id == reported_id]
        reported_sentences = [m.content for m in reported_messages]
        total_sentences = len(reported_sentences)

        if not reported_sentences:
            return self._build_result(
                case_id, reported_id, reporter_id, chat_messages,
                is_violation=False, level=ViolationLevel.NONE,
                sub_type=ViolationSubType.NONE, confidence=1.0,
                reason="被举报玩家无发言记录", needs_review=False,
                total_sents=0, violation_count=0,
            )

        detection_chain: list[DetectionResult] = []

        # === 第一级：关键词匹配 ===
        kw_result = None
        if self.keyword_engine:
            kw_result = self.keyword_engine.detect(reported_sentences)
            if kw_result.is_violation:
                detection_chain.append(kw_result)

        # 基于关键词结果进行句数阈值判罚
        if kw_result and kw_result.is_violation:
            decision = self._apply_threshold_rules(kw_result, total_sentences)
            if decision["decided"]:
                return self._build_result(
                    case_id, reported_id, reporter_id, chat_messages,
                    is_violation=decision["is_violation"],
                    level=decision["level"],
                    sub_type=decision["sub_type"],
                    confidence=decision["confidence"],
                    reason=decision["reason"],
                    needs_review=decision["needs_review"],
                    review_reason=decision.get("review_reason", ""),
                    chain=detection_chain,
                    rules=kw_result.sentence_hits[0].hit_rules if kw_result.sentence_hits else [],
                    sent_hits=kw_result.sentence_hits,
                    total_sents=total_sentences,
                    violation_count=decision.get("violation_count", 0),
                )

        # === 第二级：LLM 语义判罚 ===
        if self.llm_engine:
            llm_result = self.llm_engine.detect(
                chat_messages, reported_sentences, reported_id, reporter_id,
            )
            detection_chain.append(llm_result)

            if llm_result.is_violation:
                return self._build_result(
                    case_id, reported_id, reporter_id, chat_messages,
                    is_violation=True,
                    level=llm_result.violation_level,
                    sub_type=llm_result.violation_sub_type,
                    confidence=llm_result.confidence,
                    reason=llm_result.reason,
                    needs_review=llm_result.confidence < 0.6,
                    review_reason="LLM置信度较低，建议人工复核" if llm_result.confidence < 0.6 else "",
                    chain=detection_chain,
                    rules=[h for sh in llm_result.sentence_hits for h in sh.hit_rules],
                    sent_hits=llm_result.sentence_hits,
                    total_sents=total_sentences,
                    violation_count=len(llm_result.sentence_hits),
                )
            else:
                needs_review = llm_result.confidence < 0.7
                return self._build_result(
                    case_id, reported_id, reporter_id, chat_messages,
                    is_violation=False,
                    level=ViolationLevel.NONE,
                    sub_type=ViolationSubType.NONE,
                    confidence=llm_result.confidence,
                    reason=llm_result.reason,
                    needs_review=needs_review,
                    review_reason="LLM无法确定是否违规，建议人工复核" if needs_review else "",
                    chain=detection_chain,
                    total_sents=total_sentences,
                    violation_count=0,
                )

        # 无引擎可用
        return self._build_result(
            case_id, reported_id, reporter_id, chat_messages,
            is_violation=False, level=ViolationLevel.NONE,
            sub_type=ViolationSubType.NONE, confidence=0.3,
            reason="所有引擎均未检测到违规，建议人工复核",
            needs_review=True,
            review_reason="所有引擎均未命中，建议人工复核",
            chain=detection_chain,
            total_sents=total_sentences,
            violation_count=0,
        )

    def _apply_threshold_rules(self, kw_result: DetectionResult, total_sentences: int) -> dict:
        """
        核心判罚逻辑：根据句数阈值和业务规则确定最终判罚
        
        关键规则：
        1. 严重类（涉政/种族歧视/色情/赌毒）≥1句即成立
        2. 中度类（辱骂家人/辱骂女性/性侮辱）：
           - ≥3句→中度
           - 1句+其他轻度词汇→中度
           - 纯1句→降级为轻度
        3. 轻度类（口头禅/消极/骚扰）≥3句→轻度
        4. 普通辱骂≥1句→轻度
        """
        sentence_hits = kw_result.sentence_hits
        if not sentence_hits:
            return {"decided": False}

        # 统计各子类型的命中句数
        subtype_counts: dict[ViolationSubType, int] = {}
        for sh in sentence_hits:
            st = sh.sub_type
            subtype_counts[st] = subtype_counts.get(st, 0) + 1

        # 找到最严重的子类型
        most_severe_type = max(subtype_counts.keys(), key=lambda x: SEVERITY_ORDER.get(x, 0))
        most_severe_count = subtype_counts[most_severe_type]

        # 总违规句数
        total_violation_count = len(sentence_hits)

        # 规则1: 严重类 ≥1句即成立
        if most_severe_type in IMMEDIATE_SEVERE_TYPES and most_severe_count >= 1:
            return {
                "decided": True,
                "is_violation": True,
                "level": ViolationLevel.SEVERE,
                "sub_type": most_severe_type,
                "confidence": 0.95,
                "reason": self._reason(
                    most_severe_type, most_severe_count, total_violation_count, total_sentences
                ),
                "needs_review": False,
                "violation_count": total_violation_count,
            }

        # 规则1.5: 性侮辱 ≥1句即中度
        if most_severe_type in IMMEDIATE_MODERATE_TYPES and most_severe_count >= 1:
            return {
                "decided": True,
                "is_violation": True,
                "level": ViolationLevel.MODERATE,
                "sub_type": most_severe_type,
                "confidence": 0.95,
                "reason": self._reason(
                    most_severe_type, most_severe_count, total_violation_count, total_sentences
                ),
                "needs_review": False,
                "violation_count": total_violation_count,
            }

        # 规则2: 普通辱骂 ≥1句即轻度
        if ViolationSubType.MILD_INSULT in subtype_counts:
            mild_count = subtype_counts[ViolationSubType.MILD_INSULT]
            if mild_count >= 1:
                # 检查是否还有中度词汇（辱骂家人/辱骂女性各1句）
                has_family = ViolationSubType.FAMILY_CURSE in subtype_counts
                has_female = ViolationSubType.FEMALE_CURSE in subtype_counts

                if has_family or has_female:
                    # 1句中度词汇 + 轻度辱骂 → 中度
                    moderate_type = (
                        ViolationSubType.FAMILY_CURSE if has_family
                        else ViolationSubType.FEMALE_CURSE
                    )
                    return {
                        "decided": True,
                        "is_violation": True,
                        "level": ViolationLevel.MODERATE,
                        "sub_type": moderate_type,
                        "confidence": 0.9,
                        "reason": self._reason(
                            moderate_type, subtype_counts[moderate_type],
                            total_violation_count, total_sentences,
                            extra="且伴有轻度辱骂词汇，升级为中度"
                        ),
                        "needs_review": False,
                        "violation_count": total_violation_count,
                    }

                # 纯轻度辱骂
                return {
                    "decided": True,
                    "is_violation": True,
                    "level": ViolationLevel.MILD,
                    "sub_type": ViolationSubType.MILD_INSULT,
                    "confidence": 0.95,
                    "reason": self._reason(
                        ViolationSubType.MILD_INSULT, mild_count,
                        total_violation_count, total_sentences
                    ),
                    "needs_review": False,
                    "violation_count": total_violation_count,
                }

        # 规则3: 中度类（辱骂家人/辱骂女性）
        moderate_types = {ViolationSubType.FAMILY_CURSE, ViolationSubType.FEMALE_CURSE, ViolationSubType.DEATH_CURSE}
        found_moderate = moderate_types & set(subtype_counts.keys())
        if found_moderate:
            mod_type = max(found_moderate, key=lambda x: SEVERITY_ORDER.get(x, 0))
            mod_count = subtype_counts[mod_type]

            if mod_count >= 3:
                return {
                    "decided": True, "is_violation": True,
                    "level": ViolationLevel.MODERATE, "sub_type": mod_type,
                    "confidence": 0.9,
                    "reason": self._reason(mod_type, mod_count, total_violation_count, total_sentences),
                    "needs_review": False,
                    "violation_count": total_violation_count,
                }
            elif mod_count >= 2:
                # 2句中度 → 需要LLM判断恶意程度（标记需复核）
                return {
                    "decided": False,  # 交给LLM处理
                }
            elif mod_count == 1:
                # 1句中度 + 其他轻度 → 中度; 纯1句 → 降级轻度
                if total_violation_count > 1:
                    return {
                        "decided": True, "is_violation": True,
                        "level": ViolationLevel.MODERATE, "sub_type": mod_type,
                        "confidence": 0.85,
                        "reason": self._reason(
                            mod_type, mod_count, total_violation_count, total_sentences,
                            extra="且伴有其他轻度违规言论，判定为中度"
                        ),
                        "needs_review": False,
                        "violation_count": total_violation_count,
                    }
                else:
                    # 纯1句中度 → 降级为轻度
                    return {
                        "decided": True, "is_violation": True,
                        "level": ViolationLevel.MILD, "sub_type": ViolationSubType.MILD_INSULT,
                        "confidence": 0.7,
                        "reason": f"仅有1句{SUBTYPE_LABEL_CN.get(mod_type.value, '')}类词汇，无其他违规言论，降级为轻度",
                        "needs_review": True,
                        "review_reason": "1句中度词汇但无其他违规，已降级为轻度，建议复核",
                        "violation_count": 1,
                    }

        # 规则4: 地域歧视
        if ViolationSubType.REGIONAL_DISCRIM in subtype_counts:
            reg_count = subtype_counts[ViolationSubType.REGIONAL_DISCRIM]
            if reg_count >= 2:
                return {
                    "decided": True, "is_violation": True,
                    "level": ViolationLevel.SEVERE, "sub_type": ViolationSubType.REGIONAL_DISCRIM,
                    "confidence": 0.9,
                    "reason": self._reason(ViolationSubType.REGIONAL_DISCRIM, reg_count, total_violation_count, total_sentences),
                    "needs_review": False,
                    "violation_count": total_violation_count,
                }
            elif reg_count == 1:
                # 1句地域歧视 → 需LLM判断恶意程度
                return {"decided": False}

        # 规则5: 轻度类（口头禅/消极/骚扰）≥3句
        mild_types = {
            ViolationSubType.SWEARWORD,
            ViolationSubType.NEGATIVE_SPEECH,
            ViolationSubType.MINDLESS_HARASS,
            ViolationSubType.REPORT_POSITION,
        }
        found_mild = mild_types & set(subtype_counts.keys())
        if found_mild and total_violation_count >= 3:
            mild_type = max(found_mild, key=lambda x: SEVERITY_ORDER.get(x, 0))
            return {
                "decided": True, "is_violation": True,
                "level": ViolationLevel.MILD, "sub_type": mild_type,
                "confidence": 0.9,
                "reason": self._reason(mild_type, subtype_counts[mild_type], total_violation_count, total_sentences),
                "needs_review": False,
                "violation_count": total_violation_count,
            }

        # 关键词有命中但句数不够 → 交给LLM做上下文判断
        return {"decided": False}

    @staticmethod
    def _reason(sub_type: ViolationSubType, count: int, total: int, total_sents: int, extra: str = "") -> str:
        label = SUBTYPE_LABEL_CN.get(sub_type.value, sub_type.value)
        level = SUBTYPE_TO_LEVEL.get(sub_type, ViolationLevel.NONE).label_cn
        msg = f"检测到{count}句「{label}」类违规（共{total}句违规/被举报人{total_sents}句发言），判定为{level}"
        if extra:
            msg += f"。{extra}"
        return msg

    @staticmethod
    def _build_result(
        case_id: str, reported_id: str, reporter_id: str,
        chat_messages: list[ChatMessage],
        is_violation: bool, level: ViolationLevel, sub_type: ViolationSubType,
        confidence: float, reason: str, needs_review: bool,
        review_reason: str = "",
        chain: list[DetectionResult] | None = None,
        rules: list[HitRule] | None = None,
        sent_hits: list[SentenceHit] | None = None,
        total_sents: int = 0, violation_count: int = 0,
    ) -> PenaltyResult:
        return PenaltyResult(
            case_id=case_id,
            reported_player_id=reported_id,
            reporter_player_id=reporter_id,
            is_violation=is_violation,
            violation_level=level,
            violation_sub_type=sub_type,
            violation_sub_type_label=SUBTYPE_LABEL_CN.get(sub_type.value, ""),
            confidence=confidence,
            hit_rules=rules or [],
            sentence_hits=sent_hits or [],
            reason=reason,
            detection_chain=chain or [],
            needs_manual_review=needs_review,
            manual_review_reason=review_reason,
            chat_context=chat_messages,
            total_reported_sentences=total_sents,
            violation_sentence_count=violation_count,
        )

    def detect_batch(self, cases: list[dict]) -> list[PenaltyResult]:
        results = []
        for i, case in enumerate(cases):
            logger.info(f"处理第 {i+1}/{len(cases)} 个案例...")
            messages = []
            for msg in case.get("chat_messages", []):
                if isinstance(msg, ChatMessage):
                    messages.append(msg)
                elif isinstance(msg, dict):
                    messages.append(ChatMessage(**msg))
            result = self.detect(
                chat_messages=messages,
                reported_id=case["reported_id"],
                reporter_id=case["reporter_id"],
                case_id=case.get("case_id", ""),
            )
            results.append(result)
        return results
