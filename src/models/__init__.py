"""
数据模型 - 按照实际业务规则定义
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ViolationSubType(str, Enum):
    """违规子类型（对应业务中的违规小类）"""
    NONE = "none"
    # 轻度 - 需要≥3句才成立
    MINDLESS_HARASS = "mindless_harass"        # 无意义言语及骚扰、嘲讽
    NEGATIVE_SPEECH = "negative_speech"         # 消极言论
    SWEARWORD = "swearword"                     # 口头禅
    REPORT_POSITION = "report_position"         # 报点言论
    # 轻度 - ≥1句即成立
    MILD_INSULT = "mild_insult"                 # 普通辱骂(个人/群体指向性攻击)
    # 中度
    FAMILY_CURSE = "family_curse"               # 辱骂家人
    FEMALE_CURSE = "female_curse"               # 辱骂女性
    SEXUAL_INSULT = "sexual_insult"             # 性侮辱女性
    DEATH_CURSE = "death_curse"                 # 涉及家人的死亡诅咒
    # 严重
    REGIONAL_DISCRIM = "regional_discrim"       # 地域黑
    RACIAL_DISCRIM = "racial_discrim"           # 种族歧视
    SEVERE_PORN = "severe_porn"                 # 严重色情淫秽
    DRUG_GAMBLING = "drug_gambling"             # 赌毒言论
    ADVERTISEMENT = "advertisement"             # 代练/宣传/广告
    POLITICAL = "political"                     # 涉政


class ViolationLevel(str, Enum):
    """违规等级（大分类）"""
    NONE = "none"
    MILD = "mild"           # 轻度言语违规 (1天禁言)
    MODERATE = "moderate"   # 中度言语违规 (15天禁言)
    SEVERE = "severe"       # 严重言语违规 (30天禁言)

    @property
    def label_cn(self) -> str:
        mapping = {
            "none": "无违规",
            "mild": "轻度言语违规",
            "moderate": "中度言语违规",
            "severe": "严重言语违规",
        }
        return mapping.get(self.value, self.value)

    @property
    def penalty(self) -> str:
        mapping = {
            "none": "无处罚",
            "mild": "1天禁言",
            "moderate": "15天禁言",
            "severe": "30天禁言",
        }
        return mapping.get(self.value, "")


# 子类型 -> 大等级 的映射
SUBTYPE_TO_LEVEL = {
    ViolationSubType.NONE: ViolationLevel.NONE,
    ViolationSubType.MINDLESS_HARASS: ViolationLevel.MILD,
    ViolationSubType.NEGATIVE_SPEECH: ViolationLevel.MILD,
    ViolationSubType.SWEARWORD: ViolationLevel.MILD,
    ViolationSubType.REPORT_POSITION: ViolationLevel.MILD,
    ViolationSubType.MILD_INSULT: ViolationLevel.MILD,
    ViolationSubType.FAMILY_CURSE: ViolationLevel.MODERATE,
    ViolationSubType.FEMALE_CURSE: ViolationLevel.MODERATE,
    ViolationSubType.SEXUAL_INSULT: ViolationLevel.MODERATE,
    ViolationSubType.DEATH_CURSE: ViolationLevel.MODERATE,
    ViolationSubType.REGIONAL_DISCRIM: ViolationLevel.SEVERE,
    ViolationSubType.RACIAL_DISCRIM: ViolationLevel.SEVERE,
    ViolationSubType.SEVERE_PORN: ViolationLevel.SEVERE,
    ViolationSubType.DRUG_GAMBLING: ViolationLevel.SEVERE,
    ViolationSubType.ADVERTISEMENT: ViolationLevel.SEVERE,
    ViolationSubType.POLITICAL: ViolationLevel.SEVERE,
}

# 各子类型的句数阈值（被举报玩家发言需满足的违规句数）
# -1 表示该类型句数阈值需结合上下文判断（由LLM处理）
SENTENCE_THRESHOLDS = {
    ViolationSubType.MINDLESS_HARASS: 3,
    ViolationSubType.NEGATIVE_SPEECH: 3,
    ViolationSubType.SWEARWORD: 3,
    ViolationSubType.REPORT_POSITION: 3,
    ViolationSubType.MILD_INSULT: 1,
    ViolationSubType.FAMILY_CURSE: 3,     # 特殊: 1句+其他轻度=中度, 纯1句=轻度
    ViolationSubType.FEMALE_CURSE: 3,     # 特殊: 1句+其他轻度=中度, 纯1句=轻度
    ViolationSubType.SEXUAL_INSULT: 1,    # 特殊: 隐晦≥2句=中度, 直接≥1句=中度
    ViolationSubType.DEATH_CURSE: 3,
    ViolationSubType.REGIONAL_DISCRIM: 2,  # 特殊: 恶意强1句=严重, 敏感地区1句=中度
    ViolationSubType.RACIAL_DISCRIM: 1,
    ViolationSubType.SEVERE_PORN: 1,
    ViolationSubType.DRUG_GAMBLING: 1,
    ViolationSubType.ADVERTISEMENT: 3,
    ViolationSubType.POLITICAL: 1,
}

SUBTYPE_LABEL_CN = {
    "none": "无违规",
    "mindless_harass": "无意义言语及骚扰",
    "negative_speech": "消极言论",
    "swearword": "口头禅",
    "report_position": "报点言论",
    "mild_insult": "普通辱骂",
    "family_curse": "辱骂家人",
    "female_curse": "辱骂女性",
    "sexual_insult": "性侮辱女性",
    "death_curse": "死亡诅咒",
    "regional_discrim": "地域歧视",
    "racial_discrim": "种族歧视",
    "severe_porn": "严重色情淫秽",
    "drug_gambling": "赌毒言论",
    "advertisement": "代练/广告",
    "political": "涉政",
}


class DetectionSource(str, Enum):
    KEYWORD = "keyword"
    RULE = "rule"
    LLM = "llm"


class HitRule(BaseModel):
    source: DetectionSource
    matched_text: str
    rule_id: str
    rule_description: str
    sub_type: ViolationSubType = ViolationSubType.NONE
    sentence_index: int = Field(default=-1, description="该命中在哪句话中(0-based)")


class ChatMessage(BaseModel):
    sender_id: str
    sender_name: str
    content: str
    timestamp: Optional[str] = None


class SentenceHit(BaseModel):
    """单句话的检测结果"""
    sentence: str
    sentence_index: int
    is_violation: bool
    sub_type: ViolationSubType = ViolationSubType.NONE
    confidence: float = 0.0
    hit_rules: list[HitRule] = Field(default_factory=list)


class DetectionResult(BaseModel):
    """单个检测引擎的输出"""
    source: DetectionSource
    sentence_hits: list[SentenceHit] = Field(default_factory=list)
    is_violation: bool = False
    violation_level: ViolationLevel = ViolationLevel.NONE
    violation_sub_type: ViolationSubType = ViolationSubType.NONE
    confidence: float = 0.0
    reason: str = ""


class PenaltyResult(BaseModel):
    """最终判罚结果"""
    case_id: str = ""
    reported_player_id: str
    reporter_player_id: str
    is_violation: bool
    violation_level: ViolationLevel = ViolationLevel.NONE
    violation_sub_type: ViolationSubType = ViolationSubType.NONE
    violation_sub_type_label: str = ""
    confidence: float = 0.0
    hit_rules: list[HitRule] = Field(default_factory=list)
    sentence_hits: list[SentenceHit] = Field(default_factory=list)
    reason: str = ""
    detection_chain: list[DetectionResult] = Field(default_factory=list)
    needs_manual_review: bool = False
    manual_review_reason: str = ""
    chat_context: list[ChatMessage] = Field(default_factory=list)
    total_reported_sentences: int = Field(default=0, description="被举报人发言总句数")
    violation_sentence_count: int = Field(default=0, description="违规句数")
