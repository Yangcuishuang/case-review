"""
LLM 语义判罚引擎 - 注入完整业务规则
"""
from __future__ import annotations
import json
import asyncio
import concurrent.futures
from pathlib import Path

from src.models import (
    ViolationLevel, ViolationSubType, DetectionSource,
    DetectionResult, SentenceHit, HitRule, ChatMessage,
    SUBTYPE_TO_LEVEL, SUBTYPE_LABEL_CN,
)
from src.utils.text_normalizer import TextNormalizer

SYSTEM_PROMPT = """你是一个游戏聊天内容审核专家。你的任务是分析玩家对话，判断被举报玩家是否存在违规行为。

## 违规等级和处罚

### 轻度言语违规（1天禁言）
以下类型原则上需要被举报玩家≥3句才成立：

1. **无意义言语及骚扰、嘲讽**
   - 不包含辱骂词汇，但言语与游戏无关或扰乱队友
   - 如：我想跟你处对象、软辅有什么用、你回家种地吧、你还不如超级兵
   - 特殊逻辑：无

2. **消极言论**
   - 涉及投降、主动向敌方消极言论
   - 如：六分投、上票，右上角福袋，或向敌方发送：我给你们送头
   - 特殊逻辑：无

3. **口头禅**
   - 辱骂名词，但无指向性
   - 如：猪、FW、傻嗨、没长眼睛、废物、逼傻、菜鸡、叼毛
   - 特殊逻辑：无

4. **报点言论**
   - 向敌方报点（队友位置信息、野怪/BUFF/龙的刷新情况）
   - 如：我们打野在打龙，我们的蓝buff刷了快去打
   - 特殊逻辑：要区分是否全部发送，若只发给队友不处罚

以下类型≥1句即成立：

5. **普通辱骂**
   - 个人指向性攻击（你/我/某人/某分路+辱骂性词汇）
   - 如：你个傻逼、小乔脑残、都是弱智、队友都是智障
   - 群体指向性攻击（一群/全/都/X个+辱骂性词汇）
   - 如：一群傻逼、三只猪、全是废物
   - 特殊逻辑：无

### 中度言语违规（15天禁言）

6. **辱骂家人**
   - 涉及家人的辱骂或诅咒
   - 如：你全家死绝、母狗、女人不配赢
   - 原则上≥3句成立；恶意较强≥2句成立；严重程度非常高≥1句成立
   - **特殊逻辑**：若只有1句辱骂家人+无其他言论→轻度；1句辱骂家人+其他轻度辱骂词汇→中度

7. **辱骂女性**
   - 涉及女性的辱骂或诅咒
   - 如：你这种母狗只配在我胯下
   - 句数逻辑同上
   - **特殊逻辑**：同性恋侮辱≥3句处罚；涉及男性=轻度（如男炮、男娘）；涉及女性=中度

8. **性侮辱女性**
   - 涉及侮辱或践踏女性（性侮辱性词汇）
   - 如：妲己卖自己的洞洞一晚上20
   - **特殊逻辑**：色情隐晦≥2句→中度；色情直接无动作（如约炮吗）≥1句→中度

9. **死亡诅咒**
   - 涉及家人的死亡诅咒
   - 如：你妈今天出门小心车，今晚全家福就你一个
   - ≥3句成立

### 严重言语违规（30天禁言）

10. **地域黑**
    - 涉及地域歧视、民族歧视、轻微政治
    - 如：中国人素质真低、南蛮子、东北狗、河南人偷井盖
    - 原则上≥2句成立；恶意较强≥1句成立
    - **特殊逻辑**：只说敏感地区词汇无辱骂（如新疆人、河南人）≥3句→轻度；高敏地区（大陆狗/台湾狗/香港狗）≥1句→中度；只一句轻度地域黑→轻度

11. **种族歧视**
    - 外国人对中国人的歧视、台湾人对大陆人的歧视、大陆内部各省歧视
    - ≥1句成立
    - **特殊逻辑**：骂对方是日本人→轻度

12. **严重色情淫秽**
    - 特别严重、恶意较强的色情言论，常伴随动作
    - 如：信不信直接给你插满，通水直流小乔
    - ≥1句成立

13. **赌毒言论**
    - 如：海洛因真是好、买冰毒加我、加群免费参加赌球
    - ≥1句成立

14. **代练/宣传/广告**
    - 如：通天+V13214567M代+Q1234564897科技群
    - ≥3句成立

15. **涉政**
    - 将国家历史用于辱骂、辱骂国家及领导人、涉及国家分裂
    - ≥1句成立
    - 如：你妈就是慰安妇、中国就该灭亡

## 判罚原则
1. 重点看被举报玩家的发言
2. 如果举报人先辱骂对方，被举报人的反击可适当减轻判定
3. 拼音缩写、谐音替换、符号规避等同于原词
4. 需要统计违规句数来决定最终等级

## 输出格式
严格按以下JSON输出（不要输出其他内容）：
```json
{
    "is_violation": true,
    "violation_sub_type": "none/swearword/mild_insult/family_curse/female_curse/sexual_insult/death_curse/regional_discrim/racial_discrim/severe_porn/drug_gambling/advertisement/political/negative_speech/mindless_harass/report_position",
    "violation_level": "none/mild/moderate/severe",
    "confidence": 0.95,
    "reason": "判罚理由",
    "violation_sentences": ["违规句子1", "违规句子2"],
    "hit_keywords": ["命中的关键词"],
    "sentence_count_total": 10,
    "violation_sentence_count": 3,
    "context_analysis": "上下文分析"
}
```
"""

FEWSHOT_SECTION = """
## 参考案例（人工已标注，请严格对齐这些案例的判罚尺度）

{examples}
"""

USER_PROMPT = """请分析以下游戏对话，判断被举报玩家是否违规。

## 完整对话记录
{chat_history}

## 举报人: {reporter_name} (标记为[举报])
## 被举报人: {reported_name} (标记为[被举报])

请严格按照上述规则进行判罚，特别注意句数阈值和特殊逻辑。
"""


class LLMEngine:
    def __init__(self, config: dict, normalizer: TextNormalizer, fewshot_file: str | None = None):
        self.api_url = config.get("api_url", "")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2048)
        self.timeout = config.get("timeout", 60)
        self.normalizer = normalizer
        self.fewshot_examples: list[dict] = []

        if fewshot_file and Path(fewshot_file).exists():
            self._load_fewshot(fewshot_file)

    def _load_fewshot(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            self.fewshot_examples = data.get("examples", [])
        elif isinstance(data, list):
            self.fewshot_examples = data

    def _build_prompt(self, chat_messages: list[ChatMessage], reported_id: str, reporter_id: str):
        # Few-shot
        fewshot_text = ""
        if self.fewshot_examples:
            ex_parts = []
            for i, ex in enumerate(self.fewshot_examples[:20], 1):  # 最多20个
                ex_parts.append(
                    f"### 案例{i}\n"
                    f"判罚: {ex.get('judgment', '')}\n"
                    f"等级: {ex.get('level_label', '')}\n"
                    f"备注: {ex.get('note', '')}\n"
                    f"对话:\n{ex.get('chat', 'N/A')}"
                )
            fewshot_text = FEWSHOT_SECTION.format(examples="\n\n".join(ex_parts))

        system = SYSTEM_PROMPT + fewshot_text

        # 格式化对话
        reported_name = ""
        reporter_name = ""
        chat_lines = []
        for msg in chat_messages:
            if msg.sender_id == reported_id:
                tag = "[被举报]"
                if not reported_name:
                    reported_name = msg.sender_name
            elif msg.sender_id == reporter_id:
                tag = "[举报]"
                if not reporter_name:
                    reporter_name = msg.sender_name
            else:
                tag = "[其他]"
            chat_lines.append(f"{tag} {msg.sender_name}: {msg.content}")

        user = USER_PROMPT.format(
            chat_history="\n".join(chat_lines),
            reported_name=reported_name or reported_id,
            reporter_name=reporter_name or reporter_id,
        )
        return system, user

    async def _call_llm(self, system: str, user: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def _parse_response(self, content: str, reported_sentences: list[str]) -> DetectionResult:
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return DetectionResult(
                source=DetectionSource.LLM,
                is_violation=False,
                confidence=0.1,
                reason="LLM返回格式异常",
            )

        is_violation = data.get("is_violation", False)
        sub_type_str = data.get("violation_sub_type", "none")
        level_str = data.get("violation_level", "none")
        confidence = float(data.get("confidence", 0.5))
        reason = data.get("reason", "")
        violation_sentences = data.get("violation_sentences", [])
        hit_keywords = data.get("hit_keywords", [])
        context_analysis = data.get("context_analysis", "")

        try:
            sub_type = ViolationSubType(sub_type_str)
        except ValueError:
            sub_type = ViolationSubType.NONE
        try:
            level = ViolationLevel(level_str)
        except ValueError:
            level = ViolationLevel.NONE

        # 构建句级命中
        sentence_hits = []
        for vs in violation_sentences:
            idx = -1
            for i, s in enumerate(reported_sentences):
                if vs.strip() in s or s in vs.strip():
                    idx = i
                    break
            sentence_hits.append(SentenceHit(
                sentence=vs, sentence_index=idx,
                is_violation=True, sub_type=sub_type,
                confidence=confidence,
                hit_rules=[
                    HitRule(
                        source=DetectionSource.LLM,
                        matched_text=kw,
                        rule_id=f"llm_{kw}",
                        rule_description=f"LLM语义识别: {kw}",
                        sub_type=sub_type,
                    ) for kw in hit_keywords
                ],
            ))

        full_reason = reason
        if context_analysis:
            full_reason += f"\n上下文分析: {context_analysis}"

        return DetectionResult(
            source=DetectionSource.LLM,
            sentence_hits=sentence_hits,
            is_violation=is_violation,
            violation_level=level,
            violation_sub_type=sub_type,
            confidence=max(0.0, min(1.0, confidence)),
            reason=full_reason,
        )

    async def detect_async(
        self, chat_messages: list[ChatMessage], reported_sentences: list[str],
        reported_id: str, reporter_id: str,
    ) -> DetectionResult:
        system, user = self._build_prompt(chat_messages, reported_id, reporter_id)
        try:
            content = await self._call_llm(system, user)
            return self._parse_response(content, reported_sentences)
        except Exception as e:
            return DetectionResult(
                source=DetectionSource.LLM,
                is_violation=False,
                confidence=0.1,
                reason=f"LLM调用失败: {e}",
            )

    def detect(
        self, chat_messages: list[ChatMessage], reported_sentences: list[str],
        reported_id: str, reporter_id: str,
    ) -> DetectionResult:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, self.detect_async(
                        chat_messages, reported_sentences, reported_id, reporter_id,
                    )).result(timeout=self.timeout + 10)
            return loop.run_until_complete(self.detect_async(
                chat_messages, reported_sentences, reported_id, reporter_id,
            ))
        except RuntimeError:
            return asyncio.run(self.detect_async(
                chat_messages, reported_sentences, reported_id, reporter_id,
            ))
