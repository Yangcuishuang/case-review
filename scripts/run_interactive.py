"""
交互式测试脚本 - 手动输入对话进行实时检测
"""
from __future__ import annotations
import json
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import AbuseDetector
from src.models import ChatMessage

logging.basicConfig(level=logging.WARNING)


def print_result(result):
    """格式化打印检测结果"""
    print("\n" + "-" * 60)
    if result.is_violation:
        print(f"  判定: 违规")
        print(f"  等级: {result.violation_level.label_cn}")
        print(f"  大类: {result.violation_category}")
    else:
        print(f"  判定: 无违规")
    print(f"  置信度: {result.confidence:.2f}")
    print(f"  理由: {result.reason}")
    if result.hit_rules:
        print(f"  命中规则:")
        for hit in result.hit_rules:
            print(f"    - [{hit.source.value}] \"{hit.matched_text}\" -> {hit.rule_description}")
    if result.needs_manual_review:
        print(f"  ⚠ 需要人工复核: {result.manual_review_reason}")
    print(f"  检测链路: {' -> '.join(r.source.value for r in result.detection_chain) if result.detection_chain else '无'}")
    print("-" * 60)


def interactive_mode(detector: AbuseDetector):
    """交互式检测模式"""
    print("=== 游戏聊天违规检测 - 交互模式 ===")
    print("输入对话格式: [发送者ID]:[发送者名称] 消息内容")
    print("输入 'done' 结束对话输入，开始检测")
    print("输入 'quit' 退出\n")

    reported_id = input("被举报玩家ID: ").strip()
    reporter_id = input("举报玩家ID: ").strip()

    messages = []
    print("\n--- 请输入对话记录 ---")
    while True:
        line = input().strip()
        if line.lower() == "done":
            break
        if line.lower() == "quit":
            return

        # 解析格式: ID:Name 内容
        if ":" not in line:
            print("  格式错误，请使用 ID:Name 内容")
            continue
        
        sep_idx = line.index(":")
        player_id = line[:sep_idx].strip()
        rest = line[sep_idx + 1:].strip()
        
        # 支持两种格式: "ID:Name 内容" 或 "ID: 内容"
        if ":" in rest:
            name_sep = rest.index(":")
            player_name = rest[:name_sep].strip()
            content = rest[name_sep + 1:].strip()
        else:
            player_name = player_id
            content = rest

        messages.append(ChatMessage(
            sender_id=player_id,
            sender_name=player_name,
            content=content,
        ))

    if not messages:
        print("未输入任何对话")
        return

    print(f"\n共 {len(messages)} 条消息，开始检测...")
    result = detector.detect(
        chat_messages=messages,
        reported_id=reported_id,
        reporter_id=reporter_id,
    )
    print_result(result)


def json_mode(detector: AbuseDetector, json_file: str):
    """从JSON文件读取案例并检测"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cases = data if isinstance(data, list) else data.get("cases", [data])
    results = detector.detect_batch(cases)
    
    for r in results:
        print_result(r)


def main():
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        detector = AbuseDetector()
        json_mode(detector, json_file)
    else:
        detector = AbuseDetector()
        interactive_mode(detector)


if __name__ == "__main__":
    main()
