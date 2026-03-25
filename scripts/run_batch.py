"""
批处理脚本 - 支持两种模式:
1. 从 JSON 文件加载案例
2. 从示例集自动生成测试用例并验证准确率
"""
from __future__ import annotations
import json
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import AbuseDetector
from src.models import ChatMessage, ViolationLevel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_cases(input_file: str) -> list[dict]:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    if isinstance(data, list):
        return data
    raise ValueError(f"不支持的输入格式: {type(data)}")


def examples_to_test_cases(examples_file: str) -> list[dict]:
    """将示例集转换为测试用例格式"""
    with open(examples_file, "r", encoding="utf-8") as f:
        examples = json.load(f)

    cases = []
    for i, ex in enumerate(examples):
        messages_raw = ex.get("messages", [])
        chat_messages = []
        for msg in messages_raw:
            role = msg.get("role", "unknown")
            sender_id = "reported" if role == "被举报者" else "reporter"
            sender_name = role
            chat_messages.append(ChatMessage(
                sender_id=sender_id,
                sender_name=sender_name,
                content=msg.get("content", ""),
            ))

        # 判定标准
        judgment = ex.get("judgment", "")
        level = ex.get("level", "none")
        level_label = ex.get("level_label", "")

        cases.append({
            "case_id": f"EXAMPLE_{i+1:03d}",
            "chat_messages": chat_messages,
            "reported_id": "reported",
            "reporter_id": "reporter",
            "expected_level": level,
            "expected_label": level_label,
            "expected_judgment": judgment,
            "note": ex.get("note", ""),
        })

    return cases


def evaluate(results: list, cases: list) -> dict:
    """评估准确率"""
    total = len(results)
    correct = 0
    details = []

    for r, c in zip(results, cases):
        expected = c.get("expected_level", "none")
        actual_level = r.violation_level.value

        # 判断是否正确
        if expected == "none":
            is_correct = not r.is_violation
        elif expected in ("mild_swearword", "mild_insult"):
            is_correct = actual_level == "mild"
        elif expected == "moderate":
            is_correct = actual_level == "moderate"
        elif expected == "severe":
            is_correct = actual_level == "severe"
        else:
            is_correct = actual_level == expected

        if is_correct:
            correct += 1

        details.append({
            "case_id": c["case_id"],
            "expected": c.get("expected_label", expected),
            "actual": f"{r.violation_level.label_cn}" + (f"-{r.violation_sub_type_label}" if r.violation_sub_type_label else ""),
            "correct": is_correct,
            "reason": r.reason[:80] if r.reason else "",
            "needs_review": r.needs_manual_review,
        })

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "details": details,
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "examples"

    if mode == "examples":
        # 从示例集生成测试用例并验证
        examples_file = sys.argv[2] if len(sys.argv) > 2 else "data/cases/all_examples.json"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "eval_results.json"
        logger.info(f"从示例集加载测试用例: {examples_file}")
        cases = examples_to_test_cases(examples_file)
        logger.info(f"共 {len(cases)} 个测试案例")
    elif mode == "json":
        input_file = sys.argv[2] if len(sys.argv) > 2 else "data/cases/test_cases.json"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "results.json"
        logger.info(f"加载案例: {input_file}")
        cases = load_cases(input_file)
    else:
        print("用法:")
        print("  python run_batch.py examples [示例文件] [输出文件]")
        print("  python run_batch.py json [输入文件] [输出文件]")
        sys.exit(1)

    detector = AbuseDetector()
    logger.info("开始检测...")
    results = detector.detect_batch(cases)

    if mode == "examples":
        # 评估准确率
        eval_result = evaluate(results, cases)
        output_data = eval_result
        logger.info(f"\n{'='*60}")
        logger.info(f"评估结果: {eval_result['correct']}/{eval_result['total']} 准确率={eval_result['accuracy']:.1%}")
        logger.info(f"{'='*60}")

        # 打印错误案例
        errors = [d for d in eval_result["details"] if not d["correct"]]
        if errors:
            logger.info(f"\n错误案例 ({len(errors)}个):")
            for e in errors:
                logger.info(f"  {e['case_id']}: 期望={e['expected']}, 实际={e['actual']}")
                logger.info(f"    理由: {e['reason']}")

        # 打印需复核案例
        reviews = [d for d in eval_result["details"] if d.get("needs_review")]
        if reviews:
            logger.info(f"\n需人工复核 ({len(reviews)}个):")
            for e in reviews:
                logger.info(f"  {e['case_id']}: {e['actual']} | {e['reason']}")
    else:
        output_data = [r.model_dump(mode="json") for r in results]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
