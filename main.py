"""主入口"""
from src.detector import AbuseDetector
from src.models import ChatMessage

if __name__ == "__main__":
    # 快速验证
    detector = AbuseDetector()

    # 示例: 检测一条对话
    messages = [
        ChatMessage(sender_id="P001", sender_name="玩家A", content="你能不能快点"),
        ChatMessage(sender_id="P002", sender_name="玩家B", content="卧槽，被gank了"),
        ChatMessage(sender_id="P002", sender_name="玩家B", content="我靠，又死了"),
    ]
    result = detector.detect(
        chat_messages=messages,
        reported_id="P002",
        reporter_id="P001",
    )
    import json
    print(json.dumps(json.loads(result.model_dump_json(indent=2)), ensure_ascii=False, indent=2))
