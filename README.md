# Game Chat Abuse Detection System
游戏聊天违规检测判罚系统

## 项目结构
```
├── src/
│   ├── engines/          # 检测引擎
│   │   ├── __init__.py
│   │   ├── keyword_engine.py    # 关键词匹配引擎(AC自动机)
│   │   ├── rule_engine.py       # 规则引擎(正则+模式)
│   │   └── llm_engine.py        # LLM语义判罚引擎
│   ├── models/           # 数据模型
│   │   ├── __init__.py
│   │   └── schemas.py           # 判罚结果数据结构
│   ├── utils/            # 工具
│   │   ├── __init__.py
│   │   └── text_normalizer.py   # 文本标准化(处理谐音/拆字等)
│   └── detector.py      # 三级级联判罚调度器
├── data/
│   ├── keywords/         # 关键词库
│   │   ├── mild.json     # 轻度违规关键词
│   │   ├── moderate.json # 中度违规关键词
│   │   └── severe.json   # 严重违规关键词
│   ├── rules/            # 规则配置
│   │   ├── patterns.json # 正则模式库
│   │   └── evasion.json  # 规避手段规则
│   └── cases/            # 人工判定案例
│       ├── fewshot_examples.json  # few-shot示例
│       └── test_cases.json        # 测试用例
├── config/
│   └── config.yaml       # 全局配置
├── tests/
│   └── test_detector.py  # 单元测试
├── scripts/
│   ├── run_batch.py      # 批处理脚本
│   └── run_interactive.py # 交互式测试
├── requirements.txt
└── main.py               # 入口
```

## 三级检测策略
1. **关键词精确匹配** - AC自动机，高优先级快速判定
2. **规则引擎模式识别** - 正则+规则，处理规避手段
3. **LLM语义判罚** - 大模型兜底，处理复杂语境

## 违规等级
- 轻度言语违规（口头禅违规 / 轻度辱骂违规）
- 中度言语违规（针对性辱骂 / 持续骚扰 / 歧视性言论）
- 严重言语违规（恶毒辱骂 / 敏感违规 / 极端内容）
