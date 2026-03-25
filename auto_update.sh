#!/bin/bash
# 言语违规判罚RLHF平台 - 一键更新脚本

echo "🚀 开始更新网页..."

# 进入项目目录
cd "$(dirname "$0")"

# 添加所有更改
git add .

# 提交更改
read -p "请输入更新说明（如：更新案例数据）: " commit_msg
git commit -m "$commit_msg"

# 推送到 GitHub
git push origin main

echo "✅ 更新完成！"
echo "🌐 请访问: https://yangcuishuang.github.io/case-review/"
echo "⏳ 等待 1-2 分钟后生效"
