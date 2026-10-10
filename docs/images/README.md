# README 截图目录

放 README 里展示用的界面截图。**不要把密钥、真实客户数据或未授权的数据放进截图。**

## 约定

| 项目 | 建议 |
|---|---|
| 文件名 | `demo.png`（主截图）、需要时再加 `demo-2.png`、`demo-3.png` |
| 格式 | PNG |
| 宽度 | 1600–1920 px（GitHub 正文最大宽度约 880 px，这个尺寸在高分屏上清晰） |
| 文件大小 | 单张 < 1 MiB（GitHub 仓库建议保持轻量） |
| 内容 | 界面主体。截图前把窗口拉宽，确保"结构化结论"里的 ✅/🔎/⚠️ 徽章看得清 |

## 截图怎么拍

1. 启动应用：

   ```powershell
   cd "C:\Users\35002\Documents\Codex\2026-10-04\DataPilot-AI"
   .\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py
   ```

2. 左侧选 **使用 Demo 数据**
3. 在 AI Analyst Agent 区提问：**为什么 8 月销售额下降？**
4. 等结果出来后截图（能看到"分析步骤"+"结构化结论"+ 每条结论的 fact/interpretation/hypothesis 徽章最有说服力）
5. 保存为本目录下的 `demo.png`

## 拍完怎么接到 README

README 第一屏已经留好了注释占位。拍完把这一行插入到占位位置即可：

```markdown
![DataPilot AI 演示](docs/images/demo.png)
```

（也可以让项目维护者代做：说一声"截图好了"，一条命令完成编辑、提交、推送。）
