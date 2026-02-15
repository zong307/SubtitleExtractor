# Subtitle Extractor - 音视频字幕提取工具

一个基于AI技术的音视频字幕自动生成工具，支持多种ASR（自动语音识别）引擎，提供图形界面操作，能够从视频或音频文件中提取文字并生成SRT和CSV格式的字幕文件。

## 功能特性

- 🎬 支持多种音视频格式输入（MP4, AVI, MKV, MOV, MP3, WAV等）
- 🤖 多种ASR引擎支持（Whisper, Qwen等）
- 🔊 语音活动检测（VAD）和音频增强
- 👥 可选说话人分离功能（CAM++模型）
- 📄 输出多种格式：SRT字幕文件和CSV数据表格
- 🎨 直观的图形用户界面（PyQt5）
- ⚙️ 丰富的参数配置选项
- 🛡️ 支持处理中断和部分结果保存
- 💻 支持CPU/GPU加速

## 技术架构

本项目采用模块化设计，主要包括以下组件：

- **ASR模块**：自动语音识别，支持Whisper等多种引擎
- **VAD处理器**：语音活动检测，精准定位语音片段
- **音频处理器**：音频提取与增强，提升识别质量
- **说话人分离器**：区分不同说话人（可选功能）
- **字幕生成器**：生成最终的字幕文件
- **GUI界面**：PyQt5图形界面，方便操作

## 安装说明

### 环境要求

- Python 3.8+
- pip包管理器

### 安装步骤

1. 克隆项目：
   ```bash
   git clone <repository-url>
   cd subtitle2
   ```

2. 创建虚拟环境（推荐）：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. 启动应用：
   ```bash
   python -m src.main
   ```

## 使用方法

1. 运行程序后，点击"浏览"按钮选择要处理的音视频文件
2. 配置ASR模型类型、大小和运行设备
3. 设置处理参数（语言、VAD阈值等）
4. 选择输出文件路径（CSV和SRT格式）
5. 点击"开始处理"按钮进行字幕提取
6. 查看处理进度和日志输出

## 参数说明

- **模型类型**：选择ASR引擎（如Whisper）
- **模型规格**：选择模型大小（影响准确度和速度）
- **计算设备**：选择CPU或GPU进行处理
- **目标语言**：指定音频的主要语言
- **VAD阈值**：语音活动检测的敏感度
- **静音结束延迟**：检测到静音后等待的时间
- **片段前后扩充**：对语音片段的前后扩展时间
- **单条字幕上限**：每条字幕的最大字符数

## Docker部署（可选）

项目提供了Dockerfile和docker-compose.yml文件，可以使用Docker进行部署：

```bash
# 构建镜像
docker-compose build

# 运行容器
docker-compose up -d
```

## 开发说明

项目遵循模块化设计原则，主要目录结构如下：

```
src/
├── asr/           # ASR引擎实现
├── config/        # 配置管理
├── core/          # 核心处理流程
├── models/        # 数据模型定义
├── ui/            # 用户界面
├── utils/         # 工具函数
└── main.py        # 应用入口
```

## 贡献指南

欢迎提交Issue和Pull Request来改进此项目。

## 许可证

请参阅LICENSE文件获取详细信息。

## 致谢

- OpenAI Whisper - 语音识别模型
- PyQt5 - 图形界面框架
- Silero VAD - 语音活动检测