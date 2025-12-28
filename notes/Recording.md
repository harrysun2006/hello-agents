下面给你整理一份 **适合 macOS 的录屏 + 后期剪辑工具推荐**，覆盖从 **简单录制、细粒度控制、音轨分离、外挂字幕到 TTS 音轨替换/合成** 的全流程。

---

# 📌 一、录屏 + 音频录制（macOS 本地 & 在线）

### 🖥 1. **屏幕录制（本地）**

#### ✅ **QuickTime Player（系统自带，免费）**

✔ 简单易用
✔ 支持录屏 + 录本机/麦克风音频
❌ 不支持分轨输出（合在一起）

**适合**：快速、轻度录制

---

#### ✅ **OBS Studio（免费 & 多轨）**

[https://obsproject.com/](https://obsproject.com/)

✔ 支持屏幕/窗口录制
✔ 支持多音频来源（麦克 / 系统声音 / 音乐）分轨录制
✔ 支持场景切换
✔ 输出为独立音轨（可选）

**适合**：教学场景（画面 + 讲解 + 系统音）

💡 Tips：

* 在 `Settings → Audio` 里可以设置不同的音频设备
* 在 `Advanced → Recording` 里勾选“音频轨道”并输出多轨

---

### 🟡 2. **在线录屏工具（简单快速）**

#### 📌 **Loom**

[https://www.loom.com](https://www.loom.com)

✔ 截屏 + 摄像头
✔ 自动上传
✔ 支持注释
❌ 不太适合后期复杂剪辑

---

#### 📌 **Screencastify**

Chrome 插件录屏
✔ 简单
✔ 自动云存储
❌ 需要付费解锁分轨等高级

---

## 📌 核心建议

* **想要分轨录制 = 强烈推荐 OBS Studio**
* **想要快速简单录制 = 用 QuickTime**

---

# 🎬 二、后期剪辑 & 音轨处理（本地）

### 🧠 1. **DaVinci Resolve（免费 & 专业）**

[https://www.blackmagicdesign.com/products/davinciresolve/](https://www.blackmagicdesign.com/products/davinciresolve/)

✔ 支持 **多音轨（Video + Audio）**
✔ 支持 **剪辑 + 调色 + 音频混合**
✔ 支持字幕 & TTS 外挂字幕导入
✔ 无水印免费版

**特色**：

* Fairlight 音频模块可做专业音频混剪
* 支持高精度定点剪辑

📌 **学习曲线较陡**，但功能最强

---

### ✂ 2. **CapCut（本地/移动/桌面）**

[https://www.capcut.com](https://www.capcut.com)

✔ 简易剪辑
✔ 支持音轨分离
✔ 支持外挂字幕 / 自动字幕生成
✔ TTS 内置（特定版本）

**适合**：内容创作者快速剪辑

---

### 🧑‍🎤 3. **Final Cut Pro（付费 Apple 生态）**

[https://www.apple.com/final-cut-pro/](https://www.apple.com/final-cut-pro/)

✔ 专业剪辑
✔ 音轨分离
✔ 支持语音转字幕与外挂字幕
✔ 支持插件扩展

---

### 🪄 4. **ScreenFlow**

[https://www.telestream.net/screenflow/](https://www.telestream.net/screenflow/)

✔ macOS 专用录屏 + 剪辑
✔ 多轨视频 + 音频
✔ 支持动画与字幕
✔ 内置简单 TTS

---

# 🔈 三、**音频分轨 & TTS 处理**

---

## 🔹 1. **音频分离 / 编辑软件**

### 🎧 **Audacity（免费）**

[https://www.audacityteam.org/](https://www.audacityteam.org/)

✔ 打开录制的音频
✔ 分轨清理噪音
✔ 轻量剪辑

---

### 🎹 **Adobe Audition（付费）**

✔ 多轨音频剪辑
✔ 降噪 / 修复 / 混响 / 音色调整

---

## 🔊 2. **TTS（语音合成）**

如果你要**重新生成解说音轨**：

### 🌐 在线 TTS

* **ElevenLabs** — 自然语音最好
* **Google Cloud TTS** / **Azure Cognitive TTS** — 强大云服务
* **Speechify** / **Play.ht** — 方便直接下载

输出 WAV/MP3，然后导入剪辑工具即可。

---

# 🅰 字幕（可选外挂字幕）

---

## 📌 自动字幕生成（快速）

### 🟡 **CapCut 自动字幕**

✔ 支持中英自动识别
✔ 可编辑 / 区段导出 SRT

---

### 🟡 **Descript**

[https://www.descript.com/](https://www.descript.com/)

✔ 自动字幕
✔ 支持 Speaker ID
✔ TTS Overdub（可以用 AI 生成声音替换原声）

🟢 特别强大，适合教学剪辑

---

## 📌 在线字幕服务

* **Veed.io**
* **Happy Scribe**
* **Otter.ai**

可以生成 SRT/ASS 然后导入剪辑软件

---

## 📌 手动字幕 / SRT 编辑

像这样：

```
1
00:00:05,000 --> 00:00:08,000
这是一条字幕

2
00:00:08,500 --> 00:00:12,000
下一条字幕
```

最终导入 DaVinci / CapCut / FCP

---

# 🧩 四、完整推荐路径（按你描述需求）

---

## 👉 **方案 A：专业输出（最强剪辑能力 + 分轨 + 字幕）**

📌 OBS（录制）
📌 DaVinci Resolve（剪辑 + 分轨 + 字幕）
📌 在线 TTS（ElevenLabs）生成新音轨
📌 SRT 字幕导出/嵌入

✅ 最干净的流程
⚠️ 学习曲线中等

---

## 👉 **方案 B：极速输出（适合网课发布）**

📌 OBS（录制分轨）
📌 CapCut（桌面/移动）剪辑
📌 CapCut 自动字幕
📌 若需要更自然语音 → 在线 TTS
📌 输出 MP4 + SRT

✅ 操作简单
⚠️ CapCut 字幕需校对

---

## 👉 **方案 C：轻量级 + 本地快速**

📌 QuickTime（录制）
📌 Audacity（音频清理 + 分轨）
📌 DaVinci Resolve（剪辑 + 字幕）

⚡ 对资源较强的 Mac 非常适合

---

# 🛠 五、常见问题 & 小技巧

---

## 🎤 **如何录系统声音 + 麦克风？**

macOS 默认不支持系统音录制，需要：

🟦 **BlackHole / iShowU Audio Capture / Loopback**
安装并在 OBS 里设置音频设备

---

## 🧹 **降噪技巧**

* Audacity 降噪效果好
* OBS 自带 noise suppression（NS）
* DaVinci Fairlight 降噪更专业

---

## 🎯 **分轨录制是关键**

如果你要后期换音频（TTS）：

**一定要分轨，不要把麦克风跟系统音录到同一轨**

OBS 设置 → Output → Recording → Audio Tracks

---

## 🌐 **字幕同步技巧**

* 先生成文本稿（可导出）
* 再导出 SRT
* 最后校对时间戳

---

## 📦 **推荐组合模板**

| 需求   | 工具组合                       |
| ---- | -------------------------- |
| 极速录课 | OBS + CapCut               |
| 详细剪辑 | OBS + DaVinci Resolve      |
| 语音替换 | OBS + DaVinci + ElevenLabs |
| 自动字幕 | CapCut / Descript          |

---

如果你愿意，我还可以给你：

👉 不同预算的推荐清单（免费 / 付费）
👉 安装配置指南（比如 macOS 上 OBS + BlackHole）
👉 模版剪辑流程
👉 一套“录课标准工作流”自动化脚本/CheckList

需要哪个版本？
