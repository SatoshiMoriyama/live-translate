# realtime-transcriber

macOS のシステム音声をリアルタイムで文字起こし・翻訳する CLI ツールです。

BlackHole 2ch 経由でキャプチャした音声を MLX-Whisper で文字起こしし、AWS Translate で日本語に翻訳して表示します。60秒ごとに Amazon Bedrock（Claude Haiku 4.5）で内容を要約し、Whisper の認識精度向上にも活用します。

## アーキテクチャ

```
システム音声 → BlackHole 2ch → sounddevice
  → VAD（Silero VAD）で発話区間を検出
  → MLX-Whisper で文字起こし（要約から生成した英語ヒントで精度向上）
  → ハルシネーション除去（定型フレーズ・繰り返しパターン検出）
  → AWS Translate で英→日翻訳（複数文は並列処理）
  → ターミナルに表示 + ログファイルに記録

  [60秒ごと]
  → Amazon Bedrock（Claude Haiku 4.5）で日本語要約 + 英語キーワード要約を生成
  → 英語キーワード要約を Whisper の initial_prompt に反映
```

## 前提条件

- macOS（Apple Silicon）
- Python 3.11〜3.13
- [uv](https://docs.astral.sh/uv/) （パッケージ管理）
- [BlackHole 2ch](https://existential.audio/blackhole/) （仮想オーディオデバイス）
- macOS の「Audio MIDI 設定」で Multi-Output Device を作成済み
- AWS CLI で SSO ログイン済み（`aws sso login`）
- Amazon Bedrock で Claude Haiku 4.5 のモデルアクセスが有効

## セットアップ

```bash
# 依存パッケージのインストール
uv sync

# AWS SSO ログイン
aws sso login
```

## 使い方

```bash
uv run realtime-transcriber
```

起動すると音声出力先の確認が行われます。Multi-Output Device に切り替えた後、Enter を押すと文字起こしが開始されます。

ログファイルは `logs/` ディレクトリにセッションごとに生成されます。

終了は `Ctrl+C` です。

## 出力例

```
● Recording... 3s
  [00:12] Hope you all had a great time at re:Invent so far!
  [00:12] これまでのre:Inventで素晴らしい時間を過ごしたことを願っています!

  [00:18] Raise your hand if you ever been paged at midnight.
  [00:18] 真夜中にページングされたことがある方は手を挙げてください。

--- 要約 ---
• re:Inventセッション。インシデント対応における「コンテキスト麻痺」の問題を提起
• データは溢れているが文脈がないため、MTTRの改善にはMTTCが重要と主張
---
```

## 主要モジュール

| ファイル | 役割 |
|---------|------|
| `main.py` | CLI エントリポイント。パイプライン全体の制御 |
| `audio.py` | 音声キャプチャと VAD による発話区間検出 |
| `transcriber.py` | MLX-Whisper による文字起こしとハルシネーション除去 |
| `translator.py` | AWS Translate による翻訳 |
| `summarizer.py` | Amazon Bedrock による定期要約（日本語要約 + Whisper用英語ヒント） |
| `session_logger.py` | セッションごとのログファイル管理 |

## テスト

```bash
uv run pytest
```
