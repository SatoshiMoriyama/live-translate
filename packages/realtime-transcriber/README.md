# realtime-transcriber

macOS のシステム音声をリアルタイムで文字起こし・翻訳する CLI ツールです。

このツールはシステム音声を直接キャプチャするため、YouTube Live・ウェビナー・Zoom・Teams など任意のアプリの英語音声を、プラットフォームや配信者の設定に関係なくリアルタイムで日本語に翻訳できます。

BlackHole 2ch 経由でキャプチャした音声を MLX-Whisper（`mlx-community/whisper-large-v3-turbo-q4`）で文字起こしし、Amazon Bedrock（デフォルト: Amazon Nova Pro）で日本語に翻訳して表示します。60秒ごとに Amazon Bedrock（Claude Haiku 4.5）で内容を要約し、Whisper の認識精度向上にも活用します。

## アーキテクチャ

```
システム音声 → BlackHole 2ch → sounddevice
  → VAD（Silero VAD）で発話区間を検出
  → MLX-Whisper で文字起こし（要約から生成した英語ヒントで精度向上）
  → ハルシネーション除去（定型フレーズ・繰り返しパターン検出）
  → Amazon Bedrock（Nova Pro）で英→日翻訳（複数文は並列処理、バックエンド切替可能）
  → ターミナルに表示 + ログファイルに記録

  [60秒ごと]
  → Amazon Bedrock（Claude Haiku 4.5）で日本語要約 + 英語キーワード要約を生成
  → 英語キーワード要約を Whisper の initial_prompt に反映
```

## 必要なディスク容量

- MLX-Whisper モデル（`whisper-large-v3-turbo-q4`）: 約 400 MB（初回起動時に `~/.cache/huggingface/hub/` へ自動ダウンロード）
- Python 依存パッケージ: 数百 MB

## 前提条件

- macOS（Apple Silicon）
- Python 3.11〜3.13
- [uv](https://docs.astral.sh/uv/) （パッケージ管理）
- [BlackHole 2ch](https://existential.audio/blackhole/) （仮想オーディオデバイス）
- macOS の「Audio MIDI 設定」で Multi-Output Device（複数出力装置） を作成済み（後述）
- AWS CLI で SSO ログイン済み（後述）
- Amazon Bedrock で以下のモデルアクセスが有効（us-east-1 リージョン）
  - Amazon Nova Pro（翻訳用、デフォルト）
  - Claude Haiku 4.5（要約用）

## セットアップ

### 1. Multi-Output Device（複数出力装置） の作成

BlackHole 2ch をインストールしたら、macOS の「Audio MIDI 設定」で Multi-Output Device（複数出力装置） を作成します。

1. 「Audio MIDI 設定」を開く（Spotlight で「Audio MIDI」と検索）
2. 左下の「+」ボタン →「複数出力装置を作成」
3. 「BlackHole 2ch」と通常のスピーカー（例: MacBook Pro のスピーカー）の両方にチェックを入れる

これにより、音声がスピーカーから聞こえると同時に BlackHole 経由でアプリにもキャプチャされます。

### 2. 依存パッケージのインストール

```bash
uv sync
```

依存パッケージのバージョンは `uv.lock` で固定されているため、再現性のある環境が構築されます。

### 3. AWS の設定

Bedrock のモデルアクセスは us-east-1 リージョンで有効化してください（クロスリージョン推論プロファイルを使用するため）。

```bash
aws sso login
```

`~/.aws/config` のデフォルトプロファイルに Bedrock へのアクセス権限を持つ SSO 設定を入れておいてください。

## 使い方

```bash
uv run realtime-transcriber
```

デフォルト以外のプロファイルを使う場合は `--profile` オプションで指定できます。

```bash
uv run realtime-transcriber --profile your-profile-name
```

起動すると音声出力先の確認が行われます。Multi-Output Device（複数出力装置） に切り替えた後、Enter を押すと文字起こしが開始されます。

ログファイルは `logs/` ディレクトリにセッションごとに生成されます。

終了は `Ctrl+C` です。

## 出力例

```
● Recording... 5s ▼
  [00:21] We're driven by the idea that the products and services we create
          should help people unleash their creativity and potential.
  [00:21] 私たちは、私たちが作る製品やサービスが人々の創造性と可能性を
          引き出す手助けをするべきだという考えに導かれています。

  [00:27] We build some of the largest internet services on the planet
          and many of them run on AWS.
  [00:27] 私たちは世界最大級のインターネットサービスの一部を構築しており、
          それらの多くはAWS上で稼働しています。

--- 要約 ---
ペイアム・ウラシディ氏が登壇し、Appleのクラウドインフラストラクチャ戦略に
ついて説明しました。同氏のチームはApp Store、Apple Music、Apple TV、
Podcastsなど、数十億人が利用する主要サービスを開発・運営しており、これらは
AWSと自社データセンターの両方で稼働しています。
---
```

## 処理の仕組み

### 音声キャプチャと VAD（発話区間検出）

BlackHole 2ch 経由でシステム音声をステレオで取得し、モノラルに変換して Silero VAD に渡します。VAD が発話区間を検出すると音声バッファへの蓄積を開始し、以下の条件で区切ります。

- 無音が一定時間続いた時点で発話終了と判定（初期値 500ms、前回の翻訳結果の文数に応じて 200ms〜800ms の範囲で動的に調整）
- 発話が 30 秒に達したら強制的に区切り
- 1 秒未満の発話はノイズとして無視

### 文の結合と分割

Whisper の出力が文末（`.` `!` `?` `;`）で終わっていない場合、音声を次のチャンクと結合して再処理します。これにより文の途中で切られることを防ぎます。ただし 15 秒以上蓄積しても文が完結しない場合は、そのまま翻訳に回します（最大で 30 + 15 = 45 秒分の音声がつながる可能性があります）。

文が完結したら、略語（Mr. Dr. e.g. など）のピリオドで誤分割しないよう保護しつつ、文単位に分割して並列翻訳します。

### Whisper のコンテキスト引き継ぎ

Whisper の `initial_prompt` に以下を渡して認識精度を向上させています。

- 直近の文字起こし結果（最大 200 文字、文単位で切り出し）
- 60 秒ごとの要約から生成された英語キーワード（最大 400 文字）

これにより、セッション固有の専門用語や固有名詞が文字起こしに反映されやすくなります。

### ハルシネーション除去

Whisper が無音や短い音声に対して出力する定型フレーズ（"Thank you." "Bye." など）をフィルタリングします。加えて以下のパターンも検出します。

- 同じ単語の 5 回以上の連続（例: "too too too too too"）
- 同じ文字の 10 回以上の連続（例: "llllllllll"）
- 同じ 2〜6 文字パターンの 5 回以上の繰り返し（例: "結論の結論の結論の..."）
- 英数字がほぼ含まれないテキスト

繰り返しが検出された場合、除去後のテキストが元の 50% 未満ならハルシネーションとして破棄し、50% 以上残る場合は繰り返し部分のみ除去して翻訳に回します。

### 定期要約（60 秒ごと）

バックグラウンドのデーモンスレッドが 60 秒ごとに Amazon Bedrock（Claude Haiku 4.5）を呼び出し、前回の要約と直近の翻訳テキストから更新された要約を生成します。1 回の API 呼び出しで以下の 2 つを同時に生成します。

- 日本語要約（ターミナル表示・ログ記録用）: 英語圏特有の表現や略語を補足した自然な文章
- 英語キーワード要約（Whisper の `initial_prompt` 用）: セッションのトピック、専門用語、話者名などを 400 文字以内で記述

### ステータス表示

ターミナルにはリアルタイムで処理状態が表示されます。

- `● Recording... Ns` — VAD が発話を検出し、音声を蓄積中（N は秒数）。無音閾値が変更された場合は ▼（短縮）/ ▲（延長）が表示される
- `⏳ Transcribing...` — Whisper が音声を文字起こし中
- `⏳ Translating...` — 翻訳中
- `... waiting` — 文が未完結のため、次のチャンクを待機中

## 翻訳バックエンドの切り替え

`translator.py` の `TRANSLATION_BACKEND` を変更することで翻訳エンジンを切り替えられます。

```python
# Bedrock（デフォルト）— 文脈を考慮した自然な日本語
TRANSLATION_BACKEND = "bedrock"

# AWS Translate — 高速・低コスト。直訳傾向あり
TRANSLATION_BACKEND = "aws_translate"
```

Bedrock の場合、`BEDROCK_MODEL_ID` でモデルも変更できます。

```python
# Amazon Nova Pro（デフォルト、高品質・高速・低コスト）
BEDROCK_MODEL_ID = "us.amazon.nova-pro-v1:0"

# Claude Haiku 4.5（高品質、Nova Proより高価）
BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Amazon Nova Lite（高速、低コスト、品質はやや劣る）
BEDROCK_MODEL_ID = "us.amazon.nova-lite-v1:0"

# Amazon Nova Micro（最速、最安、短文向き）
BEDROCK_MODEL_ID = "us.amazon.nova-micro-v1:0"
```

### 翻訳品質の比較

同じ英語セッション（Apple の re:Invent 登壇）に対して実際に出力された翻訳の比較です。

| 英語原文 | AWS Translate | Nova Lite | Nova Pro（デフォルト） | Haiku 4.5 |
|---------|--------------|-----------|---------------------|-----------|
| power everything from Apple Silicon | **電力を供給** | **動かして** | **動かすために使用** | **動かして** |
| we're a Java shop | **Javaショップ** | **Javaショップ** | **Javaを主に使用** | **Javaを使う企業** |
| can raise eyebrows | 眉をひそめる | 眉をひそめさせる（誤解釈あり） | **注目を集める可能性** | 眉をひそめられる |
| a win for the planet | **地球**にとってもメリット | **地球**にとってより小さい… | お客様と**地球**にとってwin-win | **プラネット**にとっても… |
| homomorphic encryption | ― | 同型暗号化 | **ホモモルフィック暗号化**（補足付き） | 準同型暗号化 |

Bedrock はプロンプトで「日本語話者にわかりやすく、技術用語は英語のまま」と指示しているため、AWS, API, Swift, Graviton などは英語のまま出力します。

### モデル選定の指針

1時間あたりの翻訳コストは、~300回の呼び出し（1回あたり入力 ~150 トークン・出力 ~100 トークン）を想定した目安です。

| モデル | 翻訳品質 | 速度 | 入力 / 100万トークン | 出力 / 100万トークン | 翻訳コスト / 時間 |
|--------|---------|------|---------------------|---------------------|------------------|
| Nova 2 Lite（デフォルト） | Nova Pro と同等以上 | 速い | $0.30 | $2.50 | ~$0.09 |
| Nova Pro | Haiku 4.5 と同等 | 速い | $0.80 | $3.20 | ~$0.13 |
| Haiku 4.5 | 高品質 | 普通 | $1.00 | $5.00 | ~$0.20 |
| AWS Translate | 直訳傾向（多義語に弱い） | 最速 | $15.00 / 100万文字 | — | ~$0.08〜$0.23 |

Nova 2 Lite は Nova Pro と同等の翻訳品質・速度でコストが約 1/3 のため、デフォルトとしています。実測では短文 600〜700ms、中文 700〜850ms 程度のレスポンスタイムで、リアルタイム翻訳に十分な速度です。なお、Nova 2 Lite の推論（thinking）モードは TTFT が大幅に増加するため無効にしています。

## 主要モジュール

| ファイル | 役割 |
|---------|------|
| `main.py` | CLI エントリポイント。パイプライン全体の制御 |
| `audio.py` | 音声キャプチャと VAD による発話区間検出 |
| `transcriber.py` | MLX-Whisper による文字起こしとハルシネーション除去 |
| `translator.py` | Bedrock または AWS Translate による翻訳（切り替え可能） |
| `summarizer.py` | Amazon Bedrock による定期要約（日本語要約 + Whisper用英語ヒント） |
| `session_logger.py` | セッションごとのログファイル管理 |

## 料金の目安

文字起こし（MLX-Whisper）と VAD（Silero VAD）はローカル実行のため無料です。
AWS の有料サービスは翻訳バックエンドの選択によって異なります。

### Bedrock（Amazon Nova Pro）翻訳 — デフォルト

- 入力: $0.80 / 100万トークン、出力: $3.20 / 100万トークン
- 翻訳: 1回あたり入力 ~150 トークン・出力 ~100 トークン。1時間（~300回）で約 $0.13
- 要約（Haiku 4.5）: 60秒ごと、1回あたり入力 ~600 トークン・出力 ~200 トークン。1時間（60回）で約 $0.07
- 合計目安: 約 **$0.20 / 時間**

### Bedrock（Claude Haiku 4.5）翻訳

- 入力: $1.00 / 100万トークン、出力: $5.00 / 100万トークン
- 翻訳 + 要約で合計目安: 約 **$0.30 / 時間**

### AWS Translate 翻訳

- $15.00 / 100万文字
- 1時間のセッションで約 5,000〜15,000 文字を翻訳すると仮定すると約 $0.08〜$0.23
- Bedrock 要約（$0.07）と合わせて合計目安: 約 **$0.15〜$0.30 / 時間**

## テスト

```bash
uv run pytest
```

## 既知の制限事項

- 英語→日本語のみ対応（他の言語ペアは未検証）
- macOS（Apple Silicon）専用。Linux / Windows では動作しない
- 仮想オーディオデバイスは BlackHole 2ch のみ検証済み（Soundflower 等は未検証）
- Whisper の認識精度はスピーカーの発音、音質、背景ノイズに依存する
- 話者名の認識は不安定（セッションごとに `KNOWN_SPEAKERS` を設定すると改善）

## トラブルシューティング

### 「Device 'BlackHole 2ch' not found」と表示される

BlackHole 2ch がインストールされていないか、認識されていません。[BlackHole](https://existential.audio/blackhole/) をインストールし、「Audio MIDI 設定」で表示されることを確認してください。

### 音声が取れない / Recording... が表示されない

以下を確認してください。

1. macOS のシステム出力が Multi-Output Device（複数出力装置） に切り替わっているか（「システム設定 → サウンド → 出力」で確認）
2. ターミナルアプリにマイクのアクセス許可があるか（「システム設定 → プライバシーとセキュリティ → マイク」で Terminal.app や iTerm2 等を許可）

BlackHole はシステム音声をキャプチャしますが、macOS はこれを「マイク入力」として扱うため、アプリにマイク許可が必要です。

### Bedrock で AccessDeniedException が発生する

以下を確認してください。

1. `--profile` オプションまたは `AWS_PROFILE` 環境変数が正しく設定されているか
2. `aws sso login` でログイン済みか
3. us-east-1 リージョンで Amazon Nova Pro と Claude Haiku 4.5 のモデルアクセスが有効か（[Bedrock コンソール](https://console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess) で確認）

### Translation failed が繰り返し表示される

AWS SSO のセッションが期限切れの可能性があります。`aws sso login` を再実行してください。

## 技術的な補足

### silero-vad-lite について

VAD（Voice Activity Detection: 音声区間検出）には [silero-vad-lite](https://github.com/snakers4/silero-vad) を使用しています。これは Silero VAD の軽量版で、PyTorch に依存せず ONNX Runtime で動作するため、インストールサイズが小さく起動も高速です。精度は通常版の Silero VAD と同等です。

## ライセンス

MIT License
