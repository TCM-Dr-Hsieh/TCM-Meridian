# TCM-Meridian

TCM-Meridian，中文名「杏林經緯」，是一套以 NiceGUI 建立的中醫診間 AI 工作台。它整合患者資料管理、就診病歷編輯、多模態患者檔案讀取、ReAct 主 Agent、多個安全與問診 Subagent，以及可追溯的教授 RAG 諮詢流程。

本專案的定位是「醫師主導的臨床輔助系統」。AI 可以協助問診、更新病歷、檢查不確定性、諮詢知識庫並留下行為軌跡，但最終診斷、處方與處置仍必須由合格醫師決定。

## 重要聲明

本系統僅供臨床決策支援與文件輔助，不是醫療器材，不取代醫師診斷、處方或治療決策，也不應用於無人監督的自動看診流程。非醫療專業人員不應將本系統輸出作為自行診斷、治療或用藥依據；若自行使用，相關風險由使用者自行承擔。

公開或部署前請務必檢查：

- `config.json` 可能含有真實 API key。
- `patient_data/` 可能含有患者身分資料、病歷、圖片與臨床 log。
- `professor_*/doc/` 內的知識庫文件可能含有私有或受版權保護的內容（向量索引 `chroma_doc_index/`、`parent_map.jsonl` 已在 `.gitignore` 排除，不會上傳）。
- 本 repo 已提供 `.gitignore`（排除 `config.json` 與真實患者資料）與 `config.example.json`（不含 key 的範本）。請複製 `config.example.json` 為 `config.json` 後填入自己的 API key；首次 `git add` 後務必以 `git status` 確認 `config.json` 與真實患者資料未被追蹤。

## 功能特色

- 患者資料管理：以患者資料夾與 `patient_info.json` 保存資料。
- 就診 Session 管理：依日期建立 `NOTE.md` 與 `ASSESSMENT & TREATMENT.md`。
- 三欄式看診工作台：患者/session 導覽、病歷瀏覽與編輯、主 Agent 對話。
- 日期摘要索引：就診日期下拉顯示 `YYYY-MM-DD（NOTE 摘要）`，實際選取值仍為日期；套用模板下拉維持純日期。
- 病歷版本追蹤：人工改動會加上 `[人類醫師_手動修改]` 來源標籤；主 Agent 同一主輪內每次成功 `update_record` 也會各自留下 UI snapshot，方便 diff 檢視退稿修稿歷程；若 NOTE 或 A&T 某欄位在相鄰版本未變，diff 會顯示「無差異」。
- Snapshot 版本控制：支援 undo、redo、版本標示與 diff 檢視。
- 主 Agent ReAct loop：以嚴格 JSON action contract 協調工具與 Subagent。
- 病歷登載 Subagent：以行級操作更新 NOTE 或 A&T。
- 行級登載保護：Record Subagent 使用原始行號整批驗證操作，任一非法操作會整批退回並回報原因。
- 幻覺審查 Subagent：寫入前檢查矛盾、無中生有、來源錯置、過度推論與重大遺漏，採 **fail-closed**——審查未通過、達上限或審查器服務異常時病歷不寫入（不會假成功）。
- 低信心標註 Subagent：寫入後標註 NOTE 中證據薄弱或需確認的片段。
- 研究對照組模式：幻覺審查與低信心標註的「檢測強度」可設為 `0`，代表不檢查直接放行，並在步驟結果與行為紀錄明確標示，供研究做對照組。
- 病歷檢查員 Subagent：對照 `Record_Template.txt` 檢查 NOTE 完整性。
- 問診助理 Subagent：支援多回合問診，完成後自動恢復主 Agent 流程；未設定問診模型時優雅降級、不會卡在等待狀態。
- Main Agent 子模型：歷史病歷摘要/檢查、摘要並退出可各自設定獨立 API 端點與模型（URL/Key/Model/Max Tokens/Temperature），留空則沿用 Main Agent。
- 患者檔案工具：列出與讀取 `Picture_Row/` 圖片、`Medical_information/` 文字檔，以及患者根目錄的歷史病歷 Markdown。
- 病歷索引摘要：主介面「摘要並退出」會將當日 NOTE 與 A&T 各濃縮成 50 字內摘要，寫入 `patient_info.json` 供日期下拉與主 Agent 檔案清單辨識。
- 教授 RAG：包含 query expansion、prefix classification、dense retrieval、RRF、parent mapping、LLM rerank 與回答生成。
- 智能體互動行為：以 JSONL 記錄多 Agent 輸入、輸出、工具呼叫、RAG 檢索與錯誤，並在 UI 中呈現時間線。
- Session 狀態保存：保存聊天、問診、討論區、歷史摘要、RAG trace 與行為 log。
- 重要覆寫檔原子寫入：`patient_info.json`、空白 NOTE/A&T 建立、NOTE/A&T 一般保存、config、chat/interview/forum state、歷史摘要、對話紀錄與病歷模板會先寫暫存檔再置換，降低崩潰造成半截檔案的風險。
- 忙碌狀態保護：集中式 `SessionBusyGuard` 會在編輯模式、主 Agent 執行中、問診中或 session transition 期間阻止危險操作（含病歷 undo/redo），並鎖定患者/session 導覽；長任務（摘要並退出、產生歷史摘要）期間左欄與右欄送出鈕一併視覺鎖定，避免狀態混亂、錯 session 寫回或草稿遺失。
- 全域設定鎖定：標準病歷模板、模型設定、教授設定屬全域資源，只能在無患者載入時變更，避免執行中 Agent 的快取或記憶與磁碟設定脫鉤。
- 合作式中斷：主 Agent 與問診流程可停止並保留已完成的安全狀態。

## 系統架構

```text
NiceGUI UI
  TCM_Meridian_main.py
  ui_app/controllers/*
  ui_app/services/*

Agent Layer
  Main_Agent.py
  Record_Subagent.py
  Hallucination_Subagent.py
  Information_Collection_Subagent.py
  Low_Confidence_Subagent.py
  Note_Review_Subagent.py
  Professor.py

Utility Layer
  deidentification_utils.py
  ui_app/services/llm_config_resolver.py

Prompt Layer
  prompt_main_agent.txt
  prompt_record_update.txt
  prompt_hallucination_check.txt
  prompt_information_collection_subagent.txt
  prompt_low_confidence_check.txt
  prompt_note_review.txt
  professor_*/prompt_*.txt

Storage Layer
  config.json
  Record_Template.txt
  patient_data/*
  professor_*/doc/*
  professor_*/chroma_doc_index/
  professor_*/parent_map.jsonl
```

## 專案結構

```text
.
├── TCM_Meridian_main.py
├── Main_Agent.py
├── Record_Subagent.py
├── Hallucination_Subagent.py
├── Information_Collection_Subagent.py
├── Low_Confidence_Subagent.py
├── Note_Review_Subagent.py
├── Professor.py
├── agent_behavior_log.py
├── deidentification_utils.py
├── multimodal_utils.py
├── prompt_*.txt
├── Record_Template.txt
├── config.json
├── requirements.txt
├── SPEC.md
├── ui_app/
│   ├── context.py
│   ├── shell.py
│   ├── rendering.py
│   ├── services/
│   │   ├── llm_config_resolver.py
│   │   └── ...
│   └── controllers/
├── config.example.json
├── .gitignore
├── patient_data/
├── professor-Template/
├── professor_01/              # 示範教授：自編中藥安全清單（孕婦/腎臟病禁忌）
└── professor_02/              # 示範教授：醫宗金鑑（公眾領域古籍）
```

> 註：`professor_01`、`professor_02` 為隨 repo 附帶的示範教授，含 `doc/` 原始知識庫，但**不含預建向量索引**（索引屬 build artifact，已在 `.gitignore` 排除）。clone 後在「教授設定」分頁點「建立資料庫」建立索引即可試用；也可新增/刪除/重建教授。

## 環境需求

- 建議 Python 3.10 以上。
- 主 Agent 與 Subagent 需要 OpenAI-compatible chat API。
- 教授 RAG 需要 OpenAI-compatible embedding endpoint。
- 可搭配 LM Studio、OpenRouter 或其他相容服務。

安裝依賴：

```bash
pip install -r requirements.txt
```

主要套件：

- `nicegui`
- `openai`
- `chromadb`
- `langchain`
- `langchain-community`
- `numpy`
- `requests`

## 快速開始

1. 安裝依賴。

   ```bash
   pip install -r requirements.txt
   ```

2. 複製設定範本並填入自己的 API endpoint / key / 模型名稱（之後也可在「模型設定」「教授設定」分頁調整）。

   ```bash
   cp config.example.json config.json
   ```

3. 啟動應用。

   ```bash
   python TCM_Meridian_main.py
   ```

4. 開啟瀏覽器。

   ```text
   http://localhost:8080
   ```

目前程式碼會綁定 `0.0.0.0:8080`，並以 NiceGUI reload 模式啟動。

## 設定檔

`config.json` 包含主 Agent、各 Subagent 與教授 RAG 共用設定。首次使用請**複製 repo 內的 `config.example.json` 為 `config.json`** 再填入自己的 endpoint/key（`config.json` 已被 `.gitignore` 排除，不會進版控）。以下是結構範例與預設值示意；實際工作區的 `config.json` 可能已改成 OpenRouter、LM Studio 或其他 endpoint。設定檔讀取失敗時系統會在主控台印出 `WARNING` 並回落預設設定；儲存設定時使用原子寫入。

```json
{
  "main_agent": {
    "api_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model_name": "",
    "max_tokens": 4000,
    "temperature": 0.7,
    "history_summary": {
      "api_url": "http://localhost:1234/v1",
      "api_key": "lm-studio",
      "model_name": "",
      "max_tokens": 4000,
      "temperature": 0.5
    },
    "summary_exit": {
      "api_url": "http://localhost:1234/v1",
      "api_key": "lm-studio",
      "model_name": "",
      "max_tokens": 128,
      "temperature": 0.2
    },
    "history_summary_model_name": "",
    "summary_exit_model_name": ""
  },
  "record_subagent": {
    "api_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model_name": "",
    "max_tokens": 8000,
    "temperature": 0.7
  },
  "hallucination_subagent": {
    "api_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model_name": "",
    "max_tokens": 8000,
    "temperature": 1.0,
    "detection_strength": 2,
    "max_review_rounds": 5
  },
  "ic_subagent": {
    "api_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model_name": "",
    "max_tokens": 20000,
    "temperature": 0.7,
    "max_collection_rounds": 10
  },
  "lc_subagent": {
    "api_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model_name": "",
    "max_tokens": 20000,
    "temperature": 1.0,
    "max_scan_rounds": 8,
    "detection_strength": 4
  },
  "nr_subagent": {
    "api_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model_name": "",
    "max_tokens": 20000,
    "temperature": 1.0
  },
  "professor_config": {
    "answer": {
      "api_url": "http://localhost:1234/v1",
      "api_key": "lm-studio",
      "model_name": "",
      "max_tokens": 20000,
      "temperature": 0.7
    },
    "embedding": {
      "api_url": "http://localhost:1234/v1",
      "api_key": "lm-studio",
      "model_name": ""
    },
    "query_expansion": {
      "api_url": "http://localhost:1234/v1",
      "api_key": "lm-studio",
      "model_name": ""
    },
    "prefix": {
      "api_url": "http://localhost:1234/v1",
      "api_key": "lm-studio",
      "model_name": ""
    },
    "rerank": {
      "api_url": "http://localhost:1234/v1",
      "api_key": "lm-studio",
      "model_name": ""
    }
  }
}
```

常見欄位：

- `api_url`
- `api_key`
- `model_name`
- `max_tokens`
- `temperature`

程式也支援進階欄位，例如 `main_agent.max_sub_turns`、`main_agent.history_summary_review_rounds`、`hallucination_subagent.max_review_rounds`、`ic_subagent.max_collection_rounds`、`lc_subagent.max_scan_rounds` 與各檢查強度設定；其中多數可由模型設定頁調整，未寫入 `config.json` 時會使用程式預設值。

Main Agent 子模型 `main_agent.history_summary`（歷史病歷摘要/檢查）與 `main_agent.summary_exit`（摘要並退出）為完整子設定 dict，可指向與 Main Agent 不同的 API 端點與模型；`model_name` 空白時依序 fallback 到 legacy 扁平鍵 `*_model_name`、再到 `main_agent.model_name`。`api_url` / `api_key` 空白時 fallback 到 Main Agent；`history_summary.max_tokens` 未設定時 fallback 到 Main Agent `max_tokens`，`summary_exit.max_tokens` 未設定時 fallback 到 `128`；temperature 分別 fallback 到 `0.5` 與 `0.2`。模型設定頁的子模型 `model_name` 不會自動預填 Main Agent 模型，避免把繼承模型誤固化成子模型專用模型。

檢測強度（`hallucination_subagent.detection_strength`、`lc_subagent.detection_strength`）設為 `0` 代表研究對照組模式（不檢查直接放行）；`> 0` 時後端會 clamp 進 `[1, 對應最大輪次]`。模型設定頁會擋下留空、負值、以及檢測強度大於最大輪次的不合法組合。

教授 RAG 另外分成回答模型、embedding、query expansion、三前綴分類與 rerank 模型。

## UI 分頁

目前共有 10 個頂層分頁：

1. 患者登錄
2. 醫療系統主介面
3. 影像檔查詢區
4. 醫療資訊檔案存放區
5. 醫療問答討論區
6. 自動問診對話區
7. 模型設定
8. 教授設定
9. 標準病歷模板設定
10. 智能體互動行為

## 患者資料格式

每位患者是一個資料夾：

```text
patient_data/<patient_id>_<birthday>_<name>/
├── patient_info.json
├── Picture_Row/
├── Medical_information/
├── <date>-NOTE.md
├── <date>-ASSESSMENT & TREATMENT.md
└── log/
    └── <date>-log/
        ├── <date>-session.log
        ├── <date>-chat-state.json
        ├── <date>-interview-state.json
        ├── <date>-information-collection-dialogue.txt  # 問診完成後產生/追加
        ├── <date>-forum-state.json
        ├── <date>-forum.txt                            # 教授問答後由 save_forum_state 整檔重寫
        ├── <date>-Human-Agent-Interaction.md
        ├── <date>-History-Summary.md
        ├── <date>-RAG-full-behavior.txt                # 教授 RAG 回答後產生/追加
        └── <date>-agent-behavior.jsonl                 # Agent 行為事件後產生/追加
```

`Picture_Row/` 保存患者圖片，`Medical_information/` 保存文字報告與醫療資訊。患者根目錄保存各日期的 NOTE 與 A&T Markdown。主 Agent 可透過 action 列出與讀取這些檔案；歷史病歷列表會在檔名後顯示 `note_summary` 或 `assessment_treatment_summary`，若尚未執行「摘要並退出」則以檔案前 50 字作為預設摘要。

主介面左欄的「選取日期」下拉會以 NOTE 摘要輔助辨識，例如 `2026-06-11（頭痛、喉嚨痛）`；下拉 value 仍是 `2026-06-11`，因此讀取、刪除、摘要並退出等 session 操作仍使用原日期字串。「選擇模板」下拉不顯示摘要，維持純日期列表。

「摘要並退出」只退出目前就診日期，不退出患者，且只作用於**已確認載入**的日期（未載入時提示先確認日期）。「刪除 session」則作用於**下拉框目前選取**的日期；刪除非載入中的日期時會保留載入中的病歷畫面。新增與摘要的就診日期會以 `strptime` 驗證日曆合法性，擋下不存在的日期。若主 Agent 正在執行、問診子流程進行中，或病歷仍在編輯模式，系統會阻止摘要並退出以避免狀態或草稿遺失。相同忙碌保護也套用於切換患者、切換 session、新增/刪除 session 與退出患者。摘要模型呼叫過程會在執行視窗印出 system prompt、user prompt 與 output；若模型未設定或呼叫失敗，會 fallback 使用原文前 50 字。

新建或載入 session 時產生的歷史病歷摘要會使用去識別化患者基本資料，並明確傳入【本次就診日期】，避免 LLM 將歷史病歷中的其他日期誤認為當日。歷史摘要的生成與審查/重寫共用同一個子模型（`main_agent.history_summary`）；若摘要產生失敗（未設定模型或呼叫失敗），新增 session 的 UI 會以 ⚠️ 明示失敗，而非誤報「已產生」。

## Agent 工作流

主 Agent 接收醫師訊息後，會進入 bounded ReAct loop。每個子輪都必須輸出 JSON：

```json
{
  "thinking": "...",
  "action": "reply",
  "action_input": "...",
  "next_step": "..."
}
```

支援 action：

- `reply`
- `update_record`
- `information_collection_subagent`
- `low_confidence_check`
- `note_review_subagent`
- `call_professor`
- `list_patient_files`
- `read_patient_file`

`list_patient_files` 會先提供檔名清單與歷史病歷摘要作為索引；完整清單只暫存在主 Agent context 的【患者檔案清單】區塊中，`read_patient_file` 完成後會自動移除，避免檔名清單長期累積 token。摘要僅供辨識檔案，若要引用或判斷內容，主 Agent 仍需讀取對應原文。

若本輪更新 NOTE，prompt 要求最後回覆前執行 `low_confidence_check`，除非醫師明確禁止。若本輪撰寫 A&T，prompt 要求在定稿前完成安全性流程：有安全性檢查教授時，需呼叫該教授審查目前 A&T；安全性為低時必須修正後再審，安全性為中/高時需在 A&T 最上方加入對應安全性標示。若沒有安全性檢查教授，A&T 最上方必須標註 `##注意，此分析未經過安全性檢查。`，最終回覆也需提醒醫師。

`update_record` 會由 Record Subagent 套用行級操作。所有操作的 `line` 都以模型看到的原始行號為準，程式會先整批驗證再一次重建內容；`insert` 行號大於文末時會被視為文末追加，`delete` / `replace` 行號超範圍、未知 op 或同一行重複 delete/replace 則會讓整批操作退回、病歷不寫入，並把具體原因顯示在步驟結果中。已設定幻覺審查模型時會進入 Hallucination Reviewer。審查通過需累積 `detection_strength` 次 agree（累積制，跨重寫版本計次）。審查採 **fail-closed**：若反覆重寫後仍達 `max_review_rounds` 上限未通過、迴圈耗盡仍未達門檻、或審查器服務異常（LLM/JSON 失敗），病歷**不寫入**、回傳原始 NOTE/A&T，`review_result` 標示為未通過或審查失敗，主 Agent 顯示失敗而非假成功（並依 prompt 規則不得宣稱已更新）。審查器服務異常時只呼叫一次即短路，不會反覆重寫燒 token。若 `detection_strength = 0`（研究對照組模式），則跳過審查直接放行並明確標示。

## 教授 RAG

每位教授是一個資料夾：

```text
professor_XX/
├── Description.txt
├── doc/
├── prompt_system.txt
├── prompt_3_prefix.txt
├── prompt_query_expansion.txt
├── prompt_rerank.txt
├── chroma_doc_index/    # 建立資料庫後產生（build artifact，不隨 repo 發布）
└── parent_map.jsonl     # 建立資料庫後產生（build artifact，不隨 repo 發布）
```

`Description.txt` 建議格式：

```json
{
  "name": "教授名稱",
  "description": "教授專長與回答風格"
}
```

可在「教授設定」分頁新增教授、修改描述、檢查檔案、設定模型與建立 Chroma index。新增教授會從 `professor-Template/` 複製 prompt 模板。「建立資料庫」會在重建前先清除舊索引（避免 chunk 疊加重複），建立成功後清除主 Agent 對該教授的快取，使新索引立即生效；刪除教授與重建前會先釋放 Chroma 檔案控制代碼，避免 Windows 檔案占用錯誤。教授頁所有會改動全域狀態的操作（新增/儲存描述/建庫/刪除/儲存共用模型）都要求先退出患者（「檢查檔案」為 read-only 不受限）。

### 知識庫文件格式（`doc/`）

每位教授的知識庫以純文字 `.txt` 檔存放於 `professor_XX/doc/` 內，可放多個檔。

本 RAG 採 **Parent-Child Chunking（父-子分塊檢索，又稱 Parent Document Retrieval）**：建立索引時，系統把每個「父段」再切成帶重疊的「子塊」存入向量庫；檢索時以子塊命中（命中率高），再映射回所屬父段，最終把**父段**餵給 LLM。因此**父段是「被檢索後輸入 LLM 的基本單位」**。

父段切割原則：

- **建議由人類專家執行切割**，使每個父段是「一段具有連續性、難以再切割的醫學文本段落」。
- `.txt` 內的父段之間以**兩個（含）以上的空行**區隔；只要出現連續兩個以上空行，系統即自動判別為不同父段。
- **每個父段的第一個字（詞）必須是代表該父段分類的前綴詞**，供三前綴分類路徑做子集檢索。

可用的前綴詞：

| 前綴 | 涵蓋內容 |
| --- | --- |
| `case` | 中醫醫案：病歷內容、中西醫診斷、病機證型、治則、處方用藥、針灸治療、病案分析、預後轉歸。 |
| `formula` | 方劑學：方劑組成、功效、主治、病機、組方思路、配伍、加減變化、服用方法。 |
| `herb` | 中藥學：藥物基原、性味、歸經、升降浮沉、功效、主治、臨床應用、炮製、禁忌。 |
| `acupuncture` | 針灸學：針灸理論、經絡腧穴、刺法、灸法、針灸治療。 |
| `diagnoses` | 中醫診斷學：四診、八綱辨證、氣血辨證、臟腑辨證、六淫與痰食辨證、傷寒與溫病辨證等。 |
| `treatment` | 中醫治則學：治未病、治病求本、陰陽調整、扶正祛邪、標本緩急、正治/反治、同病異治、異病同治、三因制宜、八法（汗吐下和溫清補消）、常見疾病的治則。 |
| `disease-Internal` | 中醫內科疾病：肝、心、脾、肺、腎及現代醫學消化、循環、泌尿、呼吸、內分泌系統疾病。 |
| `disease-Obstetrics&Gynecology` | 中醫婦科：基礎理論、病因病機、婦科診斷、治法、經帶胎產相關疾病。 |
| `disease-Pediatrics` | 中醫兒科：基礎理論與常見病。 |
| `disease-Osteology&Traumatology` | 中醫骨傷科：骨骼、關節、肌肉、筋腱疾病與損傷，內治、外治。 |
| `disease-Surgery` | 中醫外科：瘡瘍、癭瘤、乳房病、皮膚病、肛門直腸病、男性外科病及雜病。 |
| `disease-Dermatology` | 中醫皮膚科：各種皮膚病辨證與治療。 |
| `disease-Eye&ENT` | 中醫五官科：眼科、耳鼻喉科疾病。 |
| `theory` | 中醫理論：陰陽五行、藏象、氣血津液、經絡、病因病機、防治原則等。 |
| `classic` | 中醫典籍：黃帝內經、難經、傷寒論、金匱要略、溫病學等經典內容。 |
| `others` | 若無法匹配上述任一分類，使用 `others`。 |

> 註：本 repo 附帶的 `professor_02`（醫宗金鑑）示範知識庫，其**父段切割與分類前綴均由作者手動標註**完成，非自動分割。底本《醫宗金鑑》為公眾領域古籍，而父段切割與前綴分類屬作者的編輯成果。

## 安全清單

公開前建議至少排除：

```gitignore
config.json
patient_data/
**/log/
**/__pycache__/
professor_*/chroma_doc_index/
professor_*/parent_map.jsonl
```

若 `professor_*/doc/` 內容涉及私有資料或版權，也應排除。若 API key 曾被提交或分享，請立即輪換。

送入 LLM prompt 的患者基本資料會先去識別化：姓名保留首字並以「某」遮蔽其餘字，單字名至少補一個「某」；ID 不輸出；生日顯示為 `YYYY-XX-XX (X歲Y月)`（月/日遮蔽，年齡精確到歲與月）；性別、就診日期與備註保留。但 NOTE、A&T、患者備註、讀取檔案、log 與教授知識庫仍可能含有可識別資訊，公開或使用外部模型前仍需人工審查。

## 開發備註

- `TCM_Meridian_main.py` 應維持為啟動與組裝層。
- UI 行為放在 `ui_app/controllers/`。
- 檔案與狀態操作放在 `ui_app/services/`。
- 新增主 Agent action 時，需同步更新 `prompt_main_agent.txt`、`Main_Agent.py`、UI persistence、README 與 SPEC。
- 會改動病歷的流程應保留來源歸因、寫入前審查與寫入後低信心標註。

## 測試狀態

目前專案沒有獨立自動化測試目錄。建議的手動驗收清單請見 `SPEC.md`。

## 授權

Copyright 2026 Hong-Wen Hsieh

本專案採用 Apache License 2.0 授權，詳見 [`LICENSE`](LICENSE)。正式公開前仍建議再次確認第三方資料、患者資料與教授知識庫內容的授權與可散布性。
