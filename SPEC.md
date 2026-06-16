# TCM-Meridian 系統規格

本文件描述目前工作區中最新程式碼的實作狀態。內容以實際載入的 runtime 程式碼與 prompt 為準，不以舊版 prompt 封存或既有執行資料為準。

## 1. 系統定位

TCM-Meridian，中文名「杏林經緯」，是面向中醫診間的 AI 輔助工作台，主要組成如下：

- NiceGUI Web UI。
- 檔案式患者與就診 session 儲存。
- JSON action contract 的 ReAct 主 Agent。
- 病歷登載、幻覺審查、問診、低信心標註、病歷完整性檢查等 Subagent。
- 可追溯的教授 RAG 諮詢模組。
- 圖片與文字報告的多模態患者檔案讀取。
- 每個 session 的多 Agent 行為紀錄與 UI 時間線。

系統明確以人類醫師監督為前提。AI 產出是輔助，不是最終診斷或治療決策。

醫療免責：本系統僅供臨床決策支援與文件輔助，不是醫療器材，不提供獨立診斷、處方或治療建議，也不應用於無人監督的自動看診流程；所有臨床判斷、處置與用藥責任仍由合格醫師承擔。非醫療專業人員不應將本系統輸出作為自行診斷、治療或用藥依據；若自行使用，相關風險由使用者自行承擔。

## 2. 啟動入口

檔案：`TCM_Meridian_main.py`

職責：

- 解析專案根目錄、`patient_data`、`config.json`、`Record_Template.txt`。
- 建立 `AppContext` 與共享 `app_state`。
- 透過 `ui_app.services.config_service` 讀寫 JSON config。
- 透過 `create_app_services()` 建立 service bundle。
- 透過 `TabController` 註冊頂層分頁。
- 建立 header 與 tab shell。
- 以 NiceGUI 啟動：

```python
ui.run(
    title="杏林經緯 TCM-Meridian",
    host="0.0.0.0",
    port=8080,
    reload=True,
    favicon="🌿",
)
```

## 3. 程式碼分層

```text
Root modules
  TCM_Meridian_main.py
  Main_Agent.py
  Record_Subagent.py
  Hallucination_Subagent.py
  Information_Collection_Subagent.py
  Low_Confidence_Subagent.py
  Note_Review_Subagent.py
  Professor.py
  agent_behavior_log.py
  deidentification_utils.py
  multimodal_utils.py

UI package
  ui_app/context.py
  ui_app/shell.py
  ui_app/rendering.py
  ui_app/services/*.py
  ui_app/controllers/*.py
  ui_app/services/llm_config_resolver.py

Runtime prompts
  prompt_main_agent.txt
  prompt_record_update.txt
  prompt_hallucination_check.txt
  prompt_information_collection_subagent.txt
  prompt_low_confidence_check.txt
  prompt_note_review.txt
  professor_*/prompt_*.txt
```

Runtime 程式實際載入根目錄 `prompt_*.txt` 與各 `professor_*` 資料夾內 prompt。

## 4. UI 規格

目前 app 註冊 10 個頂層分頁：

| Key | 分頁名稱 | Builder |
| --- | --- | --- |
| `patient` | 患者登錄 | `build_patient_registration_tab` |
| `main` | 醫療系統主介面 | `build_medical_main_tab_controller` |
| `image` | 影像檔查詢區 | `build_image_tab` |
| `medinfo` | 醫療資訊檔案存放區 | `build_medinfo_tab` |
| `qa` | 醫療問答討論區 | `build_forum_tab` |
| `auto` | 自動問診對話區 | `AutoInterviewController.build_tab` |
| `model` | 模型設定 | `build_model_settings_tab` |
| `professor` | 教授設定 | `build_professor_settings_tab` |
| `template` | 標準病歷模板設定 | `build_record_template_tab` |
| `agent_behavior` | 智能體互動行為 | `build_agent_behavior_tab` |

### 4.1 醫療系統主介面

相關檔案：

- `ui_app/controllers/medical_main_layout.py`
- `ui_app/controllers/medical_main_controller.py`
- `ui_app/controllers/patient_session_lifecycle_controller.py`
- `ui_app/controllers/session_busy_guard.py`
- `ui_app/controllers/medical_record_controller.py`
- `ui_app/controllers/main_agent_turn_runner.py`
- `ui_app/controllers/main_agent_result_processor.py`
- `ui_app/controllers/main_chat_input_controller.py`
- `ui_app/controllers/main_chat_renderer.py`
- `ui_app/controllers/live_steps_controller.py`

版面：

- 左欄：患者選擇、session 選擇、新增 session、摘要並退出 session、患者備註。
- Session 選擇下拉顯示 `YYYY-MM-DD（NOTE 摘要）`，但 value 維持 `YYYY-MM-DD`；新增 session 的「選擇模板」下拉維持純日期，不顯示摘要。
- 中欄：NOTE 與 A&T 瀏覽、diff、修改模式、undo、redo。
- 右欄：主 Agent 對話、送出、中斷、即時步驟。

主介面使用 `SnapshotHistory` 保存 NOTE/A&T 版本。手動修改完成時，`tag_human_edits()` 會替新增或修改行加上 `[人類醫師_手動修改]`。主 Agent 同一主輪內每次成功 `update_record` 會回傳一筆 `record_snapshots`，UI 逐筆 push 成版本；若最終結果還包含額外變化，會再補一筆 final snapshot；若 final snapshot 與最後一筆中間 snapshot 相同，會略過避免重複。diff 模式會同時比對 NOTE 與 A&T，某欄位相鄰版本沒有變化時顯示「無差異」。

注意：`SnapshotHistory` 是 UI result processor 層級的正式版本紀錄，不是 Record Subagent 內部工作稿紀錄。若 Record Subagent 在單次 `update_record` 內經歷幻覺審查退稿、重寫或多次內部套用，UI 通常只保存該次 `update_record` 回傳給 Main Agent 的最後版本。若主 Agent / 模型呼叫在中間斷線，已形成但尚未回到 UI result processor 的中間版本也可能不會成為獨立 snapshot，後續 diff 可能呈現為「前一正式版本 → 恢復後最終版本」的合併差異。

### 4.2 影像檔查詢區

檔案：`ui_app/controllers/image_controller.py`

功能：

- 需要先選取患者。
- 匯入單張或多張圖片。
- 儲存至 `Picture_Row/`。
- 單檔命名為 `<date>-<suffix>.<ext>`。
- 多檔命名為 `<date>-<suffix>_<n>.<ext>`。
- 支援勾選預覽與刪除。

上傳暫存區會記錄上傳當下的患者資料夾；切換或退出患者時清空暫存，且儲存前會再次比對暫存圖片是否屬於目前患者，避免把上一位患者尚未儲存的圖片寫入另一位患者的 `Picture_Row/`。

支援副檔名：`.jpg`、`.jpeg`、`.png`、`.gif`、`.webp`、`.bmp`、`.tiff`、`.tif`。

### 4.3 醫療資訊檔案存放區

檔案：`ui_app/controllers/medinfo_controller.py`

功能：

- 需要先選取患者。
- 在 `Medical_information/` 建立 UTF-8 `.txt` 檔。
- 檔名格式為 `<date>-<suffix>.txt`，同名時自動加數字。
- 支援讀取、編輯、儲存、取消與刪除。

### 4.4 教授設定

檔案：`ui_app/controllers/professor_controller.py`

功能：

- 顯示會注入 `{professor_list}` 的教授清單。
- 從 `professor-Template/` 新增 `professor_XX`。
- 編輯 `Description.txt`。
- 檢查教授必要檔案（read-only，不受患者鎖限制）。
- 呼叫 `build_professor_index()` 建立 Chroma index。建立成功後會清除主 Agent 對該教授的 `ProfessorInstance` cache，使重建後的索引在下次諮詢即生效。
- 設定教授共用模型：`answer`、`embedding`、`query_expansion`、`prefix`、`rerank`。
- 設定變更後清空主 Agent 的教授 instance cache。

教授頁所有會改動全域狀態的操作（新增教授、儲存描述、建立資料庫、刪除教授、儲存教授共用模型）都要求先退出患者；有患者載入時這些按鈕會 disable 並提示，handler 也保留二次檢查。此規則與「標準病歷模板設定」「模型設定」一致：全域資源只能在無患者狀態變更，避免執行中 Agent 的快取或記憶與磁碟設定脫鉤。刪除教授與重建索引前會先釋放 Chroma 連線與 mmap 檔案控制代碼（`release_chroma_handles()`），避免 Windows 上 `chroma_doc_index` 檔案被占用而刪除失敗。

### 4.5 智能體互動行為

相關檔案：

- `agent_behavior_log.py`
- `ui_app/controllers/agent_behavior_controller.py`

行為紀錄檔為 `<date>-agent-behavior.jsonl`。UI 以 7 欄時間線顯示：

- AI主治醫師 Agent
- 病歷登載 Subagent
- 幻覺檢查 Subagent
- 問診助理 Subagent
- 低信心標註 Subagent
- 病歷檢查員 Subagent
- 醫學教授 Subagent

事件類型包含 LLM input、LLM output、tool call、RAG retrieval、manual stop、model error。

注意：行為時間線記錄的是每一次 LLM 原始輸入/輸出嘗試，而不是只記錄成功解析後的 action。若某次輸出無法解析為合法 JSON，系統會在同一 sub-turn 追加修復提示並重試；因此智能體互動行為中可能連續出現兩筆或多筆同一 agent 的「輸出」。這代表 JSON 解析失敗後的重試軌跡，不一定代表工具 action 被執行了多次。

## 5. Services

### 5.1 `AppContext`

檔案：`ui_app/context.py`

共享 state 主要欄位：

- `selected_patient_folder`
- `selected_patient_info`
- `selected_session_date`
- `agent_instance`
- `ic_subagent`
- `interview_dialogue`
- `interview_active`
- `session_generation`

`reset_agent_state()` 重置 agent 與問診狀態。`reset_patient_selection()` 清除患者/session 選擇並重置 agent。

### 5.2 `file_io`

檔案：`ui_app/services/file_io.py`

職責：

- `atomic_write_text(path, text)` 先在同目錄寫入暫存檔、flush/fsync 後以 `os.replace()` 置換目標檔。
- `atomic_write_json(path, data)` 將 JSON 序列化後走同一套原子寫入流程。
- 用於重要覆寫檔，降低程式崩潰或中斷時留下半截檔案的風險。

### 5.3 `PatientDataService`

檔案：`ui_app/services/patient_service.py`

職責：

- 建立、列出、讀取、更新、刪除患者資料夾。
- 建立、列出、讀取、保存、刪除就診 session。
- 患者 ID、生日或姓名變更時同步 rename 資料夾。
- 新增 session 時可從既有 session 複製 NOTE/A&T 作為模板。
- 建立 `patient_info.json`、保存 `patient_info.json`、新建空 NOTE/A&T 與保存 NOTE/A&T 時使用原子寫入；從舊 session 複製 NOTE/A&T 模板時使用一般檔案複製。
- `get_session_summaries()` 取得 `note_summary` / `assessment_treatment_summary`；若舊資料缺欄位或摘要為空，則以 NOTE/A&T Markdown 前 50 字 fallback。
- `save_session_summaries()` 將「摘要並退出」產生的 NOTE 與 A&T 索引摘要寫回 `patient_info.json`。

患者資料夾命名：

```text
<patient_id>_<birthday>_<name>
```

### 5.4 `SessionArtifactService`

檔案：`ui_app/services/session_artifact_service.py`

職責：

- 追加 `<date>-session.log`。
- 保存與讀取 `<date>-chat-state.json`。
- 保存與讀取 `<date>-interview-state.json`。
- 保存與讀取 `<date>-forum-state.json`，並同步以原子寫入重建人類可讀的 `<date>-forum.txt`（整檔重寫，與 `forum_history` 一致；手動中斷回滾 forum 後 txt 也會跟著修正）。
- 保存與讀取 `<date>-History-Summary.md`。
- 保存 `<date>-Human-Agent-Interaction.md`。
- 覆寫式保存項目使用原子寫入；append 型 log 維持追加寫入。

`<date>-information-collection-dialogue.txt` 由 `AutoInterviewController` 寫入。

### 5.5 `HistoryContextService`

檔案：`ui_app/services/history_context_service.py`

職責：

- 從最近一次較早 session 建立上次就診 block。
- 取最多 10 次歷史病歷產生歷史摘要。
- 對摘要進行審查與必要改寫。
- 透過 `SessionArtifactService` 保存最後摘要。
- 歷史摘要產生與審查 prompt 會使用去識別化患者基本資料，並明確傳入【本次就診日期】，避免模型把病歷內其他日期誤當成當日。
- 歷史摘要的生成與審查/重寫共用同一個 LLM client 與模型，由 `resolve_main_child_llm_config()` 解析 `main_agent.history_summary` 子設定（見 §7）；子設定 `model_name` 空白時依序 fallback 到 legacy `history_summary_model_name`、再到 `main_agent.model_name`，三者皆空才回傳「未設定模型」提示。摘要產生失敗（未設定模型或 LLM 呼叫失敗）時，新增 session 流程的 UI 會以 ⚠️ 明示「已建立，但歷史病歷摘要產生失敗」，不再誤報「已產生」。

隱藏設定欄位：

- `main_agent.history_summary_review_rounds`，預設 `3`。

子模型端點解析：

- `ui_app/services/llm_config_resolver.py` 的 `resolve_main_child_llm_config()` 統一解析 Main Agent 子模型（`history_summary`、`summary_exit`）的 `api_url`、`api_key`、`model_name`、`max_tokens`、`temperature`，並提供向後相容 fallback：`model_name` 依序讀取巢狀子設定、legacy `*_model_name` 扁平鍵、`main_agent.model_name`；`api_url` / `api_key` 空白時 fallback 到 Main Agent；`max_tokens` / `temperature` 空白時使用呼叫端提供的功能預設。

### 5.6 `AgentFactory`

檔案：`ui_app/services/agent_factory.py`

建立：

- `MainAgent`
- `InformationCollectionSubagent`

`*_if_configured` helper 只會在對應 `model_name` 存在時建立 agent。

## 6. 患者資料模型

### 6.1 `patient_info.json`

建立時的基本結構：

```json
{
  "basic_info": {
    "id": "",
    "name": "",
    "gender": "",
    "birthday": "",
    "phone": "",
    "address": "",
    "remark": ""
  },
  "directories": {
    "picture_row_dir": "Picture_Row/",
    "medical_info_dir": "Medical_information/",
    "log_dir": "log/"
  },
  "raw_images": [],
  "medical_information": {
    "Lab_Data": [],
    "Image_Data": [],
    "Special_Examination": [],
    "Medication_Profile": [],
    "Admission_Progress_Discharge_Note": []
  },
  "sessions": {}
}
```

Session entry：

```json
{
  "2026-06-06": {
    "note_file": "2026-06-06-NOTE.md",
    "assessment_treatment_file": "2026-06-06-ASSESSMENT & TREATMENT.md",
    "log_folder": "log/2026-06-06-log/",
    "note_summary": "50 字內 NOTE 摘要",
    "assessment_treatment_summary": "50 字內 A&T 摘要"
  }
}
```

`note_summary` 與 `assessment_treatment_summary` 由主介面「摘要並退出」寫入。若舊資料或未執行摘要流程導致欄位不存在或為空，檔案列表與日期下拉以對應 Markdown 檔案前 50 字作為 fallback 摘要。日期下拉只使用 NOTE 摘要輔助辨識；A&T 摘要主要供歷史病歷檔案清單辨識。

`raw_images` 與 `medical_information` 欄位保留在 `patient_info.json` schema 中；目前影像與醫療資訊分頁的檔案列表主要由掃描 `Picture_Row/` 與 `Medical_information/` 目錄取得，並不依賴這兩個欄位作為單一索引來源。

### 6.2 Session artifact

```text
log/<date>-log/
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

其中 chat、interview、forum、RAG 與 agent behavior 相關檔案依實際互動流程產生；新 session 不保證一開始就具備全部 artifact。

## 7. 設定規格

`config.json` 從專案根目錄讀取。若不存在或讀取失敗，使用 `TCM_Meridian_main.py` 的 `_default_config()`；讀取失敗時會在主控台印出 `WARNING` 與錯誤原因，避免設定檔壞掉時無聲降級。設定檔保存使用原子寫入。

常見 agent 欄位：

- `api_url`
- `api_key`
- `model_name`
- `max_tokens`
- `temperature`

額外欄位：

- `main_agent.max_sub_turns`
- `main_agent.history_summary_review_rounds`
- `main_agent.history_summary`（巢狀子模型設定，見下）
- `main_agent.summary_exit`（巢狀子模型設定，見下）
- `main_agent.history_summary_model_name` / `main_agent.summary_exit_model_name`（legacy 扁平鍵，仍保留供向後相容 fallback）
- `hallucination_subagent.detection_strength`
- `hallucination_subagent.max_review_rounds`
- `ic_subagent.max_collection_rounds`
- `lc_subagent.max_scan_rounds`
- `lc_subagent.detection_strength`

Main Agent 子模型（歷史病歷摘要/檢查、摘要並退出）：

- `main_agent.history_summary` 與 `main_agent.summary_exit` 為完整子設定 dict，可各自設定 `api_url`、`api_key`、`model_name`、`max_tokens`、`temperature`，因此兩者可指向與 Main Agent 不同的 API 端點與模型。
- `model_name` 空白時，依序 fallback 到 legacy `*_model_name` 扁平鍵、再到 `main_agent.model_name`；`api_url` / `api_key` 空白時 fallback 到 `main_agent` 對應值。
- `main_agent.history_summary.max_tokens` 未設定時，呼叫端以 `main_agent.max_tokens` 作為預設；`main_agent.summary_exit.max_tokens` 未設定時，呼叫端以 `128` 作為預設。兩者 temperature 未設定時分別 fallback 到 `0.5` 與 `0.2`。
- 模型設定頁的 UI 預設不會把 `main_agent.model_name` 預填進子模型的 `model_name` 欄位，避免將繼承的模型誤固化成子模型專用模型。

檢測強度設定語意（`hallucination_subagent.detection_strength`、`lc_subagent.detection_strength`）：

- `> 0`：累積達該次數的 agree/pass 後通過；後端會 clamp 進 `[1, max_review_rounds]`（幻覺）或 `[1, max_scan_rounds]`（低信心），避免門檻永遠不可達。
- `= 0`：研究對照組模式 — 不執行該檢查、直接放行，並寫入 `severity="warning"` 的行為事件、在步驟結果標示「未審查（檢測強度0/對照組）」或「未執行（檢測強度0/對照組）」。
- 模型設定頁防呆：檢測強度欄位留空會被擋下（不會被默默當成 0）；檢測強度不可為負、不可大於對應的最大輪次；最大輪次必須至少為 1。

程式預設值重點：

| Section | max_tokens | temperature | 其他預設 |
| --- | ---: | ---: | --- |
| `main_agent` | 4000 | 0.7 | `max_sub_turns=10`、`history_summary_review_rounds=3` 由讀取處 fallback |
| `main_agent.history_summary` | `main_agent.max_tokens` | 0.5 | `model_name` 空白時沿用 Main Agent；default config 範例為 4000 |
| `main_agent.summary_exit` | 128 | 0.2 | `model_name` 空白時沿用 Main Agent |
| `record_subagent` | 8000 | 0.7 | - |
| `hallucination_subagent` | 8000 | 1.0 | `detection_strength=2`、`max_review_rounds=5` |
| `ic_subagent` | 20000 | 0.7 | `max_collection_rounds=10` |
| `lc_subagent` | 20000 | 1.0 | `max_scan_rounds=8`、`detection_strength=4` |
| `nr_subagent` | 20000 | 1.0 | - |
| `professor_config.answer` | 20000 | 0.7 | 其他 professor 子設定只需要 endpoint/key/model |

教授設定：

- `professor_config.answer`
- `professor_config.embedding`
- `professor_config.query_expansion`
- `professor_config.prefix`
- `professor_config.rerank`

所有 LLM client 都採 OpenAI-compatible API。

## 8. Main Agent 規格

檔案：`Main_Agent.py`

### 8.1 建構

`MainAgent` 接收：

- `main_config`
- `record_config`
- `hallucination_config`
- `ic_config`
- `lc_config`
- `nr_config`
- `professor_config`

建構時建立：

- OpenAI-compatible main client。
- `RecordSubagent`。
- 可選 `LowConfidenceSubagent`。
- 可選 `NoteReviewSubagent`。
- lazy `ProfessorInstance` cache。

建構時載入：

- `prompt_main_agent.txt`
- `Record_Template.txt`

### 8.2 狀態保存

`export_state()` 保存：

- `turn_count`
- `turn_history`
- `conversation_history`
- `_suspended`
- `forum_history`
- `_file_list_cache`
- `_patient_folder`
- `_active_turn`

`restore_state()` 還原上述狀態。

### 8.3 ReAct contract

每個子輪必須輸出 JSON：

```json
{
  "thinking": "reasoning summary",
  "action": "action_name",
  "action_input": "action payload",
  "next_step": "next plan"
}
```

支援 action：

| Action | 用途 |
| --- | --- |
| `reply` | 結束本主輪並回覆醫師。 |
| `update_record` | 呼叫 `RecordSubagent` 更新 `note` 或 `assessment_&_treatment`。 |
| `information_collection_subagent` | 暫停主 loop 並啟動問診助理。 |
| `low_confidence_check` | 對 NOTE 執行低信心標註。 |
| `note_review_subagent` | 檢查 NOTE 完整性，必要時啟動問診。 |
| `call_professor` | 呼叫教授 RAG。 |
| `list_patient_files` | 列出 `Picture_Row/`、`Medical_information/` 與/或患者根目錄歷史病歷 Markdown；病歷檔名後會附 50 字內摘要。 |
| `read_patient_file` | 讀取文字、歷史病歷 Markdown 或圖片檔並放入當輪暫存區。 |

`list_patient_files` 的完整清單只暫存在 `_file_list_cache`，並注入 context prompt 的【患者檔案清單】區塊；`step_record.result` 只保留分類摘要，避免完整檔名清單長期累積 token。`read_patient_file` 完成後會自動清空 `_file_list_cache`。歷史病歷摘要僅供辨識，主 Agent 若要引用或判斷內容仍需讀取 `.md` 原文。

`MAX_JSON_RETRIES` 為 `3`。`MAX_SUB_TURNS` 預設 `10`，可由 `main_agent.max_sub_turns` 覆蓋。達子輪上限強制結束時，會與 `reply` 路徑一樣呼叫 `_finalize_turn_history()` 正式收斂該主輪（寫入 `turn_history`、清除 `_active_turn`），因此下一個主輪不會把它誤判成「服務已中斷(模型呼叫失敗)」。

`information_collection_subagent` 與 `note_review_subagent` 自動銜接問診前，會先檢查 `ic_config.model_name` 是否設定；未設定時不進入 `_suspended` 暫停狀態，而是把該子輪標記為「問診助理未設定」並繼續迴圈，避免主輪永久懸置。每個主輪開始時 (`process_message`) 也會清除任何殘留的孤兒 `_suspended`，還原無 interview-state 時亦會清除，避免陳舊暫停狀態汙染後續主輪。

### 8.4 Context prompt

`_build_context_prompt()` 包含：

- 去識別化患者基本資料：姓名保留首字並以「某」遮蔽其餘字，單字名至少補一個「某」；不輸出 ID；性別保留；生日顯示為 `YYYY-XX-XX (X歲Y月)`（月/日遮蔽，年齡精確到歲與月）；就診日期與備註保留。
- 人類醫師與 AI 主治醫師詳細互動史。
- 已完成主輪。
- 上次異常中斷但保存的 active turn。
- 目前主輪 steps。
- 完整問診對話紀錄。
- 醫療問答討論區。
- 今日 NOTE。
- 今日 A&T。
- 患者檔案清單 cache。
- 當輪讀取檔案暫存區。

圖片由 `inject_images_into_messages()` 注入 OpenAI vision message。

### 8.5 合作式中斷

`request_manual_stop()` 會設定 stop event 並捕捉最新安全快照。ReAct loop 會在 model/tool 操作之間檢查此事件。`_finalize_manual_stop_turn()` 會：

- 只保留已完成 steps。
- 必要時回滾未提交的 forum additions。
- 回傳快照中的 NOTE/A&T。
- 寫入 `manual_stop` behavior event。

## 9. Subagent 規格

### 9.1 Record Subagent

檔案：`Record_Subagent.py`

職責：

- 只更新單一目標欄位：`note` 或 `assessment_&_treatment`。
- 將待修改內容加上行號。
- 要求模型輸出 line operations。
- `apply_operations()` 使用原始行號語意：所有 `line` 都指向模型看到的「待修改欄位（附行號）」原始行號，不因前序操作而位移。
- 套用前會整批驗證操作；任一操作不合法時整批退回，NOTE/A&T 保持原樣；`insert` 行號大於文末時會先正規化為文末追加。
- 可選擇執行 Hallucination Reviewer。
- 審查未通過時重寫，最多到 `max_review_rounds`。
- **Fail-closed 安全閘門**：若達 `max_review_rounds` 仍未通過、或迴圈結束仍未累積足夠 agree，`RecordSubagent` 會以 `success=False` 回傳**原始** NOTE/A&T（草稿丟棄、病歷不寫入），`review_result` 標示 `未通過（達上限）` 或 `未通過（僅通過N次）`；Main Agent 顯示失敗而非假成功。
- **審查服務異常短路**：若 Hallucination Reviewer 回傳帶 `error` 欄位（LLM 呼叫失敗或 JSON 解析失敗），`RecordSubagent` 會直接 `success=False`、不進入重寫迴圈（避免對著壞掉的審查器反覆重寫燒 token），`review_result` 標示 `審查失敗（服務異常）`。
- **對照組模式**：若 `detection_strength = 0`（且審查器有設定 model），跳過審查直接放行，`review_result` 標示 `未審查（檢測強度0/對照組）`，並寫入警示行為事件。
- 後端會將 `max_review_rounds` 保底為至少 1、`detection_strength`（>0 時）clamp 進 `[1, max_review_rounds]`。
- `prompt_main_agent.txt` 要求：若病歷登載被拒絕（審查未通過/服務異常/未寫入），主 Agent 不得在最終 reply 宣稱已更新，應給更保守方針重試、補問診/讀檔/諮詢，或如實告知未寫入原因。

操作格式：

```json
{
  "operations": [
    {"op": "replace", "line": 3, "content": "..."},
    {"op": "insert", "line": 5, "content": "..."},
    {"op": "delete", "line": 8}
  ]
}
```

支援 `insert`、`delete`、`replace`。

行級操作規則：

- `insert` 的 `line = N` 表示插在原始第 N 行之前；`line = 原始總行數 + 1` 表示末尾追加；若 `insert` 行號大於 `原始總行數 + 1`，程式會容錯正規化成文末追加並記錄 normalization log。
- 同一位置連續插入可送出多個相同 `line` 的 `insert`，依陣列順序插入。
- `delete` / `replace` 的行號必須在原始行數範圍內。
- 同一行不可重複或同時 `delete` / `replace`。
- 未知 op、非整數行號、`delete` / `replace` 行號超範圍或同行衝突會讓整批操作失敗；`execute()` 回傳 `success=False` 與具體原因，Main Agent 顯示失敗而非假成功。

### 9.2 Hallucination Subagent

檔案：`Hallucination_Subagent.py`

寫入前審查待登載內容，回傳：

```json
{
  "thinking": "...",
  "agree": "yes",
  "comment": "..."
}
```

審查類別：

- A. 直接矛盾、否定反轉、內文自相矛盾。
- B. 無中生有與模板殘留。
- C. 不確定性膨脹。
- D. 過度具體化與推論外顯化。
- E. 時序、歷史資訊與病程階段錯置。
- F. 數值、單位、頻次、檢驗與藥物紀錄錯置。
- G. 來源標註、來源歸因與實體錯位。
- H. 關鍵資訊遺漏。

`MAX_JSON_RETRIES` 為 `2`。**Fail-closed**：LLM 呼叫失敗或 JSON 解析重試耗盡時回傳 `agree="no"` 並帶 `error` 欄位（審查無法執行時拒絕寫入，而非預設放行）；`RecordSubagent` 會據此短路、不重寫。

通過採**累積制**：跨重寫版本累計 agree 次數，累積達 `detection_strength` 即通過（非連續制；累積制把關較不易在重寫迴圈中耗盡輪次，且跨過門檻的最後一次 agree 必然針對最終版本）。`detection_strength = 0` 為研究對照組模式，由 `RecordSubagent` 在呼叫審查器前直接跳過。

### 9.3 Information Collection Subagent

檔案：`Information_Collection_Subagent.py`

職責：

- 根據主 Agent 方針啟動問診循環。
- 產生聚焦的多回合問題。
- 保存當前循環對話與跨循環累積對話。
- 支援 export/restore state。
- 問診完成後產出可交給 `MainAgent.continue_after_interview()` 的摘要。

支援 action：

- `ask_patient`
- `finish_collection`

主要狀態欄位：

- `conversation`
- `all_conversations`
- `dialogue_round`
- `turn_count`
- `turn_history`
- `guidelines`
- `interaction_history`
- `forum_history`
- `finished`

### 9.4 Low Confidence Subagent

檔案：`Low_Confidence_Subagent.py`

職責：

- 寫入後掃描 NOTE。
- 以 `**片段（原因）**` 標註低信心或需確認片段。
- 迭代直到累積 `detection_strength` 次 pass 或達 `max_scan_rounds`。
- 保存 `all_phrases` 作為稽核資料。
- 後端會將 `max_scan_rounds` 保底為至少 1、`detection_strength`（>0 時）clamp 進 `[1, max_scan_rounds]`。
- **對照組模式**：`detection_strength = 0` 時跳過掃描直接放行，回傳帶 `skipped_control_group=True`、NOTE 不變，並寫入警示行為事件；Main Agent 步驟結果標示「低信心標註未執行（檢測強度0/對照組），NOTE 未變更」。

它是 post-write safety net，不是 pre-write rejection gate。

### 9.5 Note Review Subagent

檔案：`Note_Review_Subagent.py`

職責：

- 對照 `Record_Template.txt` 檢查 NOTE。
- 找出缺漏或未完成項目。
- 若需補問，產生給 Information Collection Subagent 的 `action_input`。
- 若已有資訊但尚未登載，產生 `update_reminder`。

不負責診斷、治療計畫、開藥或文筆潤飾。

## 10. Professor RAG 規格

檔案：`Professor.py`

### 10.1 教授資料夾

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

`check_professor_files()` 會檢查必要檔案。

### 10.2 Runtime flow

`ProfessorInstance.answer()` 流程：

1. `_ensure_index()` 載入 parent map、Chroma documents、metadata、embeddings。
2. 建立原始 query 與含上下文 query。
3. `_query_expansion()` 執行查詢擴展。
4. `_classify_prefixes()` 選出三個 retrieval prefix。
5. `_retrieve()` 執行 dense retrieval、RRF、prefix-boosted retrieval、parent mapping、LLM rerank 與 parent selection。
6. `_generate_answer()` 根據臨床上下文與檢索原文生成教授回答。
7. Main Agent 寫入 `<date>-RAG-full-behavior.txt`。

### 10.3 Retrieval constants

```python
RAG_TOPK_CAND = 250
RAG_TOPK_PREFIX = 45
RAG_TOPK_FULL = 30
RAG_RRF_K = 60
RAG_RERANK_FLOOR = 0.2
RAG_RERANK_AUTO = 0.7
RAG_MIN_PARENTS = 10
RAG_MAX_PARENTS = 18
EMB_NORMALIZE = True
CHUNK_SIZE = 400
CHUNK_OVER = 120
CHUNK_SIZE_CASE = 280
CHUNK_OVER_CASE = 70
```

目前這些是程式碼常數，尚未 config-driven。

### 10.4 建立索引

`build_professor_index()`：

- 讀取 `doc/*.txt`。
- 重建前先釋放控制代碼並清除既有 `chroma_doc_index/`（`Chroma.from_documents` 對既有 collection 是 append，不清除會使 chunk 疊加重複、舊 `parent_id` 變孤兒）。
- 以三個以上換行切 parent block。
- 透過 `_PREFIX_REGEX` 偵測 role prefix。
- 寫出 `parent_map.jsonl`。
- 建立 overlapping child chunks。
- 寫入 Chroma collection `doc_blocks`。
- 建立完成後釋放 Chroma 連線與 mmap 控制代碼（`release_chroma_handles()`），讓後續刪除教授或再次重建不會在 Windows 上撞檔案占用錯誤。

Role prefix 包含 case、formula、herb、acupuncture、diagnoses、treatment、各科 disease、theory、classic、others、NoRAG。

## 11. 多模態檔案規格

檔案：`multimodal_utils.py`

支援圖片副檔名：

- `.jpg`
- `.jpeg`
- `.png`
- `.gif`
- `.webp`
- `.bmp`
- `.tiff`
- `.tif`

函式：

- `encode_image_to_base64(file_path)`
- `get_image_media_type(file_path)`
- `inject_images_into_messages(messages, loaded_files)`

`read_patient_file` 會把圖片檔以 path metadata 放入 `_loaded_files`，文字檔與歷史病歷 Markdown 以 UTF-8 content 放入 `_loaded_files`。當輪暫存檔案會在每個 main turn 開始時清空。

## 12. Prompt contract

### 12.1 Main Agent prompt

檔案：`prompt_main_agent.txt`

核心限制：

- 臨床 workflow 中，人類醫師即時指令優先。
- 未經醫師要求，不啟動問診。
- 未經醫師要求，不呼叫教授。
- 未經醫師要求，不更新 A&T。
- 若本輪更新 NOTE，且醫師未明確禁止，最終 reply 前需執行 `low_confidence_check`。
- 若撰寫 A&T，需依 prompt 執行安全審核與標示流程：有安全性檢查教授時需呼叫該教授審查目前 A&T，安全性為低時修正後再審，安全性為中/高時在 A&T 最上方加入對應安全性標示；若無安全性檢查教授，需在 A&T 最上方加入 `##注意，此分析未經過安全性檢查。`，最終 reply 也需提醒醫師。
- 讀取患者檔案前需先 `list_patient_files`，再 `read_patient_file`。
- 輸出必須是 raw JSON。

### 12.2 Record 與 Review prompts

檔案：

- `prompt_record_update.txt`
- `prompt_hallucination_check.txt`
- `prompt_low_confidence_check.txt`
- `prompt_note_review.txt`
- `prompt_information_collection_subagent.txt`

共同病歷原則：

- 區分陽性、陰性、疑似、未知、矛盾、變化、不適用。
- 不把未知寫成陰性。
- 不把部分證據寫成確定結論。
- 保留來源歸因。
- 若重要資訊遺漏會影響安全或診療方向，需明確處理。

合法來源標籤：

- `[問診紀錄_患者R{N}]`
- `[問診紀錄_家屬R{N}]`
- `[問診紀錄_實習醫師R{N}]`
- `[問診紀錄_病歷系統R{N}]`
- `[人類醫師_R{N}]`
- `[人類醫師_手動修改]`
- `[上次病歷]`
- `[歷史病歷]`
- `[Forum_D{N}]`
- `[檔名]`

## 13. 狀態保存與還原

選取 session 時：

- 讀取 NOTE/A&T 並 push 到 `SnapshotHistory`。
- 日期下拉顯示 `YYYY-MM-DD（NOTE 摘要）`，但 `selected_session_date` 與 `session_select.value` 仍保存純日期。
- 從 `<date>-chat-state.json` 還原聊天狀態。
- 若有保存的主 Agent state，還原主 Agent。
- 從 `<date>-interview-state.json` 還原問診狀態。
- 從 `<date>-forum-state.json` 還原討論區。
- 重新整理智能體互動行為分頁。

`session_generation` guard 會避免切換患者或 session 後，舊背景任務結果被套用到新 session。

「摘要並退出」會摘要目前 session 的 NOTE 與 A&T，寫入 `patient_info.json` 後退出該就診日期但保留患者選取。若主 Agent 執行中、問診子流程進行中，或 UI 仍處於編輯模式，流程會被阻止，以避免背景狀態或未提交草稿遺失。

日期來源語義刻意拆分：「摘要並退出」只作用於**已確認載入**的 `selected_session_date`（未載入時提示先確認日期，不 fallback 到下拉值），因為摘要來源是畫面上的 NOTE/A&T；「刪除 session」作用於**下拉框目前選取**的日期，刪除非載入中的日期時只刷新清單、保留載入中的病歷畫面，刪除載入中的日期才清空 session UI。新增 session 與摘要並退出的就診日期除了格式 regex 外，也會用 `datetime.strptime` 驗證日曆合法性（擋下 `2026-13-99`、非閏年 `2026-02-29` 等）。

患者/session 切換與刪除入口共用 `SessionBusyGuard` 忙碌保護：確認患者、確認就診日期、新增 session、刪除 session、退出患者與摘要並退出都會先檢查編輯模式、`agent_running`、`interview_active` 與 `session_transition`。命中時只顯示提示並停止操作，不重置 agent/input 狀態。忙碌期間左側患者/session 導覽、日期下拉、新增/刪除、摘要並退出、備註輸入框與備註儲存會被 disable，避免下拉值與實際 app state 不一致。長任務（摘要並退出、新增 session 產生歷史摘要）以 `begin_transition()`/`end_transition()` 包夾，期間左欄與右欄主 Agent 送出鈕一併視覺鎖定（`begin_transition` 對 `_main_chat_input` 上 `session_transition` 鎖，`end_transition` 解除；流程中途 `reset()` 不會誤解該鎖）。中欄 undo/redo 也納入忙碌檢查，避免 Agent 執行中以舊版病歷覆寫磁碟造成競態。

同一 guard 也套用於主 Agent 送出、進入病歷修改模式、患者登錄新增/編輯/刪除、圖片與醫療資訊寫入/刪除、模型設定、教授設定與病歷模板儲存；handler 仍保留檢查，UI disable 只是第一層防誤觸。

全域設定（標準病歷模板、模型設定、教授設定三頁）另有更嚴格規則：**只能在無患者載入時變更**。有患者載入時相關儲存/變更按鈕會 disable 並顯示「請先退出患者」，handler 內保留二次檢查。患者載入/退出時由 `sync_global_settings_save_state()` 統一同步三頁的鎖定狀態。此規則的目的是消除「全域資源在 session 進行中被抽換、導致執行中 Agent 的快取（模板、教授索引）或記憶與磁碟設定脫鉤」的狀態空間（例如儲存模型設定會 `reset_agent_state()`，若 session 進行中會造成 UI 對話延續但 Agent 記憶被清空的失憶錯覺）。

問診子流程完成後，`AutoInterviewController.resume_agent_after_interview()` 會重新啟動右側即時步驟面板，讓主 Agent 恢復執行期間的更新病歷、低信心標註與最終回覆等步驟可見；流程結束時關閉面板。

## 14. 錯誤處理與限制

已實作：

- 主 Agent 與 Subagent 的 JSON 解析重試。
- JSON 解析失敗時，每一次 LLM raw output 都會先寫入智能體互動行為；同一 sub-turn 可能因此連續出現多筆「輸出」事件，屬於重試紀錄而非重複執行 action。
- config 讀取失敗時會印出 `WARNING` 與錯誤原因，並回落 default config；config 保存使用原子寫入。
- 重要覆寫檔使用原子寫入，包括 `patient_info.json`、空白 NOTE/A&T 建立、NOTE/A&T 一般保存、chat/interview/forum state、歷史摘要、對話紀錄與病歷模板；從舊 session 複製 NOTE/A&T 模板時使用一般檔案複製。
- Record Subagent 任一 line operation 不合法時整批退回，不寫入半套病歷；例外是超過文末的 `insert` 會容錯正規化為文末追加。
- chat/interview/forum state 讀取失敗時回傳空狀態。
- Main Agent 有 `max_sub_turns`。
- IC Subagent 有 `max_collection_rounds`。
- LC Subagent 有 `max_scan_rounds` 與 `detection_strength`；`detection_strength=0` 為對照組（跳過掃描）。
- Hallucination review 有 `detection_strength` 與 `max_review_rounds`，採 **fail-closed**：達上限未通過、迴圈耗盡未達門檻、或審查器服務異常（LLM/JSON 失敗）時，病歷不寫入、回傳原始 NOTE/A&T，`review_result` 標示為未通過或審查失敗；`detection_strength=0` 為對照組（跳過審查直接放行）。
- 主 Agent 達 `MAX_SUB_TURNS` 上限時會正式收斂主輪，不殘留 `_active_turn`。
- IC 未設定 model 時，主 Agent 不會進入永久暫停；殘留孤兒 `_suspended` 會在下個主輪或還原時清除。
- 主 Agent 與問診流程支援合作式中斷。
- session generation guard 會丟棄過時背景結果。
- 問診送出/終止之間以 `_ic_run_id` 防競態：終止問診會遞增 run id，使仍在背景執行緒中的問診結果回來後被丟棄（不重新點亮 UI、不恢復主 Agent），並以啟動前的 rollback point 還原被背景執行緒汙染的問診狀態。
- session busy guard 會在編輯、AI 執行、問診與 session transition 期間拒絕危險操作，並同步鎖定主介面導覽（含右欄送出鈕於 transition 期間）。

已知限制：

- 沒有 transaction database layer。
- append 型 log 仍使用追加寫入；原子寫入主要覆蓋覆寫式保存檔案。
- 患者/session 刪除直接操作檔案系統。
- Professor retrieval constants 寫死在程式碼。
- 目前沒有獨立自動化測試目錄。
- secret management 尚未與 `config.json` 分離。
- runtime 患者資料與教授文件可能含有隱私或版權資料。
- UI diff 版本歷史只保證記錄已回到 UI result processor 的正式 snapshot；模型斷線、手動中斷或 Record Subagent 內部退稿重寫期間的中間稿，可能不會獨立出現在 diff 歷史中，而是被合併到下一個正式版本差異。

## 15. 隱私與安全要求

專案程式碼授權檔為 Apache License 2.0 `LICENSE`；此授權不代表 runtime 患者資料、log、第三方資料或教授知識庫內容可被一併散布。

敏感路徑：

- `config.json`
- `patient_data/`
- `patient_data/*/log/`
- `professor_*/doc/`
- `professor_*/parent_map.jsonl`
- `professor_*/chroma_doc_index/`

送入 LLM prompt 的患者基本資料會先去識別化：姓名保留首字並以「某」遮蔽其餘字，單字名至少補一個「某」；ID 不輸出；生日顯示為 `YYYY-XX-XX (X歲Y月)`（月/日遮蔽，年齡精確到歲與月）；性別、就診日期與備註保留。但 NOTE、A&T、患者備註、讀取檔案、log 與教授知識庫仍可能含有可識別資訊，公開或使用外部模型前仍需人工審查。

repo 已附 `.gitignore` 與 `config.example.json`。`.gitignore` 的重點：

```gitignore
config.json                       # 含 API key，改用 config.example.json 範本
patient_data/*                    # 排除所有患者資料（保護真實病歷）
.claude/                          # Claude Code 本機工具設定
professor_*/chroma_doc_index/     # 教授向量索引（build artifact，可由 doc/ 重建）
professor_*/parent_map.jsonl
**/__pycache__/
```

注意：示範教授 `professor_01`（自編安全清單）、`professor_02`（醫宗金鑑，公眾領域古籍）**只發布 `doc/` 原始知識庫 + prompt + `Description.txt`**；向量索引（`chroma_doc_index/`、`parent_map.jsonl`）屬 build artifact，不發布——請 clone 後在「教授設定」點「建立資料庫」重建。未來若自建使用版權教材的教授，需手動在 `.gitignore` 排除整個 `professor_XX/`。設定檔請複製 `config.example.json` 為 `config.json` 後填入自己的 endpoint/key；`config.example.json` 為 localhost/LM Studio 範本，不含真實 key。

正式部署建議：

- API key 改用環境變數或 secret manager。
- 加入身份驗證與 HTTPS。
- 定義患者資料保存與刪除政策。
- 定義 log 保存期限與存取權限。
- 稽核教授知識庫授權與可散布性。
- 輪換任何曾被提交或分享的 API key。

## 16. 手動驗收清單

功能驗收：

- App 可在 port 8080 啟動。
- 可建立、選取、更新、刪除患者。
- 可建立、載入、從舊 session 複製、刪除 session。
- 新建空 NOTE/A&T、保存 NOTE/A&T 與保存 `patient_info.json` 使用原子寫入流程；從舊 session 複製 NOTE/A&T 模板時使用一般檔案複製。
- 就診日期下拉顯示 `YYYY-MM-DD（NOTE 摘要）`，實際選取值仍可正確載入該日期。
- 套用模板下拉維持純日期列表，不顯示摘要。
- 可手動編輯 NOTE 與 A&T。
- 手動修改會加上 `[人類醫師_手動修改]`。
- Browse、diff、undo、redo 可用；diff 模式會分別比對 NOTE 與 A&T，未變更欄位顯示「無差異」。
- 新 session 建立時可產生歷史摘要。
- 歷史摘要 LLM prompt 會收到去識別化患者基本資料與【本次就診日期】。
- 「摘要並退出」可寫入 `note_summary` / `assessment_treatment_summary`，並在主 Agent 病歷檔案清單與日期下拉中顯示。
- 主 Agent 執行中、問診中或編輯模式中按「摘要並退出」會被阻止。
- 主 Agent 執行中、問診中或編輯模式中，切換患者、切換 session、新增/刪除 session、退出患者與 undo/redo 會被阻止。
- 摘要並退出/新增 session 產生歷史摘要期間，左欄導覽與右欄主 Agent 送出鈕一併視覺鎖定，結束後恢復。
- 「摘要並退出」作用於已載入日期、「刪除 session」作用於下拉選取日期；刪除非載入日期時保留載入中的病歷畫面。
- 歷史摘要產生失敗（未設定模型/呼叫失敗）時，新增 session 的 UI 以 ⚠️ 明示失敗而非「已產生」。
- 歷史病歷摘要/檢查與摘要並退出可各自設定獨立 API 端點與模型（`main_agent.history_summary` / `main_agent.summary_exit`），`model_name` 空白時沿用 Main Agent。
- 標準病歷模板、模型設定、教授設定三頁的變更只能在無患者載入時進行，有患者時相關按鈕 disable。
- 患者基本資料年齡顯示為 `X歲Y月`。
- 教授「建立資料庫」可重複執行而不疊加重複 chunk；建庫後可立即刪除教授而不發生檔案占用錯誤。
- Main Agent 可完成 reply。
- Main Agent 可透過 Record Subagent 更新 NOTE。
- Main Agent 同一主輪內多次成功更新病歷時，UI 版本歷史會保留每次 `update_record` 的 snapshot，而不只保留最後版。
- Record Subagent 使用原始行號套用操作，刪除/取代造成的行號位移不會影響後續操作。
- Record Subagent 遇到未知 op、`delete` / `replace` 超範圍行號或同行 delete/replace 衝突時，會整批退回並在步驟結果顯示原因；超過文末的 `insert` 會被視為文末追加。
- Hallucination Reviewer 在設定後會執行。
- Low Confidence Subagent 在設定後會標註 NOTE。
- Note Review 可觸發 Information Collection。
- Information Collection 可提問、接收回答、完成、保存並恢復 Main Agent。
- Information Collection 完成後恢復 Main Agent 時，右側即時步驟面板會重新顯示執行軌跡。
- Professor RAG 可回答並寫入 forum/RAG log。
- 可列出與讀取患者文字、圖片檔與根目錄歷史病歷 Markdown；讀取檔案後患者檔案清單會自動收回。
- 智能體互動行為可顯示當前 session 的事件。
- 手動中斷會保留已完成工作，且不套用過時結果。

安全驗收：

- NOTE 事實有來源標籤。
- 未知資訊不被寫成陰性。
- 歷史資料不被無條件沿用成今日狀態。
- 證據薄弱內容會被標註低信心。
- 明顯矛盾會在寫入前被退回或重寫。
- 幻覺審查 fail-closed：達上限未通過、迴圈耗盡或審查服務異常時病歷不寫入，主 Agent 顯示失敗而非假成功。
- 審查服務異常時不會觸發重寫風暴（只呼叫一次審查器即短路）。
- `detection_strength=0` 對照組模式可放行（不審查/不掃描），且在步驟結果與行為紀錄明確標示，可與「真正通過」區分。
- 不合法的行級登載操作不會半套寫入病歷。
- A&T 會遵守安全審核 prompt workflow：有安全性檢查教授時執行教授審查與標示；沒有時標註「未經過安全性檢查」並提醒醫師。
- 公開前敏感檔案已排除。
