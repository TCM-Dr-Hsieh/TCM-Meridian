# TCM-Meridian

[Traditional Chinese](README.md) | English

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20725779.svg)](https://doi.org/10.5281/zenodo.20725779)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

TCM-Meridian, also named Xinglin Jingwei, is a NiceGUI-based clinical AI workstation for Traditional Chinese Medicine. It integrates patient data management, visit-note editing, multimodal patient-file reading, a ReAct-style Main Agent, multiple safety and interview subagents, and a traceable Professor RAG consultation workflow.

The project is designed as a physician-led clinical support system. AI can help with interviewing, record updates, uncertainty checks, knowledge-base consultation, and behavior tracing, but final diagnosis, prescription, and treatment decisions must remain with a qualified clinician.

## Important Notice

This system is intended only for clinical decision support and documentation assistance. It is not a medical device, does not replace physician diagnosis, prescriptions, or treatment decisions, and must not be used for unsupervised automated care. Non-clinicians should not treat system output as a basis for self-diagnosis, treatment, or medication use; any such use is at the user's own risk.

Before public release or deployment, carefully check:

- `config.json` may contain real API keys.
- `patient_data/` may contain patient identifiers, medical records, images, and clinical logs.
- `professor_*/doc/` may contain private or copyrighted knowledge-base material. Vector indexes (`chroma_doc_index/`) and `parent_map.jsonl` are excluded by `.gitignore`.
- This repository includes `.gitignore` rules for `config.json` and real patient data, and a key-free `config.example.json`. Copy `config.example.json` to `config.json`, fill in your own API keys, and always run `git status` after the first `git add` to confirm that secrets and real patient data are not tracked.

## Citation

If you use this project in research or derivative work, please cite:

> Hsieh, H.-W. (2026). *TCM-Meridian (杏林經緯): A multi-agent, safety-oriented AI clinical assistant for Traditional Chinese Medicine* (v1.1.2). Zenodo. https://doi.org/10.5281/zenodo.20725779

You may also use the **Cite this repository** button on the GitHub repository page to obtain APA or BibTeX metadata generated from [`CITATION.cff`](CITATION.cff). DOI `10.5281/zenodo.20725779` is the concept DOI and always points to the latest version.

## Features

- **Clinical workflow**: file-based patient and visit-session management, a three-column workstation, record browsing/editing, and snapshot version control with undo / redo / diff. Version history is persisted in the session log folder, so diff/undo/redo continue after exiting and reloading the same visit date. Overwritten redo branches and external file states are kept as audit events. Human manual edits are marked with `[人類醫師_手動修改]` without duplicate stacking on the same line, and visit dates include a 50-character NOTE summary index.
- **Multi-agent collaboration**: a ReAct-style Main Agent coordinates record registration, hallucination review, low-confidence annotation, note review, interview assistance, and Professor RAG through strict JSON contracts. Professor RAG follows query expansion -> prefix classification -> retrieval -> RRF -> rerank -> answer. The system can read patient images and text files, and records agent behavior in a timeline. Record-version evolution is injected into the Main Agent and relevant subagent prompts as a plain-text "medical record diff history", allowing agents to see which step caused each version change.
- **Safety-oriented design**: fail-closed hallucination review, no write on review failure or service error, atomic line-level record operations with all-or-nothing validation, source attribution, cooperative interruption, busy-state locks, and global-setting locks. A research control mode is available by setting detection strength to `0`.
- **Engineering robustness**: atomic writes for important files, session-state save/restore for chat/interview/forum/RAG traces/behavior logs, and independently configurable API endpoints for Main Agent submodels.

## Architecture

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
  record_edit_tags.py
  record_diff_context.py
  ui_app/services/llm_config_resolver.py
  ui_app/services/record_snapshot_store.py

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

## Project Structure

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
├── record_edit_tags.py
├── record_diff_context.py
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
├── professor_01/              # Demo professor: custom herbal safety list (pregnancy / kidney disease contraindications)
└── professor_02/              # Demo professor: Yi Zong Jin Jian, public-domain classic
```

`professor_01` and `professor_02` are bundled demo professors. They include raw `doc/` knowledge-base files but do not include prebuilt vector indexes, because indexes are build artifacts excluded by `.gitignore`. After cloning, open the Professor Settings tab and click "Build Database" to create indexes. You can also add, delete, or rebuild professors.

## Requirements

- Python 3.10 or later is recommended.
- The Main Agent and subagents require an OpenAI-compatible chat API.
- Professor RAG requires an OpenAI-compatible embedding endpoint.
- LM Studio, OpenRouter, or other compatible services can be used.

Install dependencies:

```bash
pip install -r requirements.txt
```

Main packages:

- `nicegui`
- `openai`
- `chromadb`
- `langchain`
- `langchain-community`
- `numpy`
- `requests`

## Quick Start

1. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

2. Copy the configuration template and fill in your API endpoint, key, and model names. These can also be edited later in the Model Settings and Professor Settings tabs.

   ```bash
   cp config.example.json config.json
   ```

3. Start the application.

   ```bash
   python TCM_Meridian_main.py
   ```

4. Open the browser.

   ```text
   http://localhost:8080
   ```

The current application binds to `0.0.0.0:8080` and starts with NiceGUI reload mode enabled.

## Configuration

`config.json` contains shared settings for the Main Agent, subagents, and Professor RAG. On first use, copy `config.example.json` to `config.json` and fill in your own endpoint/key. `config.json` is excluded by `.gitignore`. If configuration loading fails, the system prints a `WARNING` to the console and falls back to defaults; configuration saves use atomic writes.

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

Common fields include `api_url`, `api_key`, `model_name`, `max_tokens`, and `temperature`.

Advanced fields include `main_agent.max_sub_turns`, `main_agent.history_summary_review_rounds`, `hallucination_subagent.max_review_rounds`, `ic_subagent.max_collection_rounds`, `lc_subagent.max_scan_rounds`, and detection-strength settings. Most of these can be adjusted from the Model Settings page. Missing values fall back to program defaults.

The Main Agent submodels `main_agent.history_summary` and `main_agent.summary_exit` are full configuration dictionaries and may point to different API endpoints and models from the Main Agent. Empty `model_name` values fall back to legacy flat keys and then to `main_agent.model_name`. Empty `api_url` / `api_key` values fall back to the Main Agent. If `history_summary.max_tokens` is not set, it falls back to the Main Agent `max_tokens`; `summary_exit.max_tokens` defaults to `128`; temperatures default to `0.5` and `0.2` respectively. The Model Settings page does not automatically prefill submodel `model_name` fields with the Main Agent model, preventing inherited models from being accidentally persisted as dedicated submodel settings.

Detection strengths (`hallucination_subagent.detection_strength`, `lc_subagent.detection_strength`) set to `0` enable research control mode, meaning the corresponding check is bypassed. Values greater than `0` are clamped to the valid range. The Model Settings page blocks empty values, negative values, and detection strengths larger than their maximum review/scan rounds.

Professor RAG is configured separately for answer generation, embedding, query expansion, three-prefix classification, and reranking models.

## UI Tabs

The application currently has 10 top-level tabs:

1. Patient Registration
2. Medical System Main Interface
3. Image Query Area
4. Medical Information File Storage
5. Medical Q&A Discussion Area
6. Automatic Interview Dialogue Area
7. Model Settings
8. Professor Settings
9. Standard Record Template Settings
10. Agent Behavior Timeline

## Patient Data Format

Each patient is represented as a folder:

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
        ├── <date>-information-collection-dialogue.txt
        ├── <date>-forum-state.json
        ├── <date>-forum.txt
        ├── <date>-Human-Agent-Interaction.md
        ├── <date>-History-Summary.md
        ├── <date>-RAG-full-behavior.txt
        ├── <date>-agent-behavior.jsonl
        ├── <date>-record-snapshots.json
        └── <date>-record-snapshot-events.jsonl
```

`Picture_Row/` stores patient images, and `Medical_information/` stores text reports and medical information. The patient root folder stores NOTE and ASSESSMENT & TREATMENT Markdown files for each visit date. The Main Agent can list and read these files through actions. Historical record lists show `note_summary` or `assessment_treatment_summary`; if "summarize and exit" has not been run, the first 50 characters of the file are used as a fallback summary.

The date selector in the main interface shows NOTE summaries, such as `2026-06-11 (headache, sore throat)`, to help identify sessions. The template selector remains a pure date list.

"Summarize and exit" exits only the currently loaded visit date, not the patient. "Delete session" acts on the date selected in the dropdown and does not require that date to be loaded. Busy-state locks prevent switching patients/sessions, creating/deleting sessions, exiting patients, or summarizing while the Main Agent is running, interview flow is active, or the record editor is still in edit mode. If the summary model is not configured or the call fails, the system falls back to the first 50 characters of the original content.

Historical summary generation and review/rewrite share the same submodel (`main_agent.history_summary`). If summary generation fails, the UI explicitly reports failure instead of pretending success.

## Agent Workflow

After receiving a physician message, the Main Agent enters a bounded ReAct loop. Every sub-turn must output JSON:

```json
{
  "thinking": "...",
  "action": "reply",
  "action_input": "...",
  "next_step": "..."
}
```

Supported actions:

- `reply`
- `update_record`
- `information_collection_subagent`
- `low_confidence_check`
- `note_review_subagent`
- `call_professor`
- `list_patient_files`
- `read_patient_file`

`list_patient_files` provides filenames and historical record summaries as an index. The complete list is stored only temporarily in the Main Agent context under the patient-file-list section. After `read_patient_file` completes, the list is removed to avoid long-term token accumulation. Summaries are only for file identification; to cite or reason from a file, the Main Agent must read the original content.

The Main Agent and Record/Hallucination/Low-Confidence/Note-Review subagents receive a plain-text `medical record diff history` block. This block lists NOTE/A&T changes version by version up to the currently selected version (`snapshots[:current_index+1]`), excluding redo branches that were undone. Version sources include labels such as `init`, `人類修改`, `Turn 2-1 update_record`, and `Turn 2-2 low_confidence_check`. Therefore, "who changed what and at which step" is represented by the diff source. The system no longer adds inline labels for AI modifications; only human manual edits are still marked with `[人類醫師_手動修改]`, without duplicate stacking on the same line.

When manually editing records, try to preserve existing source labels in `[ ... ]` instead of deleting them while rewriting the sentence. These labels are important for hallucination review, low-confidence annotation, diff history, and audit tracing. Removing source labels makes later provenance checks harder.

If the current turn updates NOTE, the prompt requires `low_confidence_check` before the final reply unless the physician explicitly forbids it. If the current turn writes A&T, the prompt requires the safety flow before finalization. If a safety-check professor exists, the Main Agent must ask that professor to review the current A&T; low safety must be revised and rechecked, while medium/high safety must be marked at the top of A&T. If no safety-check professor exists, A&T must begin with `##注意，此分析未經過安全性檢查。`, and the final reply must remind the physician.

`update_record` is applied by the Record Subagent through line-level operations. All operation line numbers refer to the original lines visible to the model. The program validates the whole batch first, then rebuilds the content once. `insert` beyond the end appends to the end. Out-of-range `delete` / `replace`, unknown operations, or duplicate delete/replace operations on the same line reject the whole batch and leave the record unchanged.

When a hallucination-review model is configured, the Hallucination Reviewer runs before writing. Passing requires accumulating `detection_strength` agree results across review rounds and rewritten versions. The review is fail-closed: if review reaches `max_review_rounds`, the loop is exhausted, or the reviewer service fails, the record is not written, the original NOTE/A&T is returned, and `review_result` is marked as failed. The Main Agent must not claim that the update succeeded. If the reviewer service fails, the system short-circuits after one call instead of repeatedly rewriting and burning tokens. If `detection_strength = 0`, review is skipped as research control mode and explicitly marked as such.

## Professor RAG

Each professor is a folder:

```text
professor_XX/
├── Description.txt
├── doc/
├── prompt_system.txt
├── prompt_3_prefix.txt
├── prompt_query_expansion.txt
├── prompt_rerank.txt
├── chroma_doc_index/    # generated after database build; not distributed
└── parent_map.jsonl     # generated after database build; not distributed
```

Suggested `Description.txt` format:

```json
{
  "name": "Professor name",
  "description": "Professor expertise and answer style"
}
```

The Professor Settings tab can add professors, edit descriptions, check files, configure models, and build Chroma indexes. Adding a professor copies prompt templates from `professor-Template/`. "Build Database" clears old indexes before rebuilding, preventing duplicated chunks. After a successful build, the Main Agent cache for that professor is cleared so the new index takes effect immediately. On Windows, Chroma file handles are released before deletion or rebuild to avoid file-lock errors. Professor-page operations that change global state require exiting the current patient first; file checking is read-only and not restricted.

### Knowledge-Base File Format (`doc/`)

Each professor knowledge base uses plain `.txt` files stored under `professor_XX/doc/`. Multiple files are allowed.

This RAG uses **Parent-Child Chunking**, also known as Parent Document Retrieval. During indexing, each parent segment is split into overlapping child chunks stored in the vector database. Retrieval matches child chunks, maps them back to their parent segments, and feeds the parent segments to the LLM. Therefore, the parent segment is the basic unit ultimately supplied to the LLM after retrieval.

Parent-segment rules:

- Human expert segmentation is recommended. Each parent segment should be a continuous medical text unit that is difficult to split further. Recommended length is 1,000-3,000 Chinese characters, but longer segments may be kept intact if their internal logic is tightly connected.
- Parent segments in `.txt` files are separated by two or more blank lines.
- The first word of every parent segment must be a prefix representing its category. This supports subset retrieval through the three-prefix classification path.

Available prefixes:

| Prefix | Covered content |
| --- | --- |
| `case` | TCM case records, diagnosis, pathomechanism, pattern, treatment principles, prescriptions, acupuncture, case analysis, prognosis. |
| `formula` | Formula composition, effects, indications, pathomechanism, formulation logic, compatibility, modifications, administration. |
| `herb` | Materia medica, source, properties/flavors, channel entry, ascending/descending/floating/sinking, effects, indications, clinical use, processing, contraindications. |
| `acupuncture` | Acupuncture theory, channels, points, needling, moxibustion, acupuncture treatment. |
| `diagnoses` | TCM diagnostics, four examinations, eight principles, qi/blood, zang-fu, six excesses, phlegm/food, Shanghan/Wenbing differentiation. |
| `treatment` | Treatment principles, preventive treatment, root treatment, yin-yang adjustment, supporting upright qi, urgency/priority, regular/contrary treatment, same disease different treatments, different diseases same treatment, three-cause adaptation, eight methods. |
| `disease-Internal` | TCM internal medicine and modern digestive, circulatory, urinary, respiratory, and endocrine diseases. |
| `disease-Obstetrics&Gynecology` | TCM gynecology, theory, etiology/pathogenesis, diagnosis, treatment, menstruation/leukorrhea/pregnancy/postpartum diseases. |
| `disease-Pediatrics` | TCM pediatrics and common pediatric diseases. |
| `disease-Osteology&Traumatology` | Bone, joint, muscle, tendon diseases and injuries; internal and external treatment. |
| `disease-Surgery` | TCM surgery, sores/ulcers, goiter/tumors, breast disease, skin disease, anorectal disease, male surgical disease, miscellaneous disorders. |
| `disease-Dermatology` | TCM dermatology and pattern-based treatment of skin diseases. |
| `disease-Eye&ENT` | TCM ophthalmology and ENT diseases. |
| `theory` | TCM theory: yin-yang, five phases, zangxiang, qi/blood/fluids, channels, etiology/pathogenesis, prevention and treatment principles. |
| `classic` | TCM classics: Huangdi Neijing, Nanjing, Shanghan Lun, Jingui Yaolue, Wenbing, and related classical material. |
| `others` | Use when none of the above categories apply. |

The bundled `professor_02` demo knowledge base uses Yi Zong Jin Jian, a public-domain classic. Its parent segmentation and category prefixes were manually edited by the author and are not automatic output.

## Role Customization

Most agent behavior is driven by external prompt files. In many cases, behavior can be adjusted by editing the relevant prompt section without changing code. A practical workflow is edit -> test -> inspect behavior -> refine.

### 1. Main Agent Working Habits

Edit the **`═══ 人類醫師的使用習慣 ═══`** section in `prompt_main_agent.txt` to describe your personal or clinic workflow, such as when the agent should proactively interview, how detailed records should be, which files should be read first, and when professors should be consulted.

After editing, send several test instructions and inspect the **Agent Behavior Timeline** tab to confirm whether step-by-step decisions match your intended habit. If not, refine this section and test again.

### 2. Professor Academic Style

Each `professor_XX/prompt_system.txt` contains a **`## 教授角色設定與學術風格`** section. Use it to define the professor's academic background, specialty, reasoning style, and voice, such as classical-formula school vs. later-formula school, preference for classical citations vs. clinical evidence, and typical wording.

The professor name and one-sentence style description are separately configured in `professor_XX/Description.txt`, which is displayed in the professor list and injected into the `{description}` placeholder. The `## 教授角色設定與學術風格` section is for richer role details.

## Safety Checklist

Before public release, at minimum exclude:

```gitignore
config.json
patient_data/
**/log/
**/__pycache__/
professor_*/chroma_doc_index/
professor_*/parent_map.jsonl
```

Exclude `professor_*/doc/` as well if it contains private or copyrighted material. If an API key was ever committed or shared, rotate it immediately.

Patient demographic data sent into LLM prompts is de-identified first: names keep the first character and mask the rest with "某"; single-character names receive at least one "某"; IDs are not output; birth dates are shown as `YYYY-XX-XX (X years Y months)` with month/day masked and age retained; sex, visit date, and notes are kept. However, NOTE, A&T, patient notes, read files, logs, and professor knowledge bases may still contain identifiable information and require human review before public release or use with external models.

## Development Notes

- `TCM_Meridian_main.py` should remain the startup and assembly layer.
- UI behavior belongs in `ui_app/controllers/`.
- File and state operations belong in `ui_app/services/`.
- When adding a new Main Agent action, update `prompt_main_agent.txt`, `Main_Agent.py`, UI persistence, README, and SPEC together.
- Any workflow that modifies records should preserve source attribution, pre-write review, and post-write low-confidence annotation.

## Test Status

The project currently has no standalone automated test directory. See `SPEC.md` for the recommended manual acceptance checklist.

## License

Copyright 2026 Hong-Wen Hsieh

This project is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE). Before formal public distribution, re-check the licenses and redistributability of third-party data, patient data, and professor knowledge-base content.
