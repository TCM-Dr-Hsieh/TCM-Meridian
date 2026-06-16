"""
TCM_Meridian_main.py - 杏林經緯 (TCM-Meridian) 主介面
框架：NiceGUI
架構：多分頁式介面，雙軌並行病歷儲存系統 (JSON + Markdown)

分頁：
  1. 患者登錄          — 創建 / 修改患者基本資料
  2. 醫療系統主介面     — 三欄式看診介面
  3. 影像檔查詢區       — (先留白)
  4. 醫療資訊檔案存放區 — (先留白)
  5. 醫療問答討論區     — (先留白)
  6. 自動問診對話區     — 問診助理 Subagent 對話介面
  7. 模型設定           — API 與模型名稱設定
  8. 教授設定           — (先留白)
  9. 標準病歷模板設定   — (先留白)
"""
from __future__ import annotations

import os
from datetime import datetime

from nicegui import ui
from ui_app.context import AppContext
from ui_app.controllers.auto_interview_controller import AutoInterviewController
from ui_app.controllers.agent_behavior_controller import build_agent_behavior_tab
from ui_app.controllers.forum_controller import build_forum_tab
from ui_app.controllers.image_controller import build_image_tab
from ui_app.controllers.medinfo_controller import build_medinfo_tab
from ui_app.controllers.medical_main_controller import build_medical_main_tab as build_medical_main_tab_controller
from ui_app.controllers.model_settings_controller import build_model_settings_tab
from ui_app.controllers.patient_controller import build_patient_registration_tab
from ui_app.controllers.professor_controller import build_professor_settings_tab
from ui_app.controllers.tab_controller import TabController
from ui_app.controllers.template_controller import build_record_template_tab
from ui_app.services.config_service import load_json_config, save_json_config
from ui_app.services.app_services import create_app_services
from ui_app.shell import TabSpec, build_header, build_tab_shell

# ════════════════════════════════════════════════════════════════
# 全域設定
# ════════════════════════════════════════════════════════════════
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(CURRENT_DIR, "patient_data")
CONFIG_PATH = os.path.join(CURRENT_DIR, "config.json")
os.makedirs(DATA_ROOT, exist_ok=True)

TODAY = datetime.now().strftime("%Y-%m-%d")
RECORD_TEMPLATE_PATH = os.path.join(CURRENT_DIR, "Record_Template.txt")

# ════════════════════════════════════════════════════════════════
# 全域應用狀態 (跨分頁共享)
# ════════════════════════════════════════════════════════════════
app_context = AppContext()
app_state = app_context.state
_auto_interview_controller: AutoInterviewController | None = None

# ════════════════════════════════════════════════════════════════
# 設定檔管理 (config.json)
# ════════════════════════════════════════════════════════════════

def _default_config() -> dict:
    return {
        "main_agent": {
            "api_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "model_name": "",
            "history_summary_model_name": "",
            "summary_exit_model_name": "",
            "max_tokens": 4000,
            "temperature": 0.7,
            "history_summary": {
                "api_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model_name": "",
                "max_tokens": 4000,
                "temperature": 0.5,
            },
            "summary_exit": {
                "api_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model_name": "",
                "max_tokens": 128,
                "temperature": 0.2,
            },
        },
        "record_subagent": {
            "api_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "model_name": "",
            "max_tokens": 8000,
            "temperature": 0.7,
        },
        "hallucination_subagent": {
            "api_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "model_name": "",
            "max_tokens": 8000,
            "temperature": 1.0,
            "detection_strength": 2,
            "max_review_rounds": 5,
        },
        "ic_subagent": {
            "api_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "model_name": "",
            "max_tokens": 20000,
            "temperature": 0.7,
            "max_collection_rounds": 10,
        },
        "lc_subagent": {
            "api_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "model_name": "",
            "max_tokens": 20000,
            "temperature": 1.0,
            "max_scan_rounds": 8,
            "detection_strength": 4,
        },
        "nr_subagent": {
            "api_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "model_name": "",
            "max_tokens": 20000,
            "temperature": 1.0,
        },
        "professor_config": {
            "answer": {
                "api_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model_name": "",
                "max_tokens": 20000,
                "temperature": 0.7,
            },
            "embedding": {
                "api_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model_name": "",
            },
            "query_expansion": {
                "api_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model_name": "",
            },
            "prefix": {
                "api_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model_name": "",
            },
            "rerank": {
                "api_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model_name": "",
            },
        },
    }


def load_config() -> dict:
    return load_json_config(CONFIG_PATH, _default_config)


def save_config(cfg: dict):
    save_json_config(CONFIG_PATH, cfg)


# ════════════════════════════════════════════════════════════════
# 患者資料管理核心 (雙軌並行儲存)
# ════════════════════════════════════════════════════════════════
app_services = create_app_services(
    data_root=DATA_ROOT,
    record_template_path=RECORD_TEMPLATE_PATH,
    load_config=load_config,
)
agent_factory = app_services.agent_factory

create_patient = app_services.patient_data.create_patient
list_patients = app_services.patient_data.list_patients
load_patient = app_services.patient_data.load_patient
save_patient_info = app_services.patient_data.save_patient_info
update_patient_basic_info = app_services.patient_data.update_patient_basic_info
delete_patient = app_services.patient_data.delete_patient
list_sessions = app_services.patient_data.list_sessions
get_session_summaries = app_services.patient_data.get_session_summaries
create_session = app_services.patient_data.create_session
delete_session = app_services.patient_data.delete_session
load_session_content = app_services.patient_data.load_session_content
save_session_content = app_services.patient_data.save_session_content
save_session_summaries = app_services.patient_data.save_session_summaries

write_session_log = app_services.session_artifacts.write_session_log
save_chat_state = app_services.session_artifacts.save_chat_state
load_chat_state = app_services.session_artifacts.load_chat_state
save_interview_state = app_services.session_artifacts.save_interview_state
load_interview_state = app_services.session_artifacts.load_interview_state
load_history_summary = app_services.session_artifacts.load_history_summary
save_forum_state = app_services.session_artifacts.save_forum_state
load_forum_state = app_services.session_artifacts.load_forum_state
save_conversation_file = app_services.session_artifacts.save_conversation_file

get_last_visit_content = app_services.history_context.get_last_visit_content
generate_history_summary = app_services.history_context.generate_history_summary

load_record_template = app_services.templates.load_record_template
save_record_template = app_services.templates.save_record_template


# ════════════════════════════════════════════════════════════════
# NiceGUI 介面
# ════════════════════════════════════════════════════════════════

# 自訂 CSS
ui.add_head_html("""
<style>
    :root {
        --primary: #2d6a4f;
        --primary-light: #40916c;
        --primary-dark: #1b4332;
        --accent: #95d5b2;
        --bg-main: #f5f7f5;
        --bg-card: #ffffff;
        --text-primary: #1b1b1b;
        --text-secondary: #555;
        --border: #d8e4d8;
        --danger: #d32f2f;
        --warning: #f9a825;
    }
    body {
        background: var(--bg-main) !important;
        font-family: "Microsoft JhengHei", "Noto Sans TC", "Segoe UI", sans-serif !important;
    }
    .q-tab__label {
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    .patient-form-card, .card-panel {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        background: var(--bg-card);
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .patient-list-card {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        background: var(--bg-card);
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: var(--primary-dark);
        margin-bottom: 12px;
    }
    .placeholder-tab {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 400px;
        color: #aaa;
        font-size: 20px;
    }
    .chat-bubble-user {
        background: #e8f5e9;
        border-radius: 12px 12px 4px 12px;
        padding: 12px 16px;
        margin: 4px 0;
        max-width: 100%;
    }
    .chat-bubble-agent {
        background: #f3f3f3;
        border-radius: 12px 12px 12px 4px;
        padding: 12px 16px;
        margin: 4px 0;
        max-width: 100%;
    }
    .step-detail-box {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #555;
        margin: 4px 0 8px 0;
    }
    .mode-btn-active {
        font-weight: 700 !important;
    }
</style>
""")


def _tab_controllers() -> list[TabController]:
    """Top-level tab controller registry. Builders retain existing button behavior."""
    return [
        TabController(
            "patient",
            "患者登錄",
            lambda: build_patient_registration_tab(
                ui,
                app_state,
                app_context.reset_patient_selection,
                list_patients=list_patients,
                load_patient=load_patient,
                create_patient=create_patient,
                update_patient_basic_info=update_patient_basic_info,
                delete_patient=delete_patient,
            ),
            tab_style="color: var(--primary-dark);",
        ),
        TabController("main", "醫療系統主介面", _build_medical_main_tab, panel_classes="q-pa-sm"),
        TabController("image", "影像檔查詢區", lambda: build_image_tab(ui, app_state, TODAY)),
        TabController("medinfo", "醫療資訊檔案存放區", lambda: build_medinfo_tab(ui, app_state, TODAY)),
        TabController("qa", "醫療問答討論區", lambda: build_forum_tab(ui, app_state)),
        TabController("auto", "自動問診對話區", lambda: get_auto_interview_controller().build_tab()),
        TabController("model", "模型設定", lambda: build_model_settings_tab(ui, app_state, load_config, save_config, app_context.reset_agent_state)),
        TabController(
            "professor",
            "教授設定",
            lambda: build_professor_settings_tab(
                ui,
                app_state,
                current_dir=CURRENT_DIR,
                load_config=load_config,
                save_config=save_config,
                default_config=_default_config,
            ),
        ),
        TabController(
            "template",
            "標準病歷模板設定",
            lambda: build_record_template_tab(
                ui,
                app_state,
                load_record_template,
                save_record_template,
                app_context.reset_agent_state,
            ),
        ),
        TabController("agent_behavior", "智能體互動行為", lambda: build_agent_behavior_tab(ui, app_state), panel_classes="q-pa-sm"),
    ]


def _tab_specs() -> list[TabSpec]:
    return [controller.spec() for controller in _tab_controllers()]


def get_auto_interview_controller() -> AutoInterviewController:
    global _auto_interview_controller
    if _auto_interview_controller is None:
        _auto_interview_controller = AutoInterviewController(
            ui,
            app_state,
            load_config=load_config,
            write_session_log=write_session_log,
            save_chat_state=save_chat_state,
            save_interview_state=save_interview_state,
            save_forum_state=save_forum_state,
            save_conversation_file=save_conversation_file,
            get_last_visit_content=get_last_visit_content,
            load_history_summary=load_history_summary,
            agent_factory=agent_factory,
        )
    return _auto_interview_controller


def build_ui():
    """建構整個 NiceGUI 介面"""
    build_header(ui, TODAY)
    build_tab_shell(ui, app_state, _tab_specs())


# ════════════════════════════════════════════════════════════════
# 分頁 2：醫療系統主介面 (三欄式)
# ════════════════════════════════════════════════════════════════

def _build_medical_main_tab():
    """醫療系統主介面：左欄(導覽) + 中欄(病歷) + 右欄(Agent 互動)"""
    build_medical_main_tab_controller(
        ui=ui,
        today=TODAY,
        app_state=app_state,
        app_context=app_context,
        load_config=load_config,
        agent_factory=agent_factory,
        list_patients=list_patients,
        load_patient=load_patient,
        save_patient_info=save_patient_info,
        list_sessions=list_sessions,
        get_session_summaries=get_session_summaries,
        create_session=create_session,
        delete_session=delete_session,
        load_session_content=load_session_content,
        save_session_content=save_session_content,
        save_session_summaries=save_session_summaries,
        write_session_log=write_session_log,
        save_chat_state=save_chat_state,
        load_chat_state=load_chat_state,
        save_interview_state=save_interview_state,
        load_interview_state=load_interview_state,
        load_history_summary=load_history_summary,
        save_forum_state=save_forum_state,
        load_forum_state=load_forum_state,
        save_conversation_file=save_conversation_file,
        get_last_visit_content=get_last_visit_content,
        generate_history_summary=generate_history_summary,
    )


def run_app():
    """Start the NiceGUI application."""
    build_ui()
    ui.run(
        title="杏林經緯 TCM-Meridian",
        host="0.0.0.0",
        port=8080,
        reload=True,
        favicon="🌿",
    )


# ════════════════════════════════════════════════════════════════
# 主程式入口
# ════════════════════════════════════════════════════════════════
if __name__ in {"__main__", "__mp_main__"}:
    run_app()
