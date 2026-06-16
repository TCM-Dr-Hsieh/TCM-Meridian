from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from typing import Any, Callable

from ui_app.controllers.agent_run_state_controller import get_agent_run_state
from ui_app.controllers.session_busy_guard import get_session_busy_guard


def build_professor_settings_tab(
    ui: Any,
    app_state: dict[str, Any],
    *,
    current_dir: str,
    load_config: Callable[[], dict],
    save_config: Callable[[dict], None],
    default_config: Callable[[], dict],
):
    """Build the professor management tab without depending on main-module globals."""
    from Professor import (
        load_all_professors,
        check_professor_files,
        build_professor_index,
        release_chroma_handles,
    )

    agent_run_state = get_agent_run_state(app_state)
    busy_guard = get_session_busy_guard(app_state)
    cfg = load_config()
    prof_cfg = cfg.get("professor_config", default_config().get("professor_config", {}))
    mutation_buttons: list[Any] = []

    def _patient_is_loaded() -> bool:
        return bool(app_state.get("selected_patient_folder"))

    def _sync_mutation_state():
        locked = _patient_is_loaded()
        for btn in mutation_buttons:
            if locked:
                btn.disable()
            else:
                btn.enable()
        sync_label = app_state.get("_professor_settings_lock_label")
        if sync_label:
            if locked:
                sync_label.text = "🔒 請先退出患者，才能修改教授設定或重建教授資料庫"
                sync_label.style("color: #999;")
            elif sync_label.text.startswith("🔒"):
                sync_label.text = ""

    def _register_mutation_button(btn: Any) -> Any:
        mutation_buttons.append(btn)
        if _patient_is_loaded():
            btn.disable()
        return btn

    def _reject_if_patient_loaded(status_label: Any | None = None) -> bool:
        if not _patient_is_loaded():
            return False
        message = "🔒 請先退出患者，才能修改教授設定或重建教授資料庫"
        if status_label is not None:
            status_label.set_text(message)
        else:
            ui.notify(message, type="warning")
        _sync_mutation_state()
        return True

    app_state["_sync_professor_settings_save_state"] = _sync_mutation_state

    preview_card = ui.card().classes("w-full").style(
        "padding: 12px 16px; background: #1a2233; border: 1px solid #334; margin-bottom: 12px;"
    )

    def _refresh_preview():
        preview_card.clear()
        profs = load_all_professors()
        with preview_card:
            ui.label("📋 目前教授清單（Agent 看到的 {professor_list}）").style(
                "font-size: 13px; color: #8af; font-weight: bold; margin-bottom: 6px;"
            )
            if not profs:
                ui.label("（無可用教授）").style("color: #aaa; font-size: 13px; font-family: monospace;")
            else:
                for p in profs:
                    name_str = p.get("name") or "（未命名）"
                    desc_str = p.get("description") or "（未設定風格描述）"
                    ui.label(f"  - {p['id']}：{name_str}（{desc_str}）").style(
                        "color: #cde; font-size: 13px; font-family: monospace; line-height: 1.6;"
                    )

    _refresh_preview()

    def _reject_if_busy(status_label: Any | None = None) -> bool:
        rejected = busy_guard.reject_if_busy(status_label=status_label)
        if rejected and status_label is None:
            ui.notify(busy_guard.message(), type="warning")
        return rejected

    ui.label("📚 教授管理").style("font-size: 20px; font-weight: bold; margin-bottom: 8px;")
    lock_label = ui.label("").style("font-size: 14px; line-height: 28px;")
    app_state["_professor_settings_lock_label"] = lock_label

    prof_cards_container = ui.column().classes("w-full gap-3")

    def _refresh_professor_cards():
        prof_cards_container.clear()
        profs = load_all_professors()
        if not profs:
            with prof_cards_container:
                ui.label("尚無教授。請點擊「新增教授」按鈕。").style("color: #aaa;")
            return
        with prof_cards_container:
            for p in profs:
                _build_professor_card(p)

    def _build_professor_card(prof: dict):
        prof_id = prof["id"]
        desc_path = os.path.join(current_dir, prof_id, "Description.txt")
        with ui.card().classes("w-full").style("padding: 16px;"):
            ui.label(f"🎓 {prof_id}").style("font-weight: bold; font-size: 16px; margin-bottom: 8px;")
            name_input = ui.input("教授名稱", value=prof.get("name", "")).classes("w-full")
            desc_input = ui.textarea("學術風格概述", value=prof.get("description", "")).classes("w-full")
            status_label = ui.label("").style("margin-top: 8px; color: #666;")

            with ui.row().classes("gap-2 mt-2"):

                def _save_desc(ni=name_input, di=desc_input, dp=desc_path, sl=status_label):
                    if _reject_if_patient_loaded(sl):
                        return
                    if _reject_if_busy(sl):
                        return
                    data = {"name": ni.value.strip(), "description": di.value.strip()}
                    os.makedirs(os.path.dirname(dp), exist_ok=True)
                    with open(dp, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    sl.set_text("✅ 描述已儲存")
                    _refresh_preview()

                _register_mutation_button(ui.button("💾 儲存描述", on_click=_save_desc).props("dense"))

                def _check_files(pid=prof_id, sl=status_label):
                    result = check_professor_files(pid)
                    if result["complete"]:
                        sl.set_text("✅ 所有檔案完整")
                    else:
                        missing_str = "\n".join(result["missing"])
                        sl.set_text(f"⚠️ 缺少：\n{missing_str}")

                ui.button("🔍 檢查檔案", on_click=_check_files).props("dense")

                async def _build_db(pid=prof_id, sl=status_label):
                    if _reject_if_patient_loaded(sl):
                        return
                    if _reject_if_busy(sl):
                        return
                    sl.set_text("🔨 建立資料庫中...")
                    current_cfg = load_config()
                    pc = current_cfg.get("professor_config", {})

                    def _do_build():
                        return build_professor_index(pid, pc, log_callback=lambda msg: print(msg))

                    result = await asyncio.get_event_loop().run_in_executor(None, _do_build)
                    if result["success"]:
                        agent = agent_run_state.get_agent()
                        if agent and hasattr(agent, "_professor_instances"):
                            agent._professor_instances.pop(pid, None)
                        sl.set_text(f"✅ {result['message']}")
                    else:
                        sl.set_text(f"❌ {result['message']}")

                _register_mutation_button(ui.button("🔨 建立資料庫", on_click=_build_db).props("dense"))

                def _delete_professor(pid=prof_id):
                    if _reject_if_patient_loaded(status_label):
                        return
                    if _reject_if_busy(status_label):
                        return
                    with ui.dialog() as dlg, ui.card().style("min-width: 320px;"):
                        ui.label(f"確定要刪除教授 {pid} 嗎？").style("font-size: 16px; font-weight: bold;")
                        ui.label("此操作將刪除整個教授資料夾（含知識庫、prompt、向量資料庫），無法復原！").style(
                            "color: red; font-size: 14px;"
                        )
                        with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
                            ui.button("取消", color="grey", on_click=dlg.close).props("flat")

                            def _do_delete(pid_inner=pid):
                                if _reject_if_patient_loaded(status_label):
                                    dlg.close()
                                    return
                                if _reject_if_busy(status_label):
                                    dlg.close()
                                    return
                                prof_dir = os.path.join(current_dir, pid_inner)
                                if os.path.isdir(prof_dir):
                                    # Chroma 的 SQLite/mmap 檔案可能仍被本程序持有
                                    # （Windows 上會擋住刪除），先釋放再重試刪除。
                                    release_chroma_handles()
                                    for _ in range(3):
                                        try:
                                            shutil.rmtree(prof_dir)
                                            break
                                        except PermissionError:
                                            time.sleep(0.5)
                                            release_chroma_handles()
                                    else:
                                        ui.notify(
                                            f"刪除失敗：{pid_inner} 的索引檔案被占用，"
                                            "請關閉程式後手動刪除該資料夾",
                                            type="negative",
                                        )
                                        dlg.close()
                                        _refresh_professor_cards()
                                        _refresh_preview()
                                        return
                                agent = agent_run_state.get_agent()
                                if agent and hasattr(agent, "_professor_instances"):
                                    agent._professor_instances.pop(pid_inner, None)
                                dlg.close()
                                ui.notify(f"🗑️ 已刪除 {pid_inner}", type="warning")
                                _refresh_professor_cards()
                                _refresh_preview()

                            ui.button("確認刪除", color="red", on_click=_do_delete)
                    dlg.open()

                _register_mutation_button(ui.button("🗑️ 刪除", on_click=_delete_professor, color="red").props("dense flat"))

    def _add_professor():
        if _reject_if_patient_loaded():
            return
        if _reject_if_busy():
            return
        existing = sorted(
            [
                d
                for d in os.listdir(current_dir)
                if d.startswith("professor_") and os.path.isdir(os.path.join(current_dir, d))
            ]
        )
        if existing:
            nums = []
            for e in existing:
                parts = e.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
        else:
            next_num = 1
        new_id = f"professor_{next_num:02d}"
        new_dir = os.path.join(current_dir, new_id)
        os.makedirs(os.path.join(new_dir, "doc"), exist_ok=True)

        template_dir = os.path.join(current_dir, "professor-Template")
        for fname in ["prompt_system.txt", "prompt_3_prefix.txt", "prompt_query_expansion.txt", "prompt_rerank.txt"]:
            src = os.path.join(template_dir, fname)
            dst = os.path.join(new_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            else:
                with open(dst, "w", encoding="utf-8") as f:
                    f.write("")

        with open(os.path.join(new_dir, "Description.txt"), "w", encoding="utf-8") as f:
            json.dump({"name": "", "description": ""}, f, ensure_ascii=False, indent=2)
        ui.notify(f"✅ 已新增 {new_id}", type="positive")
        _refresh_professor_cards()
        _refresh_preview()

    _register_mutation_button(ui.button("➕ 新增教授", on_click=_add_professor).props("color=primary"))
    _refresh_professor_cards()

    ui.separator().style("margin: 20px 0;")

    ui.label("⚙️ 教授共用模型設定").style("font-size: 20px; font-weight: bold; margin-bottom: 8px;")
    ui.label("所有教授共用以下模型設定").style("color: #888; margin-bottom: 12px;")

    model_inputs = {}
    model_sections = [
        ("answer", "Answer LLM（教授回答用）"),
        ("embedding", "Embedding Model（向量化 + 檢索用）"),
        ("query_expansion", "Query Expansion LLM（查詢擴展用）"),
        ("prefix", "Prefix Classification LLM（三前綴分類用）"),
        ("rerank", "Rerank LLM（重排序用）"),
    ]

    for section_key, section_label in model_sections:
        section_cfg = prof_cfg.get(section_key, {})
        with ui.expansion(section_label, icon="settings").classes("w-full"):
            inputs = {}
            inputs["api_url"] = ui.input("API URL", value=section_cfg.get("api_url", "http://localhost:1234/v1")).classes(
                "w-full"
            )
            inputs["api_key"] = ui.input("API Key", value=section_cfg.get("api_key", "lm-studio")).classes("w-full")
            inputs["model_name"] = ui.input("Model Name", value=section_cfg.get("model_name", "")).classes("w-full")
            if section_key == "answer":
                inputs["max_tokens"] = ui.number("Max Tokens", value=section_cfg.get("max_tokens", 20000)).classes(
                    "w-full"
                )
                inputs["temperature"] = ui.number(
                    "Temperature",
                    value=section_cfg.get("temperature", 0.7),
                    format="%.2f",
                    step=0.05,
                    min=0,
                    max=2,
                ).classes("w-full")
            model_inputs[section_key] = inputs

    def _save_professor_config():
        if _reject_if_patient_loaded():
            return
        if _reject_if_busy():
            return
        current_cfg = load_config()
        pc = {}
        for sk, _ in model_sections:
            inp = model_inputs[sk]
            pc[sk] = {
                "api_url": inp["api_url"].value.strip(),
                "api_key": inp["api_key"].value.strip(),
                "model_name": inp["model_name"].value.strip(),
            }
            if sk == "answer":
                pc[sk]["max_tokens"] = int(inp["max_tokens"].value or 20000)
                pc[sk]["temperature"] = float(inp["temperature"].value or 0.7)
        current_cfg["professor_config"] = pc
        save_config(current_cfg)
        agent = agent_run_state.get_agent()
        if agent:
            agent.professor_config = pc
            agent._professor_instances.clear()
        ui.notify("✅ 教授模型設定已儲存", type="positive")

    _register_mutation_button(
        ui.button("💾 儲存教授模型設定", on_click=_save_professor_config)
        .props("color=primary")
        .style("margin-top: 12px;")
    )
    _sync_mutation_state()
