from __future__ import annotations

from typing import Any, Callable

from agent_behavior_log import append_behavior_event
from ui_app.controllers.agent_run_state_controller import get_agent_run_state
from ui_app.controllers.live_steps_controller import LiveStepsController
from ui_app.controllers.interview_state_controller import get_interview_state
from ui_app.controllers.main_agent_result_processor import MainAgentResultProcessor
from ui_app.controllers.main_agent_turn_runner import MainAgentTurnRunner
from ui_app.controllers.main_chat_input_controller import MainChatInputController
from ui_app.controllers.main_chat_renderer import MainChatRenderer
from ui_app.controllers.medical_main_layout import build_medical_main_layout
from ui_app.controllers.medical_record_controller import MedicalRecordController
from ui_app.controllers.patient_session_lifecycle_controller import PatientSessionLifecycleController
from ui_app.controllers.session_busy_guard import get_session_busy_guard
from ui_app.controllers.session_state_restorer import SessionStateRestorer
from ui_app.rendering import build_conversation_text
from ui_app.services.snapshot_history import SnapshotHistory


def build_medical_main_tab(
    *,
    ui: Any,
    today: str,
    app_state: dict[str, Any],
    app_context: Any,
    load_config: Callable[[], dict],
    agent_factory: Any,
    list_patients: Callable[[], list[dict]],
    load_patient: Callable[[str], dict | None],
    save_patient_info: Callable[[str, dict], str],
    list_sessions: Callable[[str], list[str]],
    get_session_summaries: Callable[[str, str], dict],
    create_session: Callable[[str, str, str], str],
    delete_session: Callable[[str, str], str],
    load_session_content: Callable[[str, str], dict],
    save_session_content: Callable[[str, str, str, str], None],
    save_session_summaries: Callable[[str, str, str, str], str],
    write_session_log: Callable[[str, str, str], None],
    save_chat_state: Callable[[str, str, list, dict | None], None],
    load_chat_state: Callable[[str, str], dict | None],
    save_interview_state: Callable[[str, str, dict], None],
    load_interview_state: Callable[[str, str], dict | None],
    load_history_summary: Callable[[str, str], str],
    save_forum_state: Callable[[str, str, list[dict]], None],
    load_forum_state: Callable[[str, str], list[dict]],
    save_conversation_file: Callable[[str, str, str], None],
    record_snapshot_store: Any,
    get_last_visit_content: Callable[[str, str], str],
    generate_history_summary: Callable[[str, str], str],
):
    history = SnapshotHistory()
    app_state["history"] = history
    ui_state = {
        "view_mode": "browse",
        "agent_running": False,
        "chat_messages": [],
    }
    app_state["ui_state"] = ui_state

    refs = build_medical_main_layout(ui, today)

    patient_select = refs["patient_select"]
    session_select = refs["session_select"]
    template_select = refs["template_select"]
    btn_confirm_patient = refs["btn_confirm_patient"]
    btn_refresh_patients = refs["btn_refresh_patients"]
    btn_exit_patient = refs["btn_exit_patient"]
    btn_confirm_session = refs["btn_confirm_session"]
    btn_new_session = refs["btn_new_session"]
    btn_del_session = refs["btn_del_session"]
    btn_summary_exit_session = refs["btn_summary_exit_session"]
    btn_confirm_new = refs["btn_confirm_new"]
    btn_cancel_new = refs["btn_cancel_new"]
    btn_save_remark = refs["btn_save_remark"]
    btn_browse = refs["btn_browse"]
    btn_diff = refs["btn_diff"]
    btn_edit_mode = refs["btn_edit_mode"]
    btn_edit_done = refs["btn_edit_done"]
    btn_undo = refs["btn_undo"]
    btn_redo = refs["btn_redo"]
    agent_input = refs["agent_input"]
    btn_send = refs["btn_send"]
    btn_stop_agent = refs["btn_stop_agent"]
    agent_status = refs["agent_status"]
    chat_container = refs["chat_container"]
    live_steps_container = refs["live_steps_container"]
    agent_run_state = get_agent_run_state(app_state, ui_state)
    interview_state = get_interview_state(app_state)
    busy_guard = get_session_busy_guard(app_state, ui_state)
    main_chat_input = MainChatInputController(app_state=app_state, btn_send=btn_send)
    app_state["_main_chat_input"] = main_chat_input
    busy_guard.register_navigation_controls(
        [
            patient_select,
            btn_confirm_patient,
            btn_refresh_patients,
            btn_exit_patient,
            session_select,
            btn_confirm_session,
            btn_new_session,
            btn_del_session,
            btn_summary_exit_session,
            refs["new_session_date"],
            template_select,
            btn_confirm_new,
            btn_cancel_new,
            refs["remark_area"],
            btn_save_remark,
        ],
        status_label=refs["session_status"],
    )

    record_controller = MedicalRecordController(
        app_state=app_state,
        ui_state=ui_state,
        history=history,
        refs={
            "browse_container": refs["browse_container"],
            "edit_container": refs["edit_container"],
            "note_display": refs["note_display"],
            "at_display": refs["at_display"],
            "note_editor": refs["note_editor"],
            "at_editor": refs["at_editor"],
            "version_label": refs["version_label"],
            "record_status": refs["record_status"],
            "btn_browse": refs["btn_browse"],
            "btn_diff": refs["btn_diff"],
            "btn_edit_mode": refs["btn_edit_mode"],
            "btn_edit_done": refs["btn_edit_done"],
            "btn_undo": refs["btn_undo"],
            "btn_redo": refs["btn_redo"],
        },
        save_session_content=save_session_content,
        record_snapshot_store=record_snapshot_store,
        write_session_log=write_session_log,
        busy_guard=busy_guard,
    )

    def _update_display():
        record_controller.update_display()

    def _update_buttons():
        record_controller.update_buttons()

    def _save_current_to_disk():
        record_controller.save_current_to_disk()

    patient_session_controller = PatientSessionLifecycleController(
        ui=ui,
        app_state=app_state,
        ui_state=ui_state,
        app_context=app_context,
        history=history,
        refs={
            "patient_select": patient_select,
            "session_select": session_select,
            "template_select": template_select,
            "session_status": refs["session_status"],
            "new_session_panel": refs["new_session_panel"],
            "new_session_date": refs["new_session_date"],
            "btn_confirm_new": btn_confirm_new,
            "patient_info_label": refs["patient_info_label"],
            "remark_area": refs["remark_area"],
            "remark_status": refs["remark_status"],
            "browse_container": refs["browse_container"],
            "edit_container": refs["edit_container"],
            "note_display": refs["note_display"],
            "at_display": refs["at_display"],
            "note_editor": refs["note_editor"],
            "at_editor": refs["at_editor"],
            "record_status": refs["record_status"],
            "agent_status": agent_status,
        },
        today=today,
        list_patients=list_patients,
        list_sessions=list_sessions,
        get_session_summaries=get_session_summaries,
        load_patient=load_patient,
        save_patient_info=save_patient_info,
        load_session_content=load_session_content,
        save_session_summaries=save_session_summaries,
        create_session=create_session,
        delete_session=delete_session,
        write_session_log=write_session_log,
        load_config=load_config,
        record_snapshot_store=record_snapshot_store,
        generate_history_summary=generate_history_summary,
        get_session_restorer=lambda: _get_session_restorer(),
        update_display=_update_display,
        update_buttons=_update_buttons,
        render_chat=lambda: _render_chat(),
        busy_guard=busy_guard,
    )
    btn_save_remark.on_click(patient_session_controller.on_save_remark)

    def on_delete_session():
        patient_session_controller.on_delete_session()

    def _refresh_patient_list():
        patient_session_controller.refresh_patient_list()

    session_restorer_ref: list[SessionStateRestorer | None] = [None]

    def _get_session_restorer() -> SessionStateRestorer:
        if session_restorer_ref[0] is None:
            session_restorer_ref[0] = SessionStateRestorer(
                app_state=app_state,
                ui_state=ui_state,
                load_config=load_config,
                load_chat_state=load_chat_state,
                load_interview_state=load_interview_state,
                load_forum_state=load_forum_state,
                save_chat_state=save_chat_state,
                agent_factory=agent_factory,
                render_chat=_render_chat,
            )
        return session_restorer_ref[0]

    def _load_session_into_ui():
        patient_session_controller.load_session_into_ui()

    def on_confirm_patient():
        patient_session_controller.on_confirm_patient()

    def _clear_patient_ui():
        patient_session_controller.clear_patient_ui()

    def _update_patient_display():
        patient_session_controller.update_patient_display()

    app_state["_clear_patient_ui"] = _clear_patient_ui
    app_state["_refresh_patient_list"] = _refresh_patient_list
    app_state["_update_patient_display"] = _update_patient_display

    def on_exit_patient():
        patient_session_controller.on_exit_patient()

    def on_refresh_patients():
        patient_session_controller.on_refresh_patients()

    def on_confirm_session():
        patient_session_controller.on_confirm_session()

    async def on_summary_exit_session():
        await patient_session_controller.on_summary_exit_session()

    def on_new_session_click():
        patient_session_controller.on_new_session_click()

    def on_cancel_new_session():
        patient_session_controller.on_cancel_new_session()

    async def on_confirm_new_session():
        await patient_session_controller.on_confirm_new_session()

    live_steps = LiveStepsController(ui, app_state, live_steps_container, agent_status)
    chat_renderer = MainChatRenderer(ui, chat_container)
    main_agent_runner = MainAgentTurnRunner(
        app_state=app_state,
        history=history,
        load_config=load_config,
        agent_factory=agent_factory,
        get_last_visit_content=get_last_visit_content,
        load_history_summary=load_history_summary,
        write_session_log=write_session_log,
        save_chat_state=save_chat_state,
        save_forum_state=save_forum_state,
    )

    def _render_chat():
        chat_renderer.render(ui_state["chat_messages"])

    app_state["_render_chat"] = _render_chat

    main_agent_result_processor = MainAgentResultProcessor(
        app_state=app_state,
        ui_state=ui_state,
        history=history,
        agent_run_state=agent_run_state,
        live_steps=live_steps,
        main_chat_input=main_chat_input,
        agent_status=agent_status,
        save_current_to_disk=_save_current_to_disk,
        update_display=_update_display,
        update_buttons=_update_buttons,
        render_chat=_render_chat,
        write_session_log=write_session_log,
        save_chat_state=save_chat_state,
        save_forum_state=save_forum_state,
        save_conversation_file=save_conversation_file,
        record_snapshot_store=record_snapshot_store,
    )

    async def on_send():
        user_msg = agent_input.value or ""
        if not user_msg.strip():
            return

        if busy_guard.reject_if_busy(status_label=agent_status):
            return

        fp = app_state["selected_patient_folder"]
        dt = app_state["selected_session_date"]
        if not fp or not dt:
            agent_status.text = "⚠️ 請先選取患者與就診日期"
            return

        agent_input.value = ""
        ui_state["chat_messages"].append({"role": "user", "content": user_msg})
        _render_chat()
        agent_status.text = "🤖 AI 思考中..."
        agent_run_state.start_run()
        main_chat_input.lock("agent_running")
        btn_stop_agent.enable()
        live_steps.start()

        conversation_text = build_conversation_text(ui_state["chat_messages"])
        write_session_log(fp, dt, f"[USER_MSG] {user_msg}")

        try:
            result, gen = await main_agent_runner.run(
                user_msg=user_msg,
                fp=fp,
                dt=dt,
                conversation_text=conversation_text,
            )
            await _process_agent_result_inner(result, fp, dt, gen)
        except Exception as e:
            error_msg = "服務已中斷(模型呼叫失敗)"
            agent_inst = agent_run_state.get_agent()
            if agent_inst and getattr(agent_inst, "_suspended", None):
                current = history.get_current()
                note_content = current["note"] if current else ""
                at_content = current["at"] if current else ""
                agent_inst.finalize_suspended_turn(
                    reply_text=error_msg,
                    interrupted_step_result="服務已中斷(模型呼叫失敗)",
                    note_content=note_content,
                    at_content=at_content,
                )
            ui_state["chat_messages"].append({
                "role": "agent",
                "content": error_msg,
                "steps": [],
            })
            _render_chat()
            agent_status.text = error_msg
            conversation_text = build_conversation_text(ui_state["chat_messages"])
            save_conversation_file(fp, dt, conversation_text)
            agent_st = agent_inst.export_state() if agent_inst else None
            save_chat_state(fp, dt, ui_state["chat_messages"], agent_st)
            if agent_inst and getattr(agent_inst, "forum_history", None):
                save_forum_state(fp, dt, agent_inst.forum_history)
            write_session_log(fp, dt, f"[ERROR] {error_msg}: {e}")
            append_behavior_event(
                fp,
                dt,
                agent="main_agent",
                event_type="model_error",
                label="模型呼叫失敗",
                title="AI主治醫師模型呼叫失敗",
                content=str(e),
                severity="error",
            )
            behavior_render = app_state.get("render_agent_behavior")
            if behavior_render:
                behavior_render()
        finally:
            btn_stop_agent.disable()
            if interview_state.is_active():
                agent_run_state.finish_run()
                main_chat_input.unlock("agent_running")
                main_chat_input.lock("interview_active")
            else:
                agent_run_state.finish_run()
                main_chat_input.unlock("agent_running")
            live_steps.stop()

    btn_confirm_patient.on_click(on_confirm_patient)
    btn_refresh_patients.on_click(on_refresh_patients)
    btn_exit_patient.on_click(on_exit_patient)
    btn_confirm_session.on_click(on_confirm_session)
    btn_new_session.on_click(on_new_session_click)
    btn_cancel_new.on_click(on_cancel_new_session)
    btn_confirm_new.on_click(on_confirm_new_session)
    btn_del_session.on_click(on_delete_session)
    btn_summary_exit_session.on_click(on_summary_exit_session)

    btn_browse.on_click(record_controller.on_browse)
    btn_diff.on_click(record_controller.on_diff)
    btn_edit_mode.on_click(record_controller.on_edit_mode)
    btn_edit_done.on_click(record_controller.on_edit_done)
    btn_undo.on_click(record_controller.on_undo)
    btn_redo.on_click(record_controller.on_redo)

    def on_stop_agent():
        agent_inst = agent_run_state.get_agent()
        if not agent_inst or not getattr(agent_inst, "request_manual_stop", None):
            agent_status.text = "目前沒有可中斷的主 Agent 執行"
            btn_stop_agent.disable()
            return

        requested = agent_inst.request_manual_stop()
        if requested:
            agent_status.text = "⏹ 已收到中斷請求，等待目前動作完成..."
            btn_stop_agent.disable()
        else:
            agent_status.text = "目前沒有可中斷的主 Agent 執行"
            btn_stop_agent.disable()

    async def _process_agent_result_inner(result: dict, fp: str, dt: str, gen: int = -1):
        await main_agent_result_processor.process(result, fp, dt, gen)

    app_state["_process_agent_result"] = _process_agent_result_inner
    app_state["_btn_send"] = btn_send
    app_state["_btn_stop_agent"] = btn_stop_agent
    app_state["_agent_status"] = agent_status

    btn_send.on_click(on_send)
    btn_stop_agent.on_click(on_stop_agent)
    _refresh_patient_list()
    _update_buttons()
