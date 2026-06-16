from __future__ import annotations

from typing import Any, Callable

from ui_app.controllers.agent_run_state_controller import get_agent_run_state
from ui_app.controllers.interview_state_controller import get_interview_state
from ui_app.rendering import format_interview_conversations


class SessionStateRestorer:
    """Restore persisted per-session UI/agent state into app_state."""

    def __init__(
        self,
        *,
        app_state: dict[str, Any],
        ui_state: dict[str, Any],
        load_config: Callable[[], dict],
        load_chat_state: Callable[[str, str], dict | None],
        load_interview_state: Callable[[str, str], dict | None],
        load_forum_state: Callable[[str, str], list[dict]],
        save_chat_state: Callable[[str, str, list, dict | None], None],
        agent_factory: Any,
        render_chat: Callable[[], None],
    ):
        self.app_state = app_state
        self.ui_state = ui_state
        self.load_config = load_config
        self.load_chat_state = load_chat_state
        self.load_interview_state = load_interview_state
        self.load_forum_state = load_forum_state
        self.save_chat_state = save_chat_state
        self.agent_factory = agent_factory
        self.render_chat = render_chat
        self.agent_run_state = get_agent_run_state(app_state, ui_state)
        self.interview_state = get_interview_state(app_state)

    def restore_main_chat_state(self, fp: str, dt: str):
        chat_data = self.load_chat_state(fp, dt)
        if chat_data and chat_data.get("chat_messages"):
            self.ui_state["chat_messages"] = chat_data["chat_messages"]
            self.render_chat()
            self._restore_main_agent_state(chat_data.get("agent_state"))
        else:
            self.ui_state["chat_messages"] = []
            self.render_chat()

    def _restore_main_agent_state(self, agent_state: dict | None):
        if not agent_state or agent_state.get("turn_count", 0) <= 0:
            return

        agent_inst = self.agent_run_state.get_agent()
        if agent_inst is None:
            cfg = self.load_config()
            agent_inst = self.agent_factory.create_main_agent_if_configured(cfg)
            if agent_inst is not None:
                self.agent_run_state.set_agent(agent_inst)
        if agent_inst is not None:
            agent_inst.restore_state(agent_state)

    def restore_interview_state(self, fp: str, dt: str):
        iv_data = self.load_interview_state(fp, dt)
        if iv_data and iv_data.get("all_conversations"):
            all_convs = iv_data["all_conversations"]
            guidelines_text = iv_data.get("guidelines", "")
            is_active = bool(iv_data.get("active")) and not bool(iv_data.get("finished"))

            self._restore_ic_subagent(iv_data)

            total = iv_data.get("dialogue_round", len(all_convs) // 2)
            load_interview_fn = self.app_state.get("_load_interview_ui")
            if is_active:
                self.interview_state.start(self.interview_state.get_subagent())
                self.agent_run_state.finish_run()
                main_chat_input = self.app_state.get("_main_chat_input")
                if main_chat_input:
                    main_chat_input.unlock("agent_running")
                    main_chat_input.lock("interview_active")
                if load_interview_fn:
                    load_interview_fn(guidelines_text, all_convs, total, active=True)
            else:
                self.interview_state.finish(format_interview_conversations(all_convs))
                if load_interview_fn:
                    load_interview_fn(guidelines_text, all_convs, total)
                self._clear_stale_main_agent_suspension(fp, dt)
        else:
            self.interview_state.reset_for_session_change()
            clear_interview_fn = self.app_state.get("_clear_interview_ui")
            if clear_interview_fn:
                clear_interview_fn()
            self._clear_stale_main_agent_suspension(fp, dt)

    def _restore_ic_subagent(self, iv_data: dict):
        if self.interview_state.get_subagent() is None:
            cfg = self.load_config()
            ic_cfg = cfg.get("ic_subagent", {})
            ic_agent = self.agent_factory.create_information_collection_agent_if_configured(ic_cfg)
            if ic_agent is not None:
                self.interview_state.set_subagent(ic_agent)
        ic_sub = self.interview_state.get_subagent()
        if ic_sub:
            ic_sub.restore_state(iv_data)

    def _clear_stale_main_agent_suspension(self, fp: str, dt: str):
        agent_inst = self.agent_run_state.get_agent()
        if agent_inst and getattr(agent_inst, "_suspended", None):
            agent_inst._suspended = None
            self.save_chat_state(fp, dt, self.ui_state.get("chat_messages", []), agent_inst.export_state())

    def restore_forum_state(self, fp: str, dt: str):
        forum_data = self.load_forum_state(fp, dt)
        agent_inst = self.agent_run_state.get_agent()
        if forum_data:
            if agent_inst:
                agent_inst.forum_history = forum_data
        else:
            if agent_inst:
                agent_inst.forum_history = []
        render_fn = self.app_state.get("render_forum")
        if render_fn:
            render_fn()
