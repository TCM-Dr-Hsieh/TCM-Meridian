from __future__ import annotations

from typing import Any

from ui_app.controllers.agent_run_state_controller import get_agent_run_state


def build_forum_tab(ui: Any, app_state: dict[str, Any]):
    """Render the medical Q&A bulletin board."""
    agent_run_state = get_agent_run_state(app_state)
    forum_container = ui.column().classes("w-full gap-2")
    app_state["forum_container"] = forum_container

    def render_forum_posts():
        container = app_state.get("forum_container")
        if container is None:
            return
        container.clear()
        agent = agent_run_state.get_agent()
        posts = agent.forum_history if agent else []
        if not posts:
            with container:
                ui.label("尚無教授諮詢紀錄").style("color: #aaa; font-size: 16px; margin: 40px auto;")
            return
        with container:
            for post in posts:
                pid = post.get("post_id", "?")
                prof_name = post.get("professor_name", post.get("professor_id", "?"))
                prof_id = post.get("professor_id", "?")
                content = post.get("content", "")
                is_question = post.get("role") == "main_agent"
                if is_question:
                    with ui.card().classes("w-full").style(
                        "border-left: 4px solid #2196F3; background: #f0f7ff; padding: 12px;"
                    ):
                        ui.label(f"📝 [{pid}] AI 主治醫師呼叫 {prof_id} ({prof_name})").style(
                            "font-weight: bold; color: #1565C0; font-size: 14px;"
                        )
                        ui.markdown(content).style("margin-top: 8px;")
                else:
                    with ui.card().classes("w-full").style(
                        "border-left: 4px solid #4CAF50; background: #f0fff0; padding: 12px;"
                    ):
                        ui.label(f"🎓 [{pid}] {prof_id} ({prof_name}) 的回答").style(
                            "font-weight: bold; color: #2E7D32; font-size: 14px;"
                        )
                        ui.markdown(content).style("margin-top: 8px;")

    app_state["render_forum"] = render_forum_posts
    render_forum_posts()
