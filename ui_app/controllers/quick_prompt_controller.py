from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from ui_app.services.quick_prompt_service import load_quick_prompts, normalize_quick_prompts


class QuickPromptController:
    """Manage global quick prompts and import them into the Main Agent draft."""

    def __init__(
        self,
        *,
        ui: Any,
        load_config: Callable[[], dict],
        save_config: Callable[[dict], None],
        trigger_button: Any,
        agent_input: Any,
    ):
        self.ui = ui
        self.load_config = load_config
        self.save_config = save_config
        self.trigger_button = trigger_button
        self.agent_input = agent_input
        self.prompts: list[dict[str, str]] = []
        self.selected_id: str | None = None
        self._pending_discard_callback: Callable[[], None] | None = None
        self._pending_delete_id: str | None = None
        self._pending_import_content = ""

        self._build_dialog()
        self.trigger_button.on_click(self.open)

    def _build_dialog(self):
        ui = self.ui
        self.dialog = ui.dialog().props("persistent")
        with self.dialog, ui.card().classes("q-pa-none").style(
            "width: 960px; max-width: 95vw; max-height: 90vh; overflow: hidden;"
        ):
            with ui.row().classes("w-full items-center justify-between q-px-lg q-py-md").style(
                "border-bottom: 1px solid #e2e8e2;"
            ):
                ui.label("📌 常用提示詞管理與匯入").style(
                    "font-size: 19px; font-weight: 700; color: var(--primary-dark);"
                )
                ui.button(icon="close", on_click=self._request_close).props("flat round dense").tooltip("關閉")

            with ui.row().classes("w-full items-stretch no-wrap quick-prompt-dialog-body").style(
                "min-height: 520px; max-height: calc(90vh - 72px); overflow: hidden;"
            ):
                with ui.column().classes("q-pa-md gap-2 quick-prompt-list-pane"):
                    ui.button("新增提示詞", icon="add", on_click=lambda: self._request_select(None)).props(
                        "outline no-caps"
                    ).classes("w-full")
                    self.prompt_list = ui.column().classes("w-full gap-2")

                with ui.column().classes("q-pa-lg gap-2").style(
                    "flex: 1 1 auto; min-width: 0; overflow-y: auto;"
                ):
                    self.name_input = ui.input("提示詞名稱").classes("w-full").props("outlined")
                    self.content_input = ui.textarea(
                        "提示詞內容",
                        placeholder="輸入可重複使用的通用指令；請勿包含患者個資。",
                    ).classes("w-full").props(
                        'outlined input-style="min-height: 300px; resize: vertical; line-height: 1.6;"'
                    )
                    self.char_count = ui.label("目前字數：0 字").style("color: #888; font-size: 12px;")
                    self.content_input.on_value_change(
                        lambda e: self.char_count.set_text(f"目前字數：{len(str(e.value or ''))} 字")
                    )
                    ui.label(
                        "提示詞為全域設定。匯入只會填入右欄訊息框，不會自動送出或啟動 Main Agent；請勿儲存患者姓名、病歷或其他個資。"
                    ).style("color: #b26a00; font-size: 12px; line-height: 1.5;")
                    self.status_label = ui.label("").style("font-size: 13px; min-height: 20px;")

                    with ui.row().classes("w-full gap-2 items-center"):
                        ui.button("儲存", icon="save", color="green", on_click=self._save_editor).props(
                            "no-caps"
                        )
                        self.delete_button = ui.button(
                            "刪除", icon="delete", color="red", on_click=self._confirm_delete
                        ).props("outline no-caps")
                        ui.space()
                        ui.button(
                            "匯入至訊息框（不送出）",
                            icon="input",
                            color="primary",
                            on_click=self._import_selected,
                        ).props("no-caps")

        self._build_confirmation_dialogs()

    def _build_confirmation_dialogs(self):
        ui = self.ui

        self.discard_dialog = ui.dialog().props("persistent")
        with self.discard_dialog, ui.card().style("min-width: 360px; max-width: 90vw;"):
            ui.label("尚有未儲存的修改").style("font-size: 17px; font-weight: 700;")
            ui.label("若繼續，這些修改將會遺失。")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("取消", on_click=self._cancel_discard).props("flat")
                ui.button("放棄修改", color="red", on_click=self._discard_confirmed)

        self.delete_dialog = ui.dialog().props("persistent")
        with self.delete_dialog, ui.card().style("min-width: 360px; max-width: 90vw;"):
            ui.label("刪除常用提示詞").style("font-size: 17px; font-weight: 700;")
            self.delete_message = ui.label("")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("取消", on_click=self._cancel_delete).props("flat")
                ui.button("刪除", color="red", on_click=self._delete_confirmed)

        self.import_mode_dialog = ui.dialog().props("persistent")
        with self.import_mode_dialog, ui.card().style("min-width: 400px; max-width: 90vw;"):
            ui.label("訊息框已有草稿").style("font-size: 17px; font-weight: 700;")
            ui.label("請選擇取代現有草稿，或將提示詞附加在草稿後方。")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("取消", on_click=self._cancel_import_mode).props("flat")
                ui.button("取代", color="orange", on_click=self._replace_import).props("outline")
                ui.button("附加", color="primary", on_click=self._append_import)

    def open(self):
        self.prompts = load_quick_prompts(self.load_config())
        self.selected_id = self.prompts[0]["id"] if self.prompts else None
        self._load_editor(self.selected_id)
        self._set_status("")
        self.dialog.open()

    def _render_prompt_list(self):
        self.prompt_list.clear()
        with self.prompt_list:
            if not self.prompts:
                self.ui.label("尚未建立常用提示詞").style(
                    "color: #888; font-size: 13px; padding: 8px 4px;"
                )
                return
            for prompt in self.prompts:
                selected = prompt["id"] == self.selected_id
                button = self.ui.button(
                    prompt["name"],
                    icon="description",
                    on_click=lambda prompt_id=prompt["id"]: self._request_select(prompt_id),
                ).props("no-caps align=left")
                button.classes("w-full")
                if selected:
                    button.props("unelevated color=primary")
                else:
                    button.props("flat color=grey-8")

    def _find_prompt(self, prompt_id: str | None) -> dict[str, str] | None:
        return next((item for item in self.prompts if item["id"] == prompt_id), None)

    def _load_editor(self, prompt_id: str | None):
        self.selected_id = prompt_id
        prompt = self._find_prompt(prompt_id)
        self.name_input.value = prompt["name"] if prompt else ""
        self.content_input.value = prompt["content"] if prompt else ""
        self.char_count.set_text(f"目前字數：{len(self.content_input.value or '')} 字")
        if prompt:
            self.delete_button.enable()
        else:
            self.delete_button.disable()
        self._render_prompt_list()

    def _editor_is_dirty(self) -> bool:
        prompt = self._find_prompt(self.selected_id)
        stored_name = prompt["name"] if prompt else ""
        stored_content = prompt["content"] if prompt else ""
        return (
            str(self.name_input.value or "").strip() != stored_name
            or str(self.content_input.value or "").strip() != stored_content
        )

    def _request_select(self, prompt_id: str | None):
        if prompt_id == self.selected_id:
            return
        if self._editor_is_dirty():
            self._confirm_discard(lambda: self._load_editor(prompt_id))
            return
        self._load_editor(prompt_id)
        self._set_status("")

    def _request_close(self):
        if self._editor_is_dirty():
            self._confirm_discard(self.dialog.close)
            return
        self.dialog.close()

    def _confirm_discard(self, after_discard: Callable[[], None]):
        self._pending_discard_callback = after_discard
        self.discard_dialog.open()

    def _cancel_discard(self):
        self._pending_discard_callback = None
        self.discard_dialog.close()

    def _discard_confirmed(self):
        after_discard = self._pending_discard_callback
        self._pending_discard_callback = None
        self.discard_dialog.close()
        if after_discard:
            after_discard()

    def _save_editor(self):
        name = str(self.name_input.value or "").strip()
        content = str(self.content_input.value or "").strip()
        if not name:
            self._set_status("提示詞名稱不可空白", error=True)
            return
        if not content:
            self._set_status("提示詞內容不可空白", error=True)
            return
        duplicate = next(
            (
                prompt
                for prompt in self.prompts
                if prompt["name"].casefold() == name.casefold() and prompt["id"] != self.selected_id
            ),
            None,
        )
        if duplicate:
            self._set_status("提示詞名稱不可重複", error=True)
            return

        prompt = self._find_prompt(self.selected_id)
        candidate_prompts = [dict(item) for item in self.prompts]
        if prompt:
            for item in candidate_prompts:
                if item["id"] == prompt["id"]:
                    item.update({"name": name, "content": content})
                    break
            saved_id = prompt["id"]
        else:
            prompt = {"id": uuid4().hex, "name": name, "content": content}
            candidate_prompts.append(prompt)
            saved_id = prompt["id"]

        if not self._persist_prompts(candidate_prompts):
            return
        self._load_editor(saved_id)
        self._set_status("✅ 常用提示詞已儲存")

    def _confirm_delete(self):
        prompt = self._find_prompt(self.selected_id)
        if not prompt:
            return
        self._pending_delete_id = prompt["id"]
        self.delete_message.set_text(f"確定要刪除「{prompt['name']}」嗎？此操作無法復原。")
        self.delete_dialog.open()

    def _cancel_delete(self):
        self._pending_delete_id = None
        self.delete_dialog.close()

    def _delete_confirmed(self):
        prompt_id = self._pending_delete_id
        self._pending_delete_id = None
        self.delete_dialog.close()
        if not prompt_id or not self._find_prompt(prompt_id):
            return
        remaining = [item for item in self.prompts if item["id"] != prompt_id]
        if not self._persist_prompts(remaining):
            return
        self._load_editor(self.prompts[0]["id"] if self.prompts else None)
        self._set_status("✅ 常用提示詞已刪除")

    def _import_selected(self):
        prompt = self._find_prompt(self.selected_id)
        if not prompt:
            self._set_status("請先選取或建立提示詞", error=True)
            return
        if self._editor_is_dirty():
            self._set_status("請先儲存目前修改，再匯入提示詞", error=True)
            return

        existing = str(self.agent_input.value or "")
        if not existing.strip():
            self._apply_import(prompt["content"], append=False)
            return
        self._choose_import_mode(prompt["content"])

    def _choose_import_mode(self, content: str):
        self._pending_import_content = content
        self.import_mode_dialog.open()

    def _cancel_import_mode(self):
        self._pending_import_content = ""
        self.import_mode_dialog.close()

    def _replace_import(self):
        content = self._pending_import_content
        self._pending_import_content = ""
        self.import_mode_dialog.close()
        if content:
            self._apply_import(content, append=False)

    def _append_import(self):
        content = self._pending_import_content
        self._pending_import_content = ""
        self.import_mode_dialog.close()
        if content:
            self._apply_import(content, append=True)

    def _apply_import(self, content: str, *, append: bool):
        existing = str(self.agent_input.value or "")
        self.agent_input.value = f"{existing.rstrip()}\n\n{content}" if append and existing.strip() else content
        self.dialog.close()
        self.ui.notify("已匯入訊息框，請修改後手動送出", type="positive")

    def _persist_prompts(self, prompts: list[dict[str, str]]) -> bool:
        try:
            config = self.load_config()
            if not isinstance(config, dict):
                config = {}
            normalized = normalize_quick_prompts(prompts)
            config["quick_prompts"] = normalized
            self.save_config(config)
            self.prompts = normalized
            return True
        except Exception as exc:
            self._set_status(f"儲存失敗：{exc}", error=True)
            return False

    def _set_status(self, text: str, *, error: bool = False):
        self.status_label.set_text(text)
        self.status_label.style(f"color: {'var(--danger)' if error else 'var(--primary-dark)'};")
