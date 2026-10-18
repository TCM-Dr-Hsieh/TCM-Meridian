"""The previews of the image tab belong to its list of pictures: when the list is made again (another patient, a save, a delete, a
new scan) nothing is ticked, so the previews of the list that was are cleared. A picture of the patient before can neither stay on
show nor open the viewer again."""
from types import SimpleNamespace

from ui_app.controllers.image_controller import build_image_tab
from ui_app.controllers.image_viewer_panel import APP_STATE_KEY


class Element:
    def __init__(self, ui, kind, text="", *, value=None, on_click=None, source=None):
        self.ui, self.kind, self.text, self.value, self.on_click_, self.source = ui, kind, text, value, on_click, source
        self.children, self.handlers = [], {}
        (ui.stack[-1].children if ui.stack else ui.roots).append(self)

    def __enter__(self):
        self.ui.stack.append(self)
        return self

    def __exit__(self, *exc):
        self.ui.stack.pop()

    def classes(self, *args, **kwargs):
        return self

    def style(self, *args, **kwargs):
        return self

    def props(self, *args, **kwargs):
        return self

    def tooltip(self, *args):
        return self

    def on(self, event, handler):
        self.handlers[event] = handler
        return self

    def on_upload(self, handler):
        pass

    def on_value_change(self, handler):
        self.handlers["change"] = handler

    def reset(self):
        pass

    def clear(self):
        self.children = []

    def walk(self):
        for child in self.children:
            yield child
            yield from child.walk()


class FakeUI:
    def __init__(self):
        self.roots, self.stack = [], []

    def label(self, text=""):
        return Element(self, "label", text)

    def card(self):
        return Element(self, "card")

    def row(self):
        return Element(self, "row")

    def column(self):
        return Element(self, "column")

    def input(self, label="", *, value=None, placeholder=None):
        return Element(self, "input", label, value=value)

    def upload(self, **kwargs):
        return Element(self, "upload")

    def button(self, text="", *, icon=None, on_click=None):
        return Element(self, "button", text or icon, on_click=on_click)

    def checkbox(self, text="", *, value=False):
        return Element(self, "checkbox", text, value=value)

    def image(self, source):
        return Element(self, "image", source=source)

    def everything(self):
        for root in self.roots:
            yield root
            yield from root.walk()


class Viewer:
    def __init__(self):
        self.calls = []

    def open(self, images, index):
        self.calls.append(("open", [item["path"] for item in images], index))

    def sync(self, images):
        self.calls.append(("sync", [item["path"] for item in images]))


def patient(tmp_path, name, pictures):
    folder = tmp_path / name
    (folder / "Picture_Row").mkdir(parents=True)
    for picture in pictures:
        (folder / "Picture_Row" / picture).write_bytes(b"picture")
    return str(folder)


def build(tmp_path):
    first = patient(tmp_path, "A", ["a1.jpg", "a2.jpg"])
    second = patient(tmp_path, "B", ["b1.jpg"])
    ui, viewer = FakeUI(), Viewer()
    state = {"selected_patient_folder": first, APP_STATE_KEY: viewer}
    build_image_tab(ui, state, "2026-10-18")
    return ui, state, viewer, first, second


def tick(ui, name):
    box = next(e for e in ui.everything() if e.kind == "checkbox" and e.text == name)
    box.handlers["change"](SimpleNamespace(value=True))


def previews(ui):
    return [e for e in ui.everything() if e.kind == "image"]


def test_another_patient_clears_the_previews_of_the_patient_before(tmp_path):
    ui, state, viewer, first, second = build(tmp_path)
    tick(ui, "a1.jpg")
    [shown] = previews(ui)
    assert shown.source.startswith(first)
    shown.handlers["click"]()                                                      # the preview opens the viewer on it
    assert viewer.calls[-1][0] == "open" and viewer.calls[-1][1][0].startswith(first)

    state["selected_patient_folder"] = second                                      # B has pictures too: its list is shown
    state["_refresh_image_list"](clear_staged_files=True)
    assert previews(ui) == []                                                      # A's preview is gone, and with it its click
    assert any(e.kind == "checkbox" and e.text == "b1.jpg" for e in ui.everything())
    assert any(e.kind == "label" and e.text == "請在左側勾選圖片" for e in ui.everything())
    assert viewer.calls[-1] == ("sync", [])                                        # the viewer closes


def test_a_new_scan_clears_the_previews_whose_ticks_it_dropped(tmp_path):
    ui, _, viewer, first, _ = build(tmp_path)
    tick(ui, "a1.jpg")
    tick(ui, "a2.jpg")
    assert len(previews(ui)) == 2
    next(e for e in ui.everything() if e.kind == "button" and e.text == "🔄 重新掃描").on_click_()
    assert previews(ui) == [] and viewer.calls[-1] == ("sync", [])
    assert all(e.value is False for e in ui.everything() if e.kind == "checkbox")
