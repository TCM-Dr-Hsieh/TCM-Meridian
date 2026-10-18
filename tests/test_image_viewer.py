"""The picture viewer of the image tab: opened on a ticked picture, stepping through the ticked ones, following the ticks, and the
buttons that tell the browser to zoom, fit, show at 100 % and turn. The browser side (`window.tcmViewer`) is checked in a browser."""
import os
import time

from ui_app.controllers.image_viewer_panel import APP_STATE_KEY, JS, ImageViewerPanel, served_url


class Element:
    def __init__(self, ui, kind, text="", on_click=None, icon=None):
        self.ui, self.kind, self.text, self.icon, self.on_click_ = ui, kind, text, icon, on_click
        self.props_, self.enabled, self.visible = "", True, True
        self.id = len(ui.created)
        ui.created.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def props(self, value):
        self.props_ += " " + value
        return self

    def classes(self, value):
        return self

    def style(self, value):
        return self

    def tooltip(self, text):
        self.tip = text
        return self

    def mark(self, name):
        return self

    def set_visibility(self, visible):
        self.visible = visible

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def set_text(self, text):
        self.text = text

    def click(self):
        self.on_click_()


class FakeUI:
    def __init__(self):
        self.created, self.scripts, self.heads = [], [], []

    def element(self, tag):
        return Element(self, tag)

    def label(self, text=""):
        return Element(self, "label", text)

    def button(self, text="", *, icon=None, on_click=None):
        return Element(self, "button", text, on_click, icon)

    def add_head_html(self, html):
        self.heads.append(html)

    def run_javascript(self, code):
        self.scripts.append(code)


PICTURES = [{"name": f"p{n}.jpg", "path": f"Picture_Row/p{n}.jpg"} for n in (1, 2, 3)]


def make(fail=()):
    ui = FakeUI()
    state = {}

    def url_for(path):
        if path in fail:
            raise FileNotFoundError(path)
        return f"/_nicegui/auto/static/x/{os.path.basename(path)}"

    viewer = ImageViewerPanel(ui=ui, app_state=state, url_for=url_for)
    viewer.build()
    return viewer, ui, state


def button(ui, *, icon=None, text=None):
    return next(e for e in ui.created if e.kind == "button" and (e.icon == icon if icon else e.text == text))


def test_it_is_built_hidden_with_its_browser_script_and_opens_on_the_picture_that_was_clicked():
    viewer, ui, state = make()
    assert state[APP_STATE_KEY] is viewer and viewer.w.panel.visible is False and "data-default-left=320" in viewer.w.panel.props_
    assert any("window.tcmViewer" in head for head in ui.heads) and any(".viewer-box" in head for head in ui.heads)
    viewer.open(PICTURES, 1)
    assert viewer.opened and viewer.w.panel.visible
    assert viewer.w.title.text == "p2.jpg" and viewer.w.position.text == "2 / 3"
    assert ui.scripts[-2] == f"window.tcmFloat.show('c{viewer.w.panel.id}')"
    assert ui.scripts[-1] == f"window.tcmViewer.show('c{viewer.w.box.id}', 'c{viewer.w.pct.id}', '/_nicegui/auto/static/x/p2.jpg')"


def test_the_arrows_step_through_the_ticked_pictures_and_stop_at_the_ends():
    viewer, ui, _ = make()
    viewer.open(PICTURES, 0)
    assert viewer.w.previous.enabled is False and viewer.w.following.enabled is True
    viewer.w.following.click()
    viewer.w.following.click()
    assert viewer.w.title.text == "p3.jpg" and viewer.w.following.enabled is False and viewer.w.previous.enabled is True
    shown = len(ui.scripts)
    viewer.step(1)                                                                     # past the last: nothing happens
    assert viewer.index == 2 and len(ui.scripts) == shown
    viewer.w.previous.click()
    assert viewer.w.title.text == "p2.jpg" and ui.scripts[-1].endswith("'/_nicegui/auto/static/x/p2.jpg')")


def test_the_buttons_tell_the_browser_to_zoom_fit_show_at_its_size_and_turn():
    viewer, ui, _ = make()
    viewer.open(PICTURES, 0)
    box = f"c{viewer.w.box.id}"
    for icon, script in (("zoom_in", f"window.tcmViewer.zoom('{box}', 1.25)"), ("zoom_out", f"window.tcmViewer.zoom('{box}', 0.8)"),
                         ("fit_screen", f"window.tcmViewer.fit('{box}')"), ("rotate_left", f"window.tcmViewer.rotate('{box}', -90)"),
                         ("rotate_right", f"window.tcmViewer.rotate('{box}', 90)")):
        button(ui, icon=icon).click()
        assert ui.scripts[-1] == script, icon
    button(ui, text="100%").click()
    assert ui.scripts[-1] == f"window.tcmViewer.actual('{box}')"
    assert "只影響顯示" in button(ui, icon="rotate_left").tip                       # turning never touches the file


def test_the_viewer_follows_the_ticks_of_the_image_tab():
    viewer, ui, _ = make()
    viewer.sync(PICTURES)                                                              # closed: nothing to follow
    assert viewer.images == [] and not viewer.opened
    viewer.open(PICTURES, 1)                                                           # p2 on show
    shown = len(ui.scripts)
    viewer.sync([PICTURES[1], PICTURES[2]])                                            # p1 unticked: p2 stays, now 1 of 2
    assert viewer.w.title.text == "p2.jpg" and viewer.w.position.text == "1 / 2" and len(ui.scripts) == shown
    viewer.sync([PICTURES[0], PICTURES[2]])                                            # p2 unticked: the one now at its place (first)
    assert viewer.w.title.text == "p1.jpg" and ui.scripts[-1].endswith("'/_nicegui/auto/static/x/p1.jpg')")
    viewer.sync([])                                                                    # nothing ticked (another patient, a new scan)
    assert not viewer.opened and viewer.w.panel.visible is False


def test_a_picture_that_went_away_says_so_and_the_one_shown_before_is_cleared():
    viewer, ui, _ = make(fail={"Picture_Row/p2.jpg"})
    viewer.open(PICTURES, 0)                                                           # p1 on show
    viewer.step(1)                                                                     # p2 is gone
    box, pct = f"c{viewer.w.box.id}", f"c{viewer.w.pct.id}"
    assert viewer.w.title.text == "p2.jpg（讀取不到）"
    assert ui.scripts[-1] == f"window.tcmViewer.blank('{box}', '{pct}', '讀取不到這張圖片')"   # p1 does not stay under p2's name
    viewer.step(1)                                                                     # the next one is there again
    assert viewer.w.title.text == "p3.jpg" and ui.scripts[-1].endswith("'/_nicegui/auto/static/x/p3.jpg')")
    viewer, ui, _ = make(fail={"Picture_Row/p2.jpg"})
    viewer.open(PICTURES, 1)                                                           # opened on it
    assert viewer.w.title.text == "p2.jpg（讀取不到）" and "tcmViewer.blank" in ui.scripts[-1]


def test_a_picture_gets_one_url_per_version_of_its_file(tmp_path):
    picture = tmp_path / "tongue.jpg"
    picture.write_bytes(b"one")
    calls = []

    def add_static_file(*, local_file, strict):
        calls.append((local_file, strict))
        return f"/static/{len(calls)}/tongue.jpg"

    first = served_url(add_static_file, str(picture))
    assert served_url(add_static_file, str(picture)) == first and len(calls) == 1         # looked at again: the same route
    later = time.time() + 5
    os.utime(picture, (later, later))                                                    # the file was replaced
    assert served_url(add_static_file, str(picture)) != first and len(calls) == 2
    assert calls[0] == (os.path.realpath(picture), False)


def test_the_browser_script_keeps_the_zoom_between_ten_percent_and_eight_times():
    assert "MIN = 0.1, MAX = 8" in JS and "'wheel'" in JS and "pointerdown" in JS and "rotate(" in JS


def test_the_browser_script_clears_the_picture_before_the_next_one_comes_and_when_it_cannot_load():
    show = JS[JS.index("show: function"):JS.index("blank: function")]
    assert show.index("blank(id, '載入中…')") < show.index("p.img.src = src")                 # the old picture goes at once
    assert "p.img.onerror = function () { blank(id, '讀取不到這張圖片'); }" in show
    clear = JS[JS.index("function blank("):JS.index("function apply(")]
    assert "removeAttribute('src')" in clear and "onerror = null" in clear and "visibility = 'hidden'" in clear
