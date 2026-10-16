"""NiceGUI's script mode runs the whole main script again for every browser connection (`ui_run.run_script`): what must exist once
for the process -- the voice manager with its speech-recognition process, the routes -- may not be made by the script itself."""
from types import SimpleNamespace

import voice.service as service
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ui_app.voice_assets import REMOTE_JS_URL, register_voice_web
from voice.remote import WS_PATH, register_routes
from voice.replay import REPLAY_PATH, register_replay_routes


def routes(app, path):
    return [r for r in app.routes if getattr(r, "path", "").startswith(path)]


def test_every_run_of_the_script_gets_the_same_manager():
    service._SHARED = None
    try:
        first = service.shared_manager(load_config=lambda: {})
        again = service.shared_manager(load_config=lambda: {"other": 1})
        assert first is again and isinstance(first, service.VoiceManager)
    finally:
        service._SHARED = None


def test_the_routes_are_mounted_once_however_many_times_the_script_runs():
    app = FastAPI()
    for _ in range(3):
        register_replay_routes(app, SimpleNamespace(session=None))
        register_routes(app)
    assert len(routes(app, REPLAY_PATH)) == 1 and len(routes(app, WS_PATH)) == 1
    # the route that stays answers for the manager it was first given
    assert TestClient(app).get(REPLAY_PATH).status_code == 409


def test_the_scripts_are_served_once_but_every_page_gets_its_head_html():
    served, heads = [], []
    app = SimpleNamespace(add_static_file=lambda **kw: served.append(kw["url_path"]))
    ui = SimpleNamespace(add_head_html=lambda html: heads.append(html))
    for _ in range(2):
        register_voice_web(app, ui)
    assert served.count(REMOTE_JS_URL) == 1 and len(served) == 2
    assert len(heads) == 8 and any("tcmFloat" in h for h in heads)


def test_a_new_app_is_never_mistaken_for_an_earlier_one_whose_memory_address_was_reused():
    # the marker lives on the app itself: a freed app's id() can be handed to the next one (this used to skip the routes at random)
    for _ in range(200):
        app = FastAPI()
        register_replay_routes(app, SimpleNamespace(session=None))
        register_routes(app)
        assert len(routes(app, REPLAY_PATH)) == 1 and len(routes(app, WS_PATH)) == 1
        shell = SimpleNamespace(add_static_file=lambda **kw: None)
        served = []
        shell.add_static_file = lambda **kw: served.append(kw["url_path"])
        register_voice_web(shell, SimpleNamespace(add_head_html=lambda html: None))
        assert len(served) == 2
        del app, shell
