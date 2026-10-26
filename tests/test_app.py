from streamlit.testing.v1 import AppTest


def test_app_renders_without_errors():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6


def test_editing_budget_and_rate_reruns_cleanly():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    at.number_input[0].set_value(300.0).run()
    at.slider[0].set_value(0.12).run()
    assert not at.exception, at.exception
