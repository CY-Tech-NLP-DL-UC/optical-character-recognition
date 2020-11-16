from ocrapi import create_app


def test_config():
    assert not create_app().testing
    assert create_app({"TESTING": True}).testing


def test_home(client) -> None:
    response = client.get("/welcome")
    assert response.data == b"Welcome dear developers!"
