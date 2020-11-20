from pathlib import Path

from ocrapi import create_app

data_folder = Path(__file__).resolve().parent / "data"


def test_config():
    assert not create_app().testing
    assert create_app({"TESTING": True}).testing


def test_home(client) -> None:
    response = client.get("/welcome")
    assert response.data == b"Welcome dear developers!"


def test_predict(client) -> None:
    response = client.post(
        "/predict", data={"file": open(str(data_folder / "A007HA50.png"), "rb")}
    )
    assert response.json["prediction"] == "A007HA50"
