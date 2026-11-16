import importlib.util
import itertools
import pickle

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from mllab.deploy import cloudrun
from mllab.registry import Registry

FEATS = ["a", "b", "c"]
_n = itertools.count()


def data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=FEATS)
    return X, (X.a - 0.5 * X.b > 0).astype(int)


def serve(tmp_path, payload, *, features=None, full_lock=False):
    """Register `payload` like the lab does, render the Cloud Run bundle, import the generated app.py, return (client, module)."""
    art = tmp_path / f"m{next(_n)}.pkl"; art.write_bytes(pickle.dumps(payload))
    lock = tmp_path / "req.lock"; lock.write_text("pandas==2.3.3\nnumpy==2.4.0\nscikit-learn==1.8.0\nxgboost==3.1.3\n")
    reg = Registry(tmp_path / f"reg{next(_n)}")
    reg.register("p", "m", "v1", art, lockfile=lock, features=features)
    out = cloudrun.render_bundle(reg, "p", "m", "v1", tmp_path / f"b{next(_n)}", full_lock=full_lock)
    spec = importlib.util.spec_from_file_location(f"served_{next(_n)}", out / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return TestClient(mod.app), mod


def xgb_payload(**extra):
    X, y = data()
    m = XGBClassifier(n_estimators=15, max_depth=3, eval_metric="logloss").fit(X, y)
    return m, X, {"model": m, "features": FEATS, "fill_values": {"a": 0.0, "b": 0.0, "c": 0.0}, **extra}


def test_dict_payload_serves_and_matches_the_model_exactly(tmp_path):
    """Regression: the template used to call .predict_proba on the dict itself and crash (V10 and lab-trained models are dicts)."""
    m, X, payload = xgb_payload()
    client, _ = serve(tmp_path, payload)
    rows = X.head(20).to_dict("records")
    r = client.post("/predict", json={"rows": rows})
    assert r.status_code == 200
    assert np.allclose(r.json()["proba"], m.predict_proba(X.head(20))[:, 1], atol=1e-6)
    h = client.get("/healthz").json()
    assert h["ok"] and h["features"] == 3 and h["model"] == "XGBClassifier"


def test_missing_values_are_imputed_like_training(tmp_path):
    m, X, payload = xgb_payload()
    payload["fill_values"] = {"a": 0.0, "b": 0.0, "c": 0.7}
    client, _ = serve(tmp_path, payload)
    got = client.post("/predict", json={"rows": [{"a": 1.0, "b": -1.0, "c": None}]}).json()["proba"][0]
    want = m.predict_proba(pd.DataFrame([{"a": 1.0, "b": -1.0, "c": 0.7}]))[0, 1]  # None became the stored fill value 0.7
    assert got == pytest.approx(want, abs=1e-6)


def test_extra_columns_are_ignored_and_order_does_not_matter(tmp_path):
    m, X, payload = xgb_payload()
    client, _ = serve(tmp_path, payload)
    r = client.post("/predict", json={"rows": [{"zzz": 9, "c": 0.1, "b": 0.2, "a": 0.3}]})
    assert r.json()["proba"][0] == pytest.approx(m.predict_proba(pd.DataFrame([{"a": 0.3, "b": 0.2, "c": 0.1}]))[0, 1], abs=1e-6)


@pytest.mark.parametrize("body, code", [
    ({}, 400), ({"rows": []}, 400), ({"rows": "x"}, 400), ({"rows": [1, 2]}, 400),
    ({"rows": [{"a": 1, "b": 2}]}, 422),              # missing feature c
    ({"rows": [{"a": "hello", "b": 1, "c": 2}]}, 422),  # not numeric
    ({"rows": [{"a": 1, "b": 1, "c": 1}] * 1001}, 413),  # too many rows
])
def test_bad_requests_get_clear_errors_not_500s(tmp_path, body, code):
    client, _ = serve(tmp_path, xgb_payload()[2])
    assert client.post("/predict", json=body).status_code == code


def test_missing_feature_message_names_the_columns(tmp_path):
    client, _ = serve(tmp_path, xgb_payload()[2])
    assert "missing features: ['c']" in client.post("/predict", json={"rows": [{"a": 1, "b": 2}]}).text


def test_scaler_payloads_are_applied(tmp_path):
    X, y = data()
    sc = StandardScaler().fit(X)
    lr = LogisticRegression().fit(sc.transform(X), y)
    client, _ = serve(tmp_path, {"model": lr, "features": FEATS, "scaler": sc, "fill_values": {}})
    got = client.post("/predict", json={"rows": X.head(5).to_dict("records")}).json()["proba"]
    assert np.allclose(got, lr.predict_proba(sc.transform(X.head(5)))[:, 1], atol=1e-9)


def test_bare_estimator_uses_its_own_feature_names(tmp_path):
    m, X, _ = xgb_payload()
    client, _ = serve(tmp_path, m)
    assert client.post("/predict", json={"rows": X.head(3).to_dict("records")}).status_code == 200


def test_unsupported_payloads_fail_at_startup_not_on_the_first_request(tmp_path):
    with pytest.raises(RuntimeError, match="unsupported model payload"):
        serve(tmp_path, {"models": {}, "weights": {}, "feature_sets": {}}, full_lock=True)  # a V8-style ensemble, past the lean gate
    with pytest.raises(RuntimeError, match="unsupported model object"):
        serve(tmp_path, ["not", "a", "model"], full_lock=True)
    m, X, _ = xgb_payload()
    m2 = LogisticRegression().fit(X.to_numpy(), (X.a > 0).astype(int))  # no feature names, none registered
    with pytest.raises(RuntimeError, match="no feature list"):
        serve(tmp_path, m2, full_lock=True)


def test_lean_bundling_refuses_a_v8_style_ensemble_before_any_image_is_built(tmp_path):
    from mllab.registry import RegistryError
    with pytest.raises(RegistryError, match="could not tell which model library"):
        serve(tmp_path, {"models": {}, "weights": {}, "feature_sets": {}})
