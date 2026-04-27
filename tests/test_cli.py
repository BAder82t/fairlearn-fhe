"""Command-line verifier behavior."""

import json

from fairlearn_fhe import audit_metric
from fairlearn_fhe.cli import main


def test_verify_cli_accepts_valid_envelope(tmp_path, capsys, small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true,
        y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    path = tmp_path / "envelope.json"
    path.write_text(env.to_json(), encoding="utf-8")

    assert main([str(path)]) == 0
    assert capsys.readouterr().out.strip() == "OK"


def test_verify_cli_rejects_tampered_envelope(tmp_path, capsys, small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true,
        y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    payload = env.to_dict()
    payload["parameter_set"]["scaling_factor_bits"] = 30
    path = tmp_path / "envelope.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert main([str(path), "--json"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is False
    assert "parameter_set_hash does not match parameter_set" in output["errors"]
