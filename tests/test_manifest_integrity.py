from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_manifest_matches_release_and_provider_capabilities() -> None:
    manifest = load_yaml(ROOT / "manifest.yaml")
    provider = load_yaml(ROOT / "provider" / "sophnet.yaml")

    assert manifest["version"] == "0.0.7"
    assert manifest["meta"]["version"] == "0.0.7"
    assert manifest["meta"]["runner"]["version"] == "3.12"

    permissions = manifest["resource"]["permission"]["model"]
    expected_types = {"llm", "text-embedding", "speech2text", "tts"}
    assert set(provider["supported_model_types"]) == expected_types
    assert permissions["llm"] is True
    assert permissions["text_embedding"] is True
    assert permissions["speech2text"] is True
    assert permissions["tts"] is True
    assert manifest["resource"]["permission"]["tool"]["enabled"] is False


def test_credential_schema_defaults_are_sdk_compatible_strings() -> None:
    provider = load_yaml(ROOT / "provider" / "sophnet.yaml")
    schemas = [
        *provider["provider_credential_schema"]["credential_form_schemas"],
        *provider["model_credential_schema"]["credential_form_schemas"],
    ]

    defaults = {schema["variable"]: schema["default"] for schema in schemas if "default" in schema}
    assert defaults
    assert all(isinstance(default, str) for default in defaults.values())


def test_model_catalog_has_unique_ids_and_complete_position_file() -> None:
    model_dir = ROOT / "models" / "llm"
    model_files = sorted(path for path in model_dir.glob("*.yaml") if path.name != "_position.yaml")
    models = [load_yaml(path)["model"] for path in model_files]
    position = load_yaml(model_dir / "_position.yaml")

    assert len(models) == len(set(models))
    assert set(position) == set(models)


def test_current_public_text_models_are_present() -> None:
    model_dir = ROOT / "models" / "llm"
    models = {
        load_yaml(path)["model"]
        for path in model_dir.glob("*.yaml")
        if path.name != "_position.yaml"
    }
    expected = {
        "DeepSeek-V4-Flash",
        "DeepSeek-V4-Pro",
        "doubao-seed-2-0-mini-260428",
        "GLM-5",
        "GLM-5-Turbo",
        "GLM-5.1",
        "GLM-5.2",
        "Hy3-preview",
        "Kimi-K2.5",
        "Kimi-K2.6",
        "Kimi-K2.7-Code",
        "Kimi-K3",
        "MiniMax-M2.5",
        "MiniMax-M2.7",
        "MiniMax-M3",
        "qwen3-vl-plus",
        "Qwen3.5-397B-A17B",
        "qwen3.6-flash",
        "qwen3.6-max-preview",
        "qwen3.6-plus",
        "qwen3.7-max",
    }
    assert expected <= models


def test_retired_model_ids_remain_available_for_existing_dify_apps() -> None:
    model_dir = ROOT / "models" / "llm"
    models = {
        load_yaml(path)["model"]
        for path in model_dir.glob("*.yaml")
        if path.name != "_position.yaml"
    }

    assert {"DeepSeek-V3.1", "DeepSeek-R1", "qwen-vl-plus-latest"} <= models
