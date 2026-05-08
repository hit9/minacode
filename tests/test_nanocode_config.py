import json
import os
import tempfile

from nanocode import config


def _reset_config():
    """Reset the config singleton cache for isolation between tests."""
    config._cache = None


class TestConfigDefaultValues:
    def test_get_str_returns_default_when_no_env_and_no_config(self):
        _reset_config()
        env_var = "NANOCODE_TEST_DEFAULT_STR_12345"
        os.environ.pop(env_var, None)
        # Ensure no config file interferes
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                val = config.get_str(env_var, "my_default")
                assert val == "my_default", f"Expected 'my_default', got {val!r}"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()

    def test_get_bool_returns_default_false(self):
        _reset_config()
        env_var = "NANOCODE_TEST_DEFAULT_BOOL_12345"
        os.environ.pop(env_var, None)
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                val = config.get_bool(env_var, False)
                assert val is False, f"Expected False, got {val!r}"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()

    def test_get_bool_returns_default_true(self):
        _reset_config()
        env_var = "NANOCODE_TEST_DEFAULT_BOOL_TRUE_12345"
        os.environ.pop(env_var, None)
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                val = config.get_bool(env_var, True)
                assert val is True, f"Expected True, got {val!r}"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()

    def test_get_int_returns_default(self):
        _reset_config()
        env_var = "NANOCODE_TEST_DEFAULT_INT_12345"
        os.environ.pop(env_var, None)
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                val = config.get_int(env_var, 42)
                assert val == 42, f"Expected 42, got {val!r}"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()

    def test_get_float_returns_default(self):
        _reset_config()
        env_var = "NANOCODE_TEST_DEFAULT_FLOAT_12345"
        os.environ.pop(env_var, None)
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                val = config.get_float(env_var, 3.14)
                assert val == 3.14, f"Expected 3.14, got {val!r}"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()


class TestConfigFileFallback:
    def test_get_str_from_config_file(self):
        _reset_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                os.makedirs(os.path.join(tmpdir, "config"), exist_ok=True)
                config_path = os.path.join(tmpdir, "config", "nanocode.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump({"test_key_fallback": "file_value"}, f)

                val = config.get_str("NANOCODE_TEST_KEY_FALLBACK", "default")
                assert val == "file_value", f"Expected 'file_value', got {val!r}"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()

    def test_get_bool_from_config_file(self):
        _reset_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                os.makedirs(os.path.join(tmpdir, "config"), exist_ok=True)
                config_path = os.path.join(tmpdir, "config", "nanocode.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump({"stream": True}, f)

                val = config.get_bool("NANOCODE_STREAM", False)
                assert val is True, f"Expected True, got {val!r}"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()

    def test_get_int_from_config_file(self):
        _reset_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                os.makedirs(os.path.join(tmpdir, "config"), exist_ok=True)
                config_path = os.path.join(tmpdir, "config", "nanocode.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump({"temperature": 0}, f)

                val = config.get_int("NANOCODE_TEMPERATURE", 99)
                assert val == 0, f"Expected 0, got {val!r}"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()

    def test_get_float_from_config_file(self):
        _reset_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                os.makedirs(os.path.join(tmpdir, "config"), exist_ok=True)
                config_path = os.path.join(tmpdir, "config", "nanocode.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump({"model_timeout": "120.5"}, f)

                val = config.get_float("NANOCODE_MODEL_TIMEOUT", 0.0)
                assert val == 120.5, f"Expected 120.5, got {val!r}"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()

    def test_config_file_not_found_returns_default(self):
        _reset_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                # No config directory or file created
                val = config.get_str("NANOCODE_NO_SUCH_KEY_XYZ", "fallback_default")
                assert val == "fallback_default", f"Expected 'fallback_default', got {val!r}"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()


class TestEnvVarPrecedence:
    def test_env_overrides_config_file_str(self):
        _reset_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                os.makedirs(os.path.join(tmpdir, "config"), exist_ok=True)
                config_path = os.path.join(tmpdir, "config", "nanocode.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump({"precedence_test": "from_config"}, f)

                os.environ["NANOCODE_PRECEDENCE_TEST"] = "from_env"
                val = config.get_str("NANOCODE_PRECEDENCE_TEST", "default")
                assert val == "from_env", f"Expected 'from_env', got {val!r}"
            finally:
                os.environ.pop("NANOCODE_PRECEDENCE_TEST", None)
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()

    def test_env_overrides_config_file_bool(self):
        _reset_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                os.makedirs(os.path.join(tmpdir, "config"), exist_ok=True)
                config_path = os.path.join(tmpdir, "config", "nanocode.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump({"env_bool_test": True}, f)

                os.environ["NANOCODE_ENV_BOOL_TEST"] = "false"
                val = config.get_bool("NANOCODE_ENV_BOOL_TEST", False)
                assert val is False, f"Expected False, got {val!r}"
            finally:
                os.environ.pop("NANOCODE_ENV_BOOL_TEST", None)
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()

    def test_env_overrides_config_file_int(self):
        _reset_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                os.makedirs(os.path.join(tmpdir, "config"), exist_ok=True)
                config_path = os.path.join(tmpdir, "config", "nanocode.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump({"env_int_test": 10}, f)

                os.environ["NANOCODE_ENV_INT_TEST"] = "99"
                val = config.get_int("NANOCODE_ENV_INT_TEST", 0)
                assert val == 99, f"Expected 99, got {val!r}"
            finally:
                os.environ.pop("NANOCODE_ENV_INT_TEST", None)
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()


class TestKeyDerivation:
    def test_strips_nanocode_prefix(self):
        _reset_config()
        result = config._config_key("NANOCODE_API_URL")
        assert result == "api_url", f"Expected 'api_url', got {result!r}"

    def test_lowercases_non_prefixed(self):
        _reset_config()
        result = config._config_key("CUSTOM_KEY")
        assert result == "custom_key", f"Expected 'custom_key', got {result!r}"

    def test_preserves_case_after_prefix(self):
        _reset_config()
        # NANOCODE_MyKey -> mykey (lowercased after stripping prefix)
        result = config._config_key("NANOCODE_MyKey")
        assert result == "mykey", f"Expected 'mykey', got {result!r}"

    def test_empty_string(self):
        _reset_config()
        result = config._config_key("")
        assert result == ""


class TestCaching:
    def test_load_caches_result(self):
        _reset_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                os.makedirs(os.path.join(tmpdir, "config"), exist_ok=True)
                config_path = os.path.join(tmpdir, "config", "nanocode.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump({"cached_test": "value"}, f)

                # First call loads and caches
                cfg1 = config._load()
                # Modify the file after cache
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump({"cached_test": "modified"}, f)

                # Second call should return cached value
                cfg2 = config._load()
                assert cfg1 is cfg2, "Cache should return the same dict object"
                assert cfg2["cached_test"] == "value", f"Expected 'value' (cached), got {cfg2['cached_test']!r}"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()


class TestInvalidJSON:
    def test_invalid_json_returns_default(self):
        _reset_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                os.makedirs(os.path.join(tmpdir, "config"), exist_ok=True)
                config_path = os.path.join(tmpdir, "config", "nanocode.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write("{invalid json content!!!")

                val = config.get_str("NANOCODE_INVALID_JSON_KEY", "safe_default")
                assert val == "safe_default", f"Expected 'safe_default', got {val!r}"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)
                _reset_config()


class TestSingleton:
    def test_config_is_instance_with_cache(self):
        assert hasattr(config, "_cache"), "config should have _cache attribute"
        assert hasattr(config, "get_str"), "config should have get_str method"
        assert hasattr(config, "get_bool"), "config should have get_bool method"
        assert hasattr(config, "get_int"), "config should have get_int method"
        assert hasattr(config, "get_float"), "config should have get_float method"
        assert callable(config.get_str), "get_str should be callable"

    def test_singleton_same_object(self):
        import nanocode
        assert nanocode.config is config, "Module-level config should be the same instance"
