import unittest

from openclaw_bench.providers.inherit import clone_profile


def _vllm_only_source() -> dict:
    """Minimal but realistic source profile with a single vLLM provider.

    Shape mirrors a real benchclaw-m2-style profile (verified against
    /root/.openclaw-benchclaw-m2/openclaw.json on oc-stack 2026-05-02).
    """
    return {
        "agents": {
            "defaults": {
                "model": "vllm/qwen3.5-4b",
                "models": {
                    "vllm/qwen3.5-4b": {
                        "params": {
                            "chatTemplateKwargs": {"enable_thinking": False},
                            "maxTokens": 128,
                        }
                    }
                },
                "params": {"maxTokens": 128},
                "skipBootstrap": True,
            },
            "list": [
                {
                    "id": "main",
                    "model": "vllm/qwen3.5-4b",
                    "tools": {"profile": "coding"},
                }
            ],
        },
        "commands": {
            "native": "auto",
            "nativeSkills": "auto",
            "ownerDisplay": "raw",
            "restart": True,
        },
        "env": {"vars": {"VLLM_API_KEY": "vllm-local"}},
        "gateway": {
            "auth": {"mode": "token", "token": "source-token-XYZ"},
            "bind": "loopback",
            "mode": "local",
            "port": 19298,
            "tailscale": {"mode": "off"},
        },
        "meta": {
            "lastTouchedAt": "2026-05-02T01:24:06.575561Z",
            "lastTouchedVersion": "2026.4.27",
        },
        "models": {
            "providers": {
                "vllm": {
                    "api": "openai-completions",
                    "baseUrl": "http://10.68.198.1:8003/v1",
                    "models": [
                        {
                            "contextTokens": 16000,
                            "contextWindow": 16000,
                            "id": "qwen3.5-4b",
                            "maxTokens": 128,
                            "name": "qwen3.5-4b",
                            "reasoning": False,
                        }
                    ],
                    "request": {
                        "allowPrivateNetwork": True,
                        "auth": {
                            "mode": "authorization-bearer",
                            "token": {"id": "VLLM_API_KEY", "provider": "bench", "source": "env"},
                        },
                    },
                }
            }
        },
        "plugins": {"entries": {"vllm": {"enabled": True}}},
        "secrets": {
            "providers": {
                "bench": {
                    "allowlist": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VLLM_API_KEY"],
                    "source": "env",
                }
            }
        },
    }


def _multi_provider_source() -> dict:
    """Source with both vLLM and Ollama, plus model-class-specific param shaping
    (the GPT-OSS reasoning_effort tweak that issues #1 and #7 silently dropped).
    """
    src = _vllm_only_source()
    src["models"]["providers"]["vllm"]["models"].append(
        {
            "contextTokens": 32768,
            "contextWindow": 32768,
            "id": "gpt-oss-20b",
            "maxTokens": 512,
            "name": "gpt-oss-20b",
            "reasoning": False,
        }
    )
    src["models"]["providers"]["ollama"] = {
        "api": "ollama",
        "baseUrl": "http://127.0.0.1:11434",
        "models": [{"id": "qwen3:8b", "name": "qwen3:8b", "contextWindow": 32768}],
    }
    src["agents"]["defaults"]["models"]["vllm/gpt-oss-20b"] = {
        "params": {
            "chatTemplateKwargs": {"enable_thinking": False},
            "extra_body": {"reasoning_effort": "low"},
            "maxTokens": 512,
        }
    }
    src["agents"]["defaults"]["models"]["ollama/qwen3:8b"] = {
        "params": {"maxTokens": 512}
    }
    src["plugins"]["entries"]["ollama"] = {"enabled": True}
    return src


class CloneProfilePreservesProviderWiringTests(unittest.TestCase):
    def test_vllm_provider_block_copied_verbatim(self):
        src = _vllm_only_source()
        clone = clone_profile(
            src,
            bench_profile="bench-pmg",
            gateway_port=19350,
            gateway_token="bench-token-T",
        )
        self.assertEqual(
            clone["models"]["providers"]["vllm"],
            src["models"]["providers"]["vllm"],
            msg="provider block must be byte-equal to source — auth, baseUrl, models all preserved",
        )

    def test_secrets_and_env_copied_verbatim(self):
        src = _vllm_only_source()
        clone = clone_profile(
            src,
            bench_profile="bench-pmg",
            gateway_port=19350,
            gateway_token="t",
        )
        self.assertEqual(clone["secrets"], src["secrets"])
        self.assertEqual(clone["env"], src["env"])
        self.assertEqual(clone["commands"], src["commands"])
        self.assertEqual(clone["plugins"], src["plugins"])

    def test_per_model_param_shaping_preserved(self):
        # The exact failure mode of issues #1 / #7: parameter shaping for
        # gpt-oss (reasoning_effort=low) and qwen (enable_thinking=false)
        # disappears on the generation path. Inheritance must NOT lose this.
        src = _multi_provider_source()
        clone = clone_profile(
            src,
            bench_profile="bench-pmg",
            gateway_port=19350,
            gateway_token="t",
            bench_route_model="vllm/gpt-oss-20b",
        )
        gpt_oss_params = clone["agents"]["defaults"]["models"]["vllm/gpt-oss-20b"]["params"]
        self.assertEqual(gpt_oss_params["extra_body"]["reasoning_effort"], "low")
        self.assertFalse(gpt_oss_params["chatTemplateKwargs"]["enable_thinking"])
        # Other models' shaping also preserved
        qwen_params = clone["agents"]["defaults"]["models"]["vllm/qwen3.5-4b"]["params"]
        self.assertFalse(qwen_params["chatTemplateKwargs"]["enable_thinking"])

    def test_multi_provider_source_keeps_all_providers(self):
        src = _multi_provider_source()
        clone = clone_profile(
            src,
            bench_profile="bench-pmg",
            gateway_port=19350,
            gateway_token="t",
            bench_route_model="ollama/qwen3:8b",
        )
        self.assertIn("vllm", clone["models"]["providers"])
        self.assertIn("ollama", clone["models"]["providers"])
        self.assertEqual(
            clone["models"]["providers"]["ollama"]["baseUrl"],
            "http://127.0.0.1:11434",
        )


class CloneProfileAppliesOverlayTests(unittest.TestCase):
    def test_gateway_port_overridden_token_replaced(self):
        src = _vllm_only_source()
        clone = clone_profile(
            src,
            bench_profile="bench-pmg",
            gateway_port=19350,
            gateway_token="bench-fresh-token",
        )
        self.assertEqual(clone["gateway"]["port"], 19350)
        self.assertEqual(clone["gateway"]["auth"]["token"], "bench-fresh-token")
        # bind/mode/tailscale preserved from source
        self.assertEqual(clone["gateway"]["bind"], "loopback")
        self.assertEqual(clone["gateway"]["mode"], "local")
        self.assertEqual(clone["gateway"]["tailscale"]["mode"], "off")

    def test_gateway_token_auto_generated_when_omitted(self):
        src = _vllm_only_source()
        clone = clone_profile(
            src,
            bench_profile="bench-pmg",
            gateway_port=19350,
        )
        token = clone["gateway"]["auth"]["token"]
        self.assertIsInstance(token, str)
        self.assertGreaterEqual(len(token), 32)
        self.assertNotEqual(token, "source-token-XYZ", msg="must be a fresh token, not the source's")

    def test_agents_list_replaced_with_single_bench_agent(self):
        src = _vllm_only_source()
        clone = clone_profile(
            src,
            bench_profile="bench-pmg",
            gateway_port=19350,
            gateway_token="t",
            bench_agent_id="bench",
        )
        agents = clone["agents"]["list"]
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["id"], "bench")
        self.assertEqual(agents[0]["model"], "vllm/qwen3.5-4b")
        self.assertEqual(agents[0]["tools"]["profile"], "coding")

    def test_agents_defaults_skipBootstrap_forced_true(self):
        src = _vllm_only_source()
        # Even if the source has skipBootstrap=False, clone forces it True.
        src["agents"]["defaults"]["skipBootstrap"] = False
        clone = clone_profile(
            src,
            bench_profile="bench-pmg",
            gateway_port=19350,
            gateway_token="t",
        )
        self.assertTrue(clone["agents"]["defaults"]["skipBootstrap"])

    def test_meta_version_set_and_stale_timestamp_dropped(self):
        src = _vllm_only_source()
        # Source has lastTouchedAt; clone should drop it so OC repopulates on first edit.
        self.assertIn("lastTouchedAt", src["meta"])
        clone = clone_profile(
            src,
            bench_profile="bench-pmg",
            gateway_port=19350,
            gateway_token="t",
            openclaw_version="2026.4.27",
        )
        self.assertEqual(clone["meta"]["lastTouchedVersion"], "2026.4.27")
        self.assertNotIn("lastTouchedAt", clone["meta"])

    def test_explicit_timestamp_overrides_dropping(self):
        src = _vllm_only_source()
        clone = clone_profile(
            src,
            bench_profile="bench-pmg",
            gateway_port=19350,
            gateway_token="t",
            last_touched_at="2026-05-02T18:00:00Z",
        )
        self.assertEqual(clone["meta"]["lastTouchedAt"], "2026-05-02T18:00:00Z")

    def test_explicit_route_model_overrides_source_default(self):
        src = _multi_provider_source()
        clone = clone_profile(
            src,
            bench_profile="bench-pmg",
            gateway_port=19350,
            gateway_token="t",
            bench_route_model="vllm/gpt-oss-20b",
        )
        self.assertEqual(clone["agents"]["defaults"]["model"], "vllm/gpt-oss-20b")
        self.assertEqual(clone["agents"]["list"][0]["model"], "vllm/gpt-oss-20b")


class CloneProfileSafetyTests(unittest.TestCase):
    def test_source_dict_unchanged_after_clone(self):
        # Pure function: must deep-copy, not mutate the caller's dict.
        src = _vllm_only_source()
        original_token = src["gateway"]["auth"]["token"]
        original_port = src["gateway"]["port"]
        original_skipboot = src["agents"]["defaults"]["skipBootstrap"]
        clone_profile(
            src,
            bench_profile="bench-pmg",
            gateway_port=19350,
            gateway_token="t",
        )
        self.assertEqual(src["gateway"]["auth"]["token"], original_token)
        self.assertEqual(src["gateway"]["port"], original_port)
        self.assertEqual(src["agents"]["defaults"]["skipBootstrap"], original_skipboot)

    def test_non_dict_source_raises(self):
        with self.assertRaises(ValueError):
            clone_profile("not a dict", bench_profile="x", gateway_port=19350)

    def test_source_without_providers_raises(self):
        with self.assertRaises(ValueError) as cm:
            clone_profile(
                {"models": {"providers": {}}, "agents": {"defaults": {"model": "x/y"}}},
                bench_profile="bench-empty",
                gateway_port=19350,
            )
        self.assertIn("no models.providers", str(cm.exception))

    def test_source_without_resolvable_route_raises(self):
        # Source has providers but no agents.defaults.model and caller didn't pass override.
        src = _vllm_only_source()
        del src["agents"]["defaults"]["model"]
        with self.assertRaises(ValueError) as cm:
            clone_profile(src, bench_profile="bench-pmg", gateway_port=19350)
        self.assertIn("route_model", str(cm.exception))

    def test_route_referencing_unknown_provider_raises(self):
        src = _vllm_only_source()
        with self.assertRaises(ValueError) as cm:
            clone_profile(
                src,
                bench_profile="bench-pmg",
                gateway_port=19350,
                bench_route_model="nonexistent-provider/some-model",
            )
        self.assertIn("nonexistent-provider", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
