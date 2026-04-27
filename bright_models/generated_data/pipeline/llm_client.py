"""Unified LLM client: local Qwen3 via vLLM (default) + optional API fallbacks.

Usage:
    from config.settings import PipelineConfig
    from pipeline.llm_client import LLMClient

    config = PipelineConfig()
    client = LLMClient(config)
    text = client.generate(system="Tu es un médecin.", user="Génère un CR.")
    doc = client.parse_json_response(text)
"""

import json
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config):
        self.config = config
        self.provider = config.llm_provider
        self._model = None
        self._tokenizer = None
        self._sampling_params = None
        self._api_client = None

        if self.provider == "local":
            self._init_local()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ── Local Qwen3 via vLLM ────────────────────────────────────────────

    def _init_local(self):
        try:
            self._init_vllm()
        except Exception as e:
            logger.warning("vLLM init failed (%s), falling back to transformers", e)
            self._init_transformers()

    def _init_vllm(self):
        # Ensure CUDA device is visible (Colab sometimes doesn't set this)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        # Set HF token in env before importing vLLM (it reads from env)
        if self.config.hf_token:
            os.environ["HF_TOKEN"] = self.config.hf_token

        from vllm import LLM, SamplingParams

        quant = self.config.llm_quantization
        kwargs = dict(
            model=self.config.llm_model,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            dtype="float16",
            max_num_seqs=self.config.batch_size,
        )
        if quant and quant != "none":
            kwargs["quantization"] = quant

        # Try configured max_model_len, fall back to half if OOM
        max_len_candidates = [self.config.max_model_len, self.config.max_model_len // 2]
        for max_len in max_len_candidates:
            try:
                kwargs["max_model_len"] = max_len
                self._model = LLM(**kwargs)
                break
            except Exception as e:
                if max_len == max_len_candidates[-1]:
                    raise
                logger.warning(
                    "vLLM init failed with max_model_len=%d (%s), retrying with %d",
                    max_len, e, max_len // 2,
                )

        self._sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=0.95,
            max_tokens=self.config.max_tokens,
        )
        self._backend = "vllm"
        logger.info("vLLM loaded: %s (max_len=%d, quant=%s)",
                     self.config.llm_model, kwargs["max_model_len"], quant)

    def _init_transformers(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.config.llm_model
        token = self.config.hf_token
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

        load_kwargs = {"device_map": "auto"}
        try:
            from transformers import BitsAndBytesConfig
            import torch
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            logger.warning("bitsandbytes not available, loading in fp16")
            load_kwargs["torch_dtype"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(model_name, token=token, **load_kwargs)
        self._backend = "transformers"
        logger.info("Transformers loaded: %s", model_name)

    # ── API clients ─────────────────────────────────────────────────────

    def _init_anthropic(self):
        from anthropic import Anthropic
        self._api_client = Anthropic()
        self._backend = "anthropic"
        logger.info("Anthropic client initialised: %s", self.config.llm_model)

    def _init_openai(self):
        from openai import OpenAI
        self._api_client = OpenAI()
        self._backend = "openai"
        logger.info("OpenAI client initialised: %s", self.config.llm_model)

    # ── Generation ──────────────────────────────────────────────────────

    def generate(self, system: str, user: str, **kwargs) -> str:
        """Generate a single response. Returns raw text."""
        if self._backend == "vllm":
            try:
                return self._generate_vllm(system, user, **kwargs)
            except Exception as e:
                logger.warning("vLLM engine dead (%s), reinitializing with transformers", e)
                self._model = None
                self._init_transformers()
        if self._backend == "transformers":
            return self._generate_transformers(system, user, **kwargs)
        elif self._backend == "anthropic":
            return self._generate_anthropic(system, user, **kwargs)
        elif self._backend == "openai":
            return self._generate_openai(system, user, **kwargs)
        raise RuntimeError(f"No backend: {self._backend}")

    def generate_batch(self, prompts: list[tuple[str, str]], **kwargs) -> list[str]:
        """Generate multiple responses. Each prompt is (system, user).

        For vLLM, batches all prompts in a single forward pass for throughput.
        Falls back to sequential generation on engine errors.
        For API backends, calls sequentially with rate limiting.
        """
        if self._backend == "vllm":
            try:
                return self._generate_vllm_batch(prompts, **kwargs)
            except Exception as e:
                logger.warning("vLLM engine dead (%s), reinitializing with transformers", e)
                self._model = None
                self._init_transformers()
                # fall through to sequential generation below
        results = []
        for system, user in prompts:
            results.append(self.generate(system, user, **kwargs))
        return results

    # ── vLLM generation ─────────────────────────────────────────────────

    def _build_vllm_prompt(self, system: str, user: str) -> str:
        from transformers import AutoTokenizer
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model, token=self.config.hf_token,
            )
        messages = [
            {"role": "system", "content": system + " /no_think"},
            {"role": "user", "content": user},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _generate_vllm(self, system: str, user: str, **kwargs) -> str:
        prompt = self._build_vllm_prompt(system, user)
        outputs = self._model.generate([prompt], self._sampling_params)
        return outputs[0].outputs[0].text

    def _generate_vllm_batch(self, prompts: list[tuple[str, str]], **kwargs) -> list[str]:
        texts = [self._build_vllm_prompt(s, u) for s, u in prompts]
        outputs = self._model.generate(texts, self._sampling_params)
        return [o.outputs[0].text for o in outputs]

    # ── Transformers generation ─────────────────────────────────────────

    def _generate_transformers(self, system: str, user: str, **kwargs) -> str:
        import torch
        messages = [
            {"role": "system", "content": system + " /no_think"},
            {"role": "user", "content": user},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=0.95,
                do_sample=True,
            )
        generated = out[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    # ── Anthropic generation ────────────────────────────────────────────

    def _generate_anthropic(self, system: str, user: str, **kwargs) -> str:
        self._rate_limit()
        response = self._api_client.messages.create(
            model=self.config.llm_model,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    # ── OpenAI generation ───────────────────────────────────────────────

    def _generate_openai(self, system: str, user: str, **kwargs) -> str:
        self._rate_limit()
        response = self._api_client.chat.completions.create(
            model=self.config.llm_model,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    # ── Rate limiting (API only) ────────────────────────────────────────

    _last_call: float = 0.0

    def _rate_limit(self):
        rpm = self.config.api_rate_limit_rpm
        if rpm <= 0:
            return
        interval = 60.0 / rpm
        elapsed = time.time() - self._last_call
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_call = time.time()

    # ── JSON parsing ────────────────────────────────────────────────────

    @staticmethod
    def parse_json_response(raw: str) -> Optional[dict]:
        """Extract a JSON object from LLM output, handling common issues.

        Strips:
        - <think>...</think> blocks (Qwen3 safety net)
        - Markdown code fences (```json ... ```)
        - Trailing commas before } or ]
        """
        # Strip thinking blocks
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

        # Strip markdown code fences
        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1)

        # Find the outermost { ... }
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        end = start
        in_str = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        candidate = text[start:end]

        # Fix trailing commas
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.warning("JSON parse failed: %s\nCandidate: %.200s...", e, candidate)
            return None
