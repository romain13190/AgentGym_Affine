"""
DedEnvServer autonome pour DED (sans dépendance au paquet affine).
- Charge le dataset HF satpalsr/rl-python
- Génère un prompt + consignes
- Évalue un programme Python renvoyé par l'agent sur des cas de test
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import subprocess
import time
from typing import Dict, Optional, Tuple, Any

from datasets import load_dataset

# ------------------------------- Utils -------------------------------- #

def _to_str(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray)):
        return x.decode()
    if isinstance(x, list):
        return "\n".join(_to_str(e) for e in x)
    return json.dumps(x, ensure_ascii=False)


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").rstrip().splitlines())


def _strip_fences(reply: Optional[str]) -> str:
    text = reply or ""
    # supprimer balises <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # extraire dernier bloc ```python ... ``` ou ``` ... ```
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return code_blocks[-1].strip()
    return text.strip()


class SimpleProgramExecutor:
    def __init__(self, timeout_sec: float = 5.0):
        self.timeout_sec = timeout_sec

    def execute(self, code: str, stdin: str = "") -> Tuple[str, str]:
        try:
            p = subprocess.run(
                ["python3", "-c", code],
                input=stdin.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_sec,
            )
            return p.stdout.decode(errors="ignore"), p.stderr.decode(errors="ignore")
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT"


# ------------------------------- Env -------------------------------- #

EXTRA_HINT = (
    "\n\n---\n"
    "⚠️ **Instructions** ⚠️\n"
    "Write a complete **Python 3** program that\n"
    "• reads *all* input from **STDIN** (using `input()` / `sys.stdin`),\n"
    "• writes *only* the required answer(s) to **STDOUT** using `print`,\n"
    "• contains no additional prompts or debug text, and\n"
    "• is returned as a single ```python … ``` fenced block.\n"
)


class DedStandaloneEnv:
    def __init__(self) -> None:
        self._ds = None
        self._executor = SimpleProgramExecutor(timeout_sec=8.0)

    def _ensure_ds(self):
        if self._ds is None:
            self._ds = load_dataset("satpalsr/rl-python", split="train")

    async def generate(self) -> Dict[str, Any]:
        self._ensure_ds()
        sample = self._ds[random.randrange(0, len(self._ds))]
        prompt = (sample.get("prompt", "").rstrip() or "Solve the problem.") + EXTRA_HINT
        sample = dict(sample)
        sample["timestamp"] = time.time()
        return {"prompt": prompt, "extra": sample}

    async def evaluate(self, challenge: Dict[str, Any], raw_reply: str) -> Tuple[float, Dict[str, Any]]:
        program = _strip_fences(raw_reply)
        sample = challenge.get("extra", {})
        ver_raw = sample.get("verification_info") or sample.get("test_cases")
        try:
            if isinstance(ver_raw, str):
                try:
                    ver_json = json.loads(ver_raw)
                except json.JSONDecodeError:
                    ver_json = eval(ver_raw, {"__builtins__": {}}, {})  # dataset legacy
            else:
                ver_json = ver_raw
        except Exception as err:
            return 0.0, {"error": f"Invalid verification_info: {err}"}

        cases = (ver_json or {}).get("test_cases") if isinstance(ver_json, dict) else None
        if not cases:
            return 0.0, {"error": "No public test cases available"}

        passed, total = 0, len(cases)
        details = []
        for i, case in enumerate(cases, start=1):
            ctype = case.get("type")
            raw_inp = case.get("input")
            raw_exp = case.get("output")

            if ctype == "stdin_stdout":
                inp = _to_str(raw_inp)
                if not inp.endswith("\n"):
                    inp += "\n"
                exec_prog = program
                exp = _to_str(raw_exp)
            elif ctype == "function_call":
                fn = case.get("fn_name")
                args = case.get("input", [])
                exec_prog = (
                    program
                    + "\n"
                    + "if __name__ == '__main__':\n"
                    + f"    result = {fn}(*{args!r})\n"
                    + "    print(result)"
                )
                inp = ""
                exp = _to_str(raw_exp[0]) if isinstance(raw_exp, list) and raw_exp else _to_str(raw_exp)
            else:
                total -= 1
                continue

            out, err = self._executor.execute(exec_prog, inp)
            ok_run = not (err or "").strip()
            out_norm = _normalize(out)
            exp_norm = _normalize(exp) if exp is not None else None
            correct = ok_run and (exp_norm is None or out_norm == exp_norm)
            if correct:
                passed += 1
            details.append({
                "input": inp,
                "expected": exp_norm,
                "got": out_norm,
                "stderr": (err or "").strip(),
                "passed": bool(correct),
            })

        score = 1.0 if passed == total else 0.0
        return score, {"passed": passed, "total": total, "tests": details}


# --------------------------- HTTP Server state --------------------------- #

class _DedEnvInstance:
    def __init__(self) -> None:
        self.env = DedStandaloneEnv()
        self.challenge: Optional[Dict[str, Any]] = None

    async def ensure_challenge(self) -> Dict[str, Any]:
        if self.challenge is None:
            self.challenge = await self.env.generate()
        return self.challenge

    async def reset(self, _: Optional[int] = None) -> str:
        self.challenge = await self.env.generate()
        return self.challenge["prompt"]

    async def observation(self) -> str:
        chal = await self.ensure_challenge()
        return chal["prompt"]

    async def step(self, action: str) -> Tuple[str, float, bool, dict]:
        chal = await self.ensure_challenge()
        score, feedback = await self.env.evaluate(chal, action)
        obs = f"Score: {score}. Details: {feedback}"
        return obs, float(score), True, {"extra": feedback}


class DedEnvServer:
    def __init__(self) -> None:
        self._max_id: int = 0
        self.env: Dict[int, _DedEnvInstance] = {}
        self._lock = asyncio.Lock()

    async def create(self, id: int = 0) -> int:
        async with self._lock:
            env_idx = self._max_id
            self._max_id += 1
        self.env[env_idx] = _DedEnvInstance()
        return env_idx

    async def reset(self, env_idx: int, id: Optional[int] = None) -> str:
        return await self.env[env_idx].reset(id)

    async def observation(self, env_idx: int) -> str:
        return await self.env[env_idx].observation()

    async def step(self, env_idx: int, action: str) -> Tuple[str, float, bool, dict]:
        return await self.env[env_idx].step(action)


ded_env_server = DedEnvServer() 