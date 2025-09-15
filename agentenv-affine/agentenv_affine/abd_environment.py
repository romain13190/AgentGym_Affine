"""
Standalone ABD Env: on fournit un programme Python et une sortie attendue;
l'agent doit retourner l'input exact dans <INPUT>...</INPUT> qui reproduit la sortie.
"""
from __future__ import annotations

import json
import re
import random
import subprocess
from typing import Any, Dict, Optional, Tuple


PROMPT_TEMPLATE = (
    "You are a programming expert. Given a Python program and its expected output, you need to determine the exact input that would produce this output.\n\n"
    "Program:\n```python\n{program}\n```\n\n"
    "Expected Output:\n```\n{output}\n```\n\n"
    "Task: Analyze the program to understand what input format it expects from stdin, then provide the input data that would produce the expected output.\n\n"
    "You can provide any explanations, analysis, or reasoning you want. However, you MUST include the input data within <INPUT> </INPUT> tags.\n\n"
    "Format the input data like this:\n<INPUT>\n[line1]\\n[line2]...\n</INPUT>\n\n"
    "Requirements for the input data within the tags:\n"
    "1. Each line of input should be on a separate line\n"
    "2. Use the exact format the program expects\n"
    "3. Provide only raw input values\n"
    "4. Do not include any prefixes or extra formatting within the INPUT tags\n\n"
    "Please analyze the program and provide the required input:"
)


def _execute_python(program: str, stdin: str, timeout_s: float = 2.0) -> Tuple[str, str]:
    try:
        res = subprocess.run(
            ["python3", "-c", program],
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        return res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT"


def _strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text


def _extract_input_block(response: str) -> str:
    response = _strip_think_blocks(response or "")
    m = re.findall(r"<INPUT>(.*?)</INPUT>", response, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    lines = [ln.rstrip() for ln in m[-1].strip().splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _normalize_output(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").rstrip().splitlines())


class StandaloneABD:
    def __init__(self) -> None:
        self.current: Optional[Dict[str, Any]] = None  # {program, expected_output, prompt}

    async def generate(self) -> Tuple[str, Dict[str, Any]]:
        program, example_in = self._sample_program()
        out, err = _execute_python(program, example_in)
        if err or not out.strip():
            # resample on degenerate
            program, example_in = self._fallback_program()
            out, err = _execute_python(program, example_in)
        prompt = PROMPT_TEMPLATE.format(program=program, output=_normalize_output(out))
        self.current = {"program": program, "expected_output": _normalize_output(out), "prompt": prompt}
        return prompt, self.current

    async def evaluate(self, agent_text: str) -> Tuple[float, Dict[str, Any]]:
        if not self.current:
            raise RuntimeError("No active challenge. Call generate() first.")
        gen_input = _extract_input_block(agent_text)
        if not gen_input:
            return 0.0, {"error": "No <INPUT> block found"}
        prog = self.current["program"]
        expected = self.current["expected_output"]
        out, err = _execute_python(prog, gen_input)
        if err:
            return 0.0, {"error": err, "generated_output": out}
        ok = _normalize_output(out) == _normalize_output(expected)
        return (1.0 if ok else 0.0), {
            "outputs_match": ok,
            "generated_input": gen_input,
            "generated_output": _normalize_output(out),
            "expected_output": expected,
        }

    # ----------------------------- Program pool ---------------------------- #
    def _sample_program(self) -> Tuple[str, str]:
        pool = [
            (
                "n = int(input())\nprint(n*n)",
                "7\n",
            ),
            (
                "a,b = map(int, input().split())\nprint(a+b)\nprint(a*b)",
                "3 4\n",
            ),
            (
                "import sys\nlines = sys.stdin.read().strip().split()\nprint(sum(map(int, lines)))",
                "5 6 7\n",
            ),
            (
                "s = input().strip()\nprint(s[::-1])",
                "abcde\n",
            ),
        ]
        return random.choice(pool)

    def _fallback_program(self) -> Tuple[str, str]:
        return (
            "x = int(input())\nprint(x+1)",
            "9\n",
        )


# --------------------------- Server wrapper ------------------------------- #
class AbdEnvInstance:
    def __init__(self) -> None:
        self.env = StandaloneABD()
        self.prompt: Optional[str] = None

    async def reset(self) -> str:
        prompt, _ = await self.env.generate()
        self.prompt = prompt
        return prompt

    async def observation(self) -> str:
        return self.prompt or (await self.reset())

    async def step(self, action: str) -> Tuple[str, float, bool, dict]:
        score, info = await self.env.evaluate(action)
        obs = f"Score: {score}. Details: {info}"
        return obs, score, True, info


class AbdEnvServer:
    def __init__(self) -> None:
        self._max_id: int = 0
        self.envs: Dict[int, AbdEnvInstance] = {}

    async def create(self) -> int:
        env_idx = self._max_id
        self._max_id += 1
        self.envs[env_idx] = AbdEnvInstance()
        return env_idx

    async def observation(self, env_idx: int) -> str:
        return await self.envs[env_idx].observation()

    async def step(self, env_idx: int, action: str) -> Tuple[str, float, bool, dict]:
        return await self.envs[env_idx].step(action)

    async def reset(self, env_idx: int) -> str:
        return await self.envs[env_idx].reset()


abd_env_server = AbdEnvServer() 