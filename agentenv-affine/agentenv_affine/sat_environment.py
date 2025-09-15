"""
Standalone SAT Env: génère une formule k-SAT satisfiable et demande une assignation
sous la forme: x1=True, x2=False, ... (ou UNSAT). Score 1 si l'assignation satisfait toutes
les clauses, sinon 0. Les formules générées sont satisfaisables par construction.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple


class StandaloneSAT:
    def __init__(self, n: int = 15, k: int = 10, m: Optional[int] = None, seed: Optional[int] = None) -> None:
        self.n = int(n)
        self.k = int(k)
        self.m = int(m if m is not None else int(4.26 * self.n))
        self._rng = random.Random(seed)
        self.current: Optional[Dict[str, Any]] = None  # {sol, cls, prompt}

    async def generate(self) -> Tuple[str, Dict[str, Any]]:
        sol = {i: self._rng.choice([True, False]) for i in range(1, self.n + 1)}
        cls: List[List[int]] = []
        for _ in range(self.m):
            vs = self._rng.sample(list(sol), self.k)
            sv = self._rng.choice(vs)
            clause: List[int] = []
            for v in vs:
                if v == sv:
                    lit = v if sol[v] else -v
                else:
                    lit = v if self._rng.choice([True, False]) else -v
                clause.append(lit)
            cls.append(clause)
        formula = " ∧ ".join("(" + " ∨ ".join(("" if l > 0 else "¬") + f"x{abs(l)}" for l in c) + ")" for c in cls)
        prompt = (
            f"Find a satisfying assignment for the following {self.k}-SAT formula over variables x1..x{self.n}:\n"
            f"{formula}\n"
            "Provide your answer as comma-separated assignments like `x1=True, x2=False, ...`, "
            "or respond `UNSAT` if it has no solution."
        )
        self.current = {"sol": sol, "cls": cls, "prompt": prompt}
        return prompt, self.current

    async def evaluate(self, agent_text: str) -> Tuple[float, Dict[str, Any]]:
        if not self.current:
            raise RuntimeError("No active challenge. Call generate() first.")
        if (agent_text or "").strip().upper() == "UNSAT":
            # Les formules sont satisfaisables par construction
            return 0.0, {"error": "Formula is satisfiable; UNSAT is incorrect"}
        got = {
            int(v): val.lower() in ("true", "1")
            for v, val in re.findall(r"x(\d+)=(True|False|1|0)", agent_text or "")
        }
        if not got:
            return 0.0, {"error": "No valid assignment parsed"}
        sol = self.current["sol"]
        cls = self.current["cls"]
        ok = all(any(((lit > 0) == got.get(abs(lit), None)) for lit in c) for c in cls)
        return (1.0 if ok else 0.0), {"expected": sol, "got": got}


class SatEnvInstance:
    def __init__(self) -> None:
        self.env = StandaloneSAT()
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


class SatEnvServer:
    def __init__(self) -> None:
        self._max_id: int = 0
        self.envs: Dict[int, SatEnvInstance] = {}

    async def create(self) -> int:
        env_idx = self._max_id
        self._max_id += 1
        self.envs[env_idx] = SatEnvInstance()
        return env_idx

    async def observation(self, env_idx: int) -> str:
        return await self.envs[env_idx].observation()

    async def step(self, env_idx: int, action: str) -> Tuple[str, float, bool, dict]:
        return await self.envs[env_idx].step(action)

    async def reset(self, env_idx: int) -> str:
        return await self.envs[env_idx].reset()


sat_env_server = SatEnvServer() 