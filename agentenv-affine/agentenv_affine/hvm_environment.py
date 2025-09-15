"""
Standalone HVM Env: génère un programme de VM à trous, rend un prompt, 
parse <HOLES> depuis la réponse de l'agent et évalue la sortie.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple


class StandaloneHVM:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self.current: Optional[Dict[str, Any]] = None  # {program, inputs, expected, prompt}

    async def generate(self) -> Tuple[str, Dict[str, Any]]:
        prog = self._make_program(hard=True)
        inputs, expected = self._forge_io(prog, n_cases=3)
        prompt = self._render_prompt(prog, inputs, expected)
        self.current = {"program": prog, "inputs": inputs, "expected": expected, "prompt": prompt}
        return prompt, self.current

    async def evaluate(self, text: str) -> Tuple[float, Dict[str, Any]]:
        if not self.current:
            raise RuntimeError("No active challenge. Call generate() first.")
        holes = self._parse_holes(text or "")
        if holes is None:
            return 0.0, {"error": "Missing or invalid <HOLES> block"}

        spec = self.current["program"]
        inputs: List[List[int]] = self.current["inputs"]
        expected: List[str] = self.current["expected"]

        # Domain & completeness checks
        hole_domains: Dict[str, List[int]] = spec.get("hole_domains", {})
        hole_names: List[str] = spec.get("holes", [])
        if set(holes.keys()) != set(hole_names):
            return 0.0, {"error": "Not all holes provided"}
        for h, v in holes.items():
            dom = hole_domains.get(h) or []
            if v not in dom:
                return 0.0, {"error": f"value {v} for {h} outside domain {dom}"}

        details: List[Dict[str, Any]] = []
        passed = 0
        for inp, exp in zip(inputs, expected):
            ok, out = self._run_vm_local(spec, holes, inp)
            exp_c = self._canon(exp)
            out_c = self._canon(out) if ok else ""
            correct = ok and (out_c == exp_c)
            details.append({
                "input": inp,
                "expected": exp,
                "got": out,
                "expected_canon": exp_c,
                "got_canon": out_c,
                "passed": bool(correct),
            })
            if correct:
                passed += 1
        score = 1.0 if passed == len(inputs) else 0.0
        return score, {"passed": passed, "total": len(inputs), "details": details}

    # --------------------------- VM construction --------------------------- #
    def _make_program(self, hard: bool) -> Dict[str, Any]:
        rng = self._rng
        holes: List[str] = []
        hole_domains: Dict[str, List[int]] = {}
        code: List[Tuple[str, Optional[str]]] = []

        def new_hole(domain: List[int]) -> str:
            name = f"?{chr(ord('a') + len(holes))}"
            holes.append(name)
            hole_domains[name] = domain[:]
            return name

        dom_small = list(range(-9, 10))
        dom_pos = list(range(0, 13))
        dom_mod = [m for m in range(3, 31)]

        code.append(("LOAD", "0"))
        code.append(("LOAD", "1"))
        code.append(("PUSH", new_hole(dom_small)))  # ?a
        code.append(("MUL", None))
        code.append(("PUSH", new_hole(dom_small)))  # ?b
        code.append(("MUL", None))
        code.append(("ADD", None))
        code.append(("DUP", None))
        code.append(("PUSH", new_hole(dom_mod)))    # ?c
        code.append(("MOD", None))
        code.append(("PRINT", None))

        if hard:
            code.append(("LOAD", "2"))
            d = new_hole(dom_pos)
            e = new_hole(dom_small)
            f = new_hole(dom_mod)

            loop_start = len(code)
            code.append(("DUP", None))
            j_end = len(code); code.append(("JMPZ", None))
            code.append(("SWAP", None))
            code.append(("PUSH", d)); code.append(("MUL", None))
            code.append(("PUSH", e)); code.append(("ADD", None))
            code.append(("PUSH", f)); code.append(("MOD", None))
            code.append(("SWAP", None))
            code.append(("PUSH", "1")); code.append(("SUB", None))
            j_back = len(code); code.append(("JMP", None))
            end_addr = len(code)
            code.append(("POP", None))
            code.append(("PRINT", None))
            code.append(("HALT", None))
            j1 = new_hole([end_addr])
            j2 = new_hole([loop_start])
            code[j_end] = ("JMPZ", j1)
            code[j_back] = ("JMP", j2)
        else:
            code.append(("HALT", None))

        return {
            "code": code,
            "holes": holes,
            "hole_domains": hole_domains,
            "max_steps": 8000 if hard else 4000,
            "stack_cap": 256,
        }

    def _forge_io(self, prog: Dict[str, Any], n_cases: int) -> Tuple[List[List[int]], List[str]]:
        rng = self._rng
        chosen = {h: rng.choice(dom) for h, dom in prog["hole_domains"].items()}
        inputs: List[List[int]] = []
        expected: List[str] = []
        for _ in range(n_cases):
            a = rng.randint(-8, 8)
            b = rng.randint(-8, 8)
            if any(op == "LOAD" and arg == "2" for op, arg in prog["code"]):
                k = rng.randint(1, 8)
                case = [a, b, k]
            else:
                case = [a, b]
            ok, out = self._run_vm_local(prog, chosen, case)
            if not ok or out == "":
                return self._forge_io(prog, n_cases)
            inputs.append(case)
            expected.append(out)
        return inputs, expected

    def _render_prompt(self, prog: Dict[str, Any], inputs: List[List[int]], expected: List[str]) -> str:
        def render_program() -> str:
            lines = []
            for i, (op, arg) in enumerate(prog["code"]):
                lines.append(f"{i:03d}: {op}" + (f" {arg}" if arg is not None else ""))
            return "\n".join(lines)

        hole_lines = []
        for h in prog["holes"]:
            dom = prog["hole_domains"][h]
            if len(dom) > 15:
                hole_lines.append(f"{h} ∈ [{min(dom)}, {max(dom)}] (integers)")
            else:
                hole_lines.append(f"{h} ∈ {{{', '.join(map(str, dom))}}}")

        case_lines = []
        for i, (inp, out) in enumerate(zip(inputs, expected)):
            case_lines.append(f"Case #{i}: input={inp}  expected stdout=\n---\n{out}\n---")

        return (
            "You are given a small stack-based Virtual Machine program with UNKNOWN constants (holes).\n"
            "Instruction set:\n"
            "  PUSH n       ; push integer n (n can also be a hole like ?a)\n"
            "  LOAD i       ; push i-th input integer (0-based)\n"
            "  ADD SUB MUL DIV MOD\n"
            "  DUP SWAP POP\n"
            "  JMP k        ; absolute jump\n"
            "  JMPZ k       ; jump if top==0\n"
            "  JMPNZ k      ; jump if top!=0\n"
            "  PRINT\n"
            "  HALT\n\n"
            f"Program:\n{render_program()}\n\n"
            "Holes and domains:\n- " + "\n- ".join(hole_lines) + "\n\n"
            "Test cases:\n" + "\n".join(case_lines) + "\n\n"
            "Return ONLY the hole mapping in this exact format:\n\n"
            "<HOLES>\n?a=3\n?b=-1\n?c=42\n</HOLES>\n"
        )

    # --------------------------- VM execution ----------------------------- #
    def _run_vm_local(self, prog: Dict[str, Any], holes: Dict[str, int], inputs: List[int]) -> Tuple[bool, str]:
        ip = 0
        steps = 0
        stack: List[int] = []
        out: List[str] = []
        code = prog["code"]
        max_steps = int(prog["max_steps"])
        cap = int(prog["stack_cap"])

        def push(v: int) -> bool:
            if len(stack) >= cap:
                return False
            stack.append(int(v))
            return True

        while True:
            if steps > max_steps or ip < 0 or ip >= len(code):
                return (False, "")
            op, arg = code[ip]
            steps += 1

            if op == "PUSH":
                if arg is None:
                    return (False, "")
                if isinstance(arg, str) and arg.startswith("?"):
                    if arg not in holes or not push(holes[arg]):
                        return (False, "")
                else:
                    if not push(int(arg)):
                        return (False, "")
                ip += 1
            elif op == "LOAD":
                idx = int(arg or -1)
                if idx < 0 or idx >= len(inputs):
                    return (False, "")
                if not push(inputs[idx]):
                    return (False, "")
                ip += 1
            elif op in ("ADD", "SUB", "MUL", "DIV", "MOD"):
                if len(stack) < 2:
                    return (False, "")
                b = stack.pop(); a = stack.pop()
                if op == "ADD":
                    c = a + b
                elif op == "SUB":
                    c = a - b
                elif op == "MUL":
                    c = a * b
                elif op == "DIV":
                    if b == 0:
                        return (False, "")
                    c = int(a / b)
                else:
                    if b == 0:
                        return (False, "")
                    c = a % b
                if not push(c):
                    return (False, "")
                ip += 1
            elif op == "DUP":
                if not stack or not push(stack[-1]):
                    return (False, "")
                ip += 1
            elif op == "SWAP":
                if len(stack) < 2:
                    return (False, "")
                stack[-1], stack[-2] = stack[-2], stack[-1]
                ip += 1
            elif op == "POP":
                if not stack:
                    return (False, "")
                stack.pop(); ip += 1
            elif op in ("JMP", "JMPZ", "JMPNZ"):
                tgt_raw = arg
                tgt = holes[tgt_raw] if (isinstance(tgt_raw, str) and tgt_raw.startswith("?")) else int(tgt_raw)
                if op == "JMP":
                    ip = tgt
                else:
                    if not stack:
                        return (False, "")
                    top = stack.pop()
                    cond = (top == 0)
                    if (op == "JMPZ" and cond) or (op == "JMPNZ" and not cond):
                        ip = tgt
                    else:
                        ip += 1
            elif op == "PRINT":
                if not stack:
                    return (False, "")
                out.append(str(int(stack.pop())))
                ip += 1
            elif op == "HALT":
                break
            else:
                return (False, "")
        return (True, "\n".join(out))

    def _parse_holes(self, text: str) -> Optional[Dict[str, int]]:
        m = re.findall(r"<HOLES>\s*(.*?)\s*</HOLES>", text, flags=re.DOTALL | re.IGNORECASE)
        if not m:
            return None
        block = m[-1]
        out: Dict[str, int] = {}
        for line in block.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mm = re.match(r"(\?[a-zA-Z]\w*)\s*=\s*(-?\d+)$", line)
            if not mm:
                return None
            out[mm.group(1)] = int(mm.group(2))
        return out

    @staticmethod
    def _canon(s: str) -> str:
        if s is None:
            return ""
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        if s.endswith("\n"):
            s = s[:-1]
        return "\n".join(line.rstrip() for line in s.split("\n"))


# --------------------------- Server wrapper ------------------------------- #
class HvmEnvInstance:
    def __init__(self) -> None:
        self.env = StandaloneHVM()
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


class HvmEnvServer:
    def __init__(self) -> None:
        self._max_id: int = 0
        self.envs: Dict[int, HvmEnvInstance] = {}

    async def create(self) -> int:
        env_idx = self._max_id
        self._max_id += 1
        self.envs[env_idx] = HvmEnvInstance()
        return env_idx

    async def observation(self, env_idx: int) -> str:
        return await self.envs[env_idx].observation()

    async def step(self, env_idx: int, action: str) -> Tuple[str, float, bool, dict]:
        return await self.envs[env_idx].step(action)

    async def reset(self, env_idx: int) -> str:
        return await self.envs[env_idx].reset()


hvm_env_server = HvmEnvServer() 