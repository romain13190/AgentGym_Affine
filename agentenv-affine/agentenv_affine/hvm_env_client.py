from typing import Any, Mapping, Dict

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class AffineHvmEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "You are a reasoning agent. You will receive a VM program with unknown constants (holes) and test cases. Return ONLY the hole mapping in <HOLES>...</HOLES>.",
            }
        ),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 120, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(
            f"{self.env_server_base}/create",
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")
        self.env_id = ok.json()

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int) -> Dict[str, Any]:
        response = self._post("reset", {})
        return response


class AffineHvmTask(BaseTask):
    env_client_cls = AffineHvmEnvClient
    env_name = "AffineHVM"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs) 