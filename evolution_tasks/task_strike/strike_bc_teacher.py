"""Frozen Strike BC actor used for the grasp phase in both training and evaluation."""
from __future__ import annotations

import json
import os
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn.functional as F


def morphology_sha256(urdf_path: str) -> str:
    """Hash physical geometry and dynamics independently of generated paths."""
    path = Path(urdf_path).resolve()
    root = ET.parse(path).getroot()
    root.attrib.pop("name", None)
    for mesh in root.iter("mesh"):
        mesh_path = Path(mesh.attrib["filename"])
        if not mesh_path.is_absolute():
            mesh_path = path.parent / mesh_path
        mesh.attrib["filename"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()

    def canonical(element):
        return [element.tag, sorted(element.attrib.items()),
                (element.text or "").strip(), [canonical(child) for child in element]]

    return hashlib.sha256(json.dumps(canonical(root), separators=(",", ":")).encode()).hexdigest()


class FrozenStrikeBCActor:
    def __init__(self, checkpoint: str, joint_names: list[str], device: str, urdf_path: str):
        path = Path(checkpoint).resolve()
        contract_path = path.parent.parent / "params" / "policy_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        binding_path = Path(__file__).with_name("verified_bc_teachers.json")
        checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        registry = json.loads(binding_path.read_text(encoding="utf-8"))
        binding = registry.get(checkpoint_sha256, {})
        if binding.get("checkpoint_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Strike BC teacher geometry binding checkpoint hash mismatch")
        self.morphology_sha256 = morphology_sha256(urdf_path)
        if binding.get("morphology_sha256") != self.morphology_sha256:
            raise ValueError("Strike BC teacher physical morphology differs from environment")
        if contract.get("controller") != "strike_joint_target_v1":
            raise ValueError("Strike BC teacher requires direct joint actions")
        if contract.get("actual_joint_names") != list(joint_names):
            raise ValueError("Strike BC teacher joint morphology differs from environment")
        for name, required in contract.get("environment", {}).items():
            if os.environ.get(name) != required:
                raise ValueError(f"Strike BC teacher reset contract mismatch: {name}")
        state = torch.load(path, map_location="cpu", weights_only=False)["model"]
        self.mean = state["running_mean_std.running_mean"].to(device=device, dtype=torch.float32)
        self.var = state["running_mean_std.running_var"].to(device=device, dtype=torch.float32)
        self.layers = [
            (state[f"a2c_network.actor_mlp.{i}.weight"].to(device),
             state[f"a2c_network.actor_mlp.{i}.bias"].to(device))
            for i in (0, 2, 4, 6)
        ]
        self.output = (
            state["a2c_network.mu.weight"].to(device),
            state["a2c_network.mu.bias"].to(device),
        )
        if self.mean.numel() != 162 or self.output[1].numel() != 23:
            raise ValueError("Strike BC teacher observation/action dimensions differ")

    @torch.inference_mode()
    def actions(self, observations: torch.Tensor) -> torch.Tensor:
        x = ((observations.clamp(-5.0, 5.0) - self.mean) / torch.sqrt(self.var + 1e-5)).clamp(-5.0, 5.0)
        for weight, bias in self.layers:
            x = F.elu(F.linear(x, weight, bias))
        return F.linear(x, *self.output).clamp(-1.0, 1.0)
