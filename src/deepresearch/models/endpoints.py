"""EndpointSet — typed lookup over configured endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from deepresearch.config.schema import EndpointConfig, ModelsSection
from deepresearch.schemas.models import Endpoint


@dataclass
class EndpointSet:
    endpoints: dict[str, Endpoint]
    profiles: dict[str, dict[str, str]]  # profile_name -> role -> endpoint_name

    @classmethod
    def from_config(cls, section: ModelsSection) -> EndpointSet:
        eps: dict[str, Endpoint] = {}
        for name, cfg in section.endpoints.items():
            assert isinstance(cfg, EndpointConfig)
            eps[name] = Endpoint(
                name=name,
                base_url=cfg.base_url,
                api_key=cfg.api_key or "EMPTY",
                model_id=cfg.model_id,
                role_hint=cfg.role_hint,
            )
        profiles = {
            name: prof.model_dump() for name, prof in section.profiles.items()
        }
        return cls(endpoints=eps, profiles=profiles)

    def get(self, name: str) -> Endpoint:
        try:
            return self.endpoints[name]
        except KeyError as e:
            raise KeyError(f"unknown endpoint '{name}' (have: {sorted(self.endpoints)})") from e

    def endpoint_for(self, profile: str, role: str) -> Endpoint:
        try:
            ep_name = self.profiles[profile][role]
        except KeyError as e:
            raise KeyError(f"profile/role not found: {profile}/{role}") from e
        return self.get(ep_name)
