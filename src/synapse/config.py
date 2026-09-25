"""Synapse configuration — all optimization parameters in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SynapseConfig:
    """Configuration for Synapse memory provider.

    All optimization parameters have sensible defaults validated during spikes.
    Override via environment variables (SYNAPSE_*) or directly.
    """

    # FalkorDB connection
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: str | None = None
    falkordb_database: str = "synapse"

    # LLM config (OpenAI-compatible endpoint)
    llm_api_key: str = ""
    llm_base_url: str = ""
    embedding_base_url: str = ""           # empty = use llm_base_url
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Optimization parameters (validated in spikes)
    batch_size: int = 5                    # turns per episode (−86% LLM calls)
    half_life_days: float = 7.0            # forgetting curve base
    salience_boost: float = 3.0            # high-salience decay multiplier
    recall_boost: float = 1.5              # recall strengthening factor
    prune_threshold: float = 0.05          # below = prune
    trivial_turn_threshold: int = 10       # chars; below = skip turn
    prefetch_mode: str = "bm25"            # "bm25" (fast) or "reranker" (accurate)
    tool_name: str = "synapse_query"       # merged tool name

    # Consolidation
    consolidation_interval_hours: float = 6.0

    # Write behavior — for slow/local LLM backends
    sync_writes: bool = False              # wait for Graphiti writes instead of background detach
    hydrate_on_init: bool = False          # preload explicit memories into the prompt cache at init
    llm_temperature: float | None = None   # None = client/library default (upstream behavior)
    structured_output_mode: str = "json_object"

    # Operation timeouts (seconds) — raise for slow local LLMs
    init_timeout: int = 30                 # build_indices_and_constraints()
    episode_timeout: int = 120             # add_episode() (turn-batch ingestion)
    remember_timeout: int = 60             # add_episode() (explicit remember)
    memory_write_timeout: int = 30         # add_episode() (native-memory mirror)
    close_timeout: int = 10                # graphiti.close()

    @classmethod
    def from_env(cls) -> SynapseConfig:
        """Create config from environment variables (SYNAPSE_* prefix)."""
        temperature_env = os.environ.get("SYNAPSE_LLM_TEMPERATURE")
        return cls(
            falkordb_host=os.environ.get("SYNAPSE_FALKORDB_HOST", "localhost"),
            falkordb_port=int(os.environ.get("SYNAPSE_FALKORDB_PORT", "6379")),
            falkordb_password=os.environ.get("SYNAPSE_FALKORDB_PASSWORD"),
            falkordb_database=os.environ.get("SYNAPSE_DATABASE", "synapse"),
            llm_api_key=os.environ.get("SYNAPSE_LLM_API_KEY", ""),
            llm_base_url=os.environ.get("SYNAPSE_LLM_BASE_URL", ""),
            embedding_base_url=os.environ.get("SYNAPSE_EMBEDDING_BASE_URL", ""),
            llm_model=os.environ.get("SYNAPSE_LLM_MODEL", "gpt-4o-mini"),
            embedding_model=os.environ.get("SYNAPSE_EMBEDDING_MODEL", "text-embedding-3-small"),
            batch_size=int(os.environ.get("SYNAPSE_BATCH_SIZE", "5")),
            half_life_days=float(os.environ.get("SYNAPSE_HALF_LIFE_DAYS", "7.0")),
            sync_writes=os.environ.get("SYNAPSE_SYNC_WRITES", "").lower() in ("1", "true", "yes"),
            hydrate_on_init=os.environ.get("SYNAPSE_HYDRATE_ON_INIT", "").lower() in ("1", "true", "yes"),
            llm_temperature=float(temperature_env) if temperature_env else None,
            structured_output_mode=os.environ.get("SYNAPSE_STRUCTURED_OUTPUT_MODE", "json_object"),
            init_timeout=int(os.environ.get("SYNAPSE_INIT_TIMEOUT", "30")),
            episode_timeout=int(os.environ.get("SYNAPSE_EPISODE_TIMEOUT", "120")),
            remember_timeout=int(os.environ.get("SYNAPSE_REMEMBER_TIMEOUT", "60")),
            memory_write_timeout=int(os.environ.get("SYNAPSE_MEMORY_WRITE_TIMEOUT", "30")),
            close_timeout=int(os.environ.get("SYNAPSE_CLOSE_TIMEOUT", "10")),
        )
