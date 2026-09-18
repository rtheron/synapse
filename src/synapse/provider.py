"""SynapseMemoryProvider — full MemoryProvider ABC implementation.

Optimizations baked in:
- Batch episode ingestion (5 turns/episode) via TurnBuffer
- BM25-only prefetch (no reranker) via RetrievalEngine
- Background prefetch caching (zero blocking)
- Merged tool schema (synapse_query, 76 tokens)
- Minimal system prompt (15 tokens)
- Trivial turn skipping (<10 chars)
- Configurable half-life
- Hippocampus coordinator wired into episode ingestion + recall
- Bounded graph fetch (get_recent_edges) for post-episode processing
- Pattern completion (CA3 BFS) wired into synapse_query results
- Retrieval-induced forgetting (RIF) — competing memories sink in ranking
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SynapseMemoryProvider:
    """Temporal knowledge graph memory provider for Hermes Agent.

    Designed to implement the Hermes MemoryProvider ABC when installed
    as a plugin. The ABC methods match the Hermes agent.memory_provider
    interface exactly.

    When native MEMORY.md/USER.md are disabled, Synapse becomes the
    agent's primary memory system. The system_prompt_block() detects
    this and instructs the agent to use synapse_remember for explicit
    facts and synapse_query for recall — replacing the dead 'memory'
    tool path.
    """

    def __init__(self):
        self._config = None
        self._graphiti = None
        self._driver = None
        self._retrieval = None
        self._turn_buffer = None
        self._hippocampus = None
        self._session_id = ""
        self._group_id = "default"
        self._initialized = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bg_thread: Optional[threading.Thread] = None
        self._native_memory_active = False
        self._remembered_facts: list[dict] = []  # cached for system prompt

    @property
    def name(self) -> str:
        return "synapse"

    # -- Core lifecycle ------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if FalkorDB connection info and LLM credentials are configured."""
        return bool(
            os.environ.get("SYNAPSE_FALKORDB_HOST")
            and os.environ.get("SYNAPSE_LLM_API_KEY")
            and os.environ.get("SYNAPSE_LLM_BASE_URL")
        )

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize Graphiti connection and build indices."""
        from synapse.config import SynapseConfig
        from synapse.encoding import TurnBuffer
        from synapse.retrieval import RetrievalEngine

        self._config = SynapseConfig.from_env()
        self._session_id = session_id
        self._group_id = kwargs.get("agent_identity", "default")

        # Initialize Graphiti (for episode ingestion)
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        self._driver = FalkorDriver(
            host=self._config.falkordb_host,
            port=self._config.falkordb_port,
            password=self._config.falkordb_password,
            database=self._config.falkordb_database,
        )
        llm_kwargs = dict(
            api_key=self._config.llm_api_key,
            model=self._config.llm_model,
            base_url=self._config.llm_base_url,
            small_model=self._config.llm_model,
        )
        if self._config.llm_temperature is not None:
            llm_kwargs["temperature"] = self._config.llm_temperature
        llm_config = LLMConfig(**llm_kwargs)
        self._graphiti = Graphiti(
            graph_driver=self._driver,
            # ponytail: json_object works on all OpenAI-compatible endpoints
            # (DeepSeek, OpenRouter, vLLM, etc.); json_schema only works on
            # api.openai.com + constrained-decoding servers.
            # Set SYNAPSE_STRUCTURED_OUTPUT_MODE=json_schema if needed.
            llm_client=OpenAIGenericClient(
                config=llm_config,
                structured_output_mode=self._config.structured_output_mode,
            ),
            embedder=OpenAIEmbedder(config=OpenAIEmbedderConfig(
                api_key=self._config.llm_api_key,
                base_url=self._config.llm_base_url,
                embedding_model=self._config.embedding_model,
            )),
            cross_encoder=OpenAIRerankerClient(config=llm_config),
        )

        # Background asyncio loop for async operations
        self._loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(target=self._run_bg_loop, daemon=True)
        self._bg_thread.start()

        # Build indices
        future = asyncio.run_coroutine_threadsafe(
            self._graphiti.build_indices_and_constraints(), self._loop
        )
        future.result(timeout=self._timeout_seconds(30))

        # Initialize retrieval engine (BM25-only, no Graphiti needed)
        self._retrieval = RetrievalEngine(
            host=self._config.falkordb_host,
            port=self._config.falkordb_port,
            group_id=self._group_id,
            password=self._config.falkordb_password,
        )

        # Hydrate the prompt cache from the graph — one-time per init.
        # system_prompt_block() reads _remembered_facts (RAM only); without this
        # step every process start renders an empty block no matter what the
        # graph holds. Facts come from Explicit episode content, NOT RELATES_TO
        # edges — identity channel, not recall channel.
        if self._config.hydrate_on_init:
            try:
                from synapse.falkor import FalkorHelper
                _hydrate_graph = FalkorHelper(
                    host=self._config.falkordb_host,
                    port=self._config.falkordb_port,
                    password=self._config.falkordb_password,
                ).get_graph(self._group_id)
                _loaded = {"user_profile": [], "environment": []}
                for _cat in _loaded:
                    _res = _hydrate_graph.query(
                        "MATCH (e:Episodic) WHERE e.name = $name "
                        "RETURN e.content ORDER BY e.valid_at DESC LIMIT 10",
                        {"name": f"Explicit memory: {_cat}"},
                    )
                    _seen = set()
                    if _res and _res.result_set:
                        for _row in _res.result_set:
                            _content = _row[0]
                            if _content and _content not in _seen:
                                _seen.add(_content)
                                _loaded[_cat].append({
                                    "content": _content,
                                    "category": _cat,
                                    "timestamp": "hydrated-from-graph",
                                })
                for _cat in _loaded:
                    self._remembered_facts.extend(_loaded[_cat])
                _bullets = [f"- {f['content']}" for _fs in _loaded.values() for f in _fs]
                logger.warning(
                    "Synapse hydrate OK: %d user_profile + %d environment facts:\n%s",
                    len(_loaded["user_profile"]), len(_loaded["environment"]),
                    "\n".join(_bullets) if _bullets else "(none found)",
                )
            except Exception as _e:
                # Covered and non-fatal — provider still initializes without hydration
                logger.warning("Synapse hydrate FAILED (non-fatal): %s", _e)

        # Initialize turn buffer (batch ingestion)
        self._turn_buffer = TurnBuffer(
            batch_size=self._config.batch_size,
            trivial_threshold=self._config.trivial_turn_threshold,
        )

        # Initialize hippocampus coordinator (wires all 9 algorithms)
        from synapse.hippocampus import Hippocampus
        self._hippocampus = Hippocampus(
            half_life_days=self._config.half_life_days,
            salience_boost=self._config.salience_boost,
            recall_boost=self._config.recall_boost,
            prune_threshold=self._config.prune_threshold,
            group_id=self._group_id,
        )

        self._initialized = True
        self._native_memory_active = False

        logger.info(f"Synapse initialized (session={session_id}, group={self._group_id})")

    def _run_bg_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _env_bool(self, name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.lower() in ("1", "true", "yes")

    def _write_failure_log_level(self) -> str:
        level = os.environ.get("SYNAPSE_WRITE_FAILURE_LOG_LEVEL", "warning").lower()
        return level if level in ("warning", "error") else "warning"

    def _timeout_log_level(self) -> str:
        level = os.environ.get("SYNAPSE_TIMEOUT_LOG_LEVEL", "0").lower()
        return level if level in ("0", "warn", "error") else "0"

    def _retry_queue_dir(self) -> Optional[str]:
        return os.environ.get("SYNAPSE_RETRY_QUEUE_DIR") or None

    def _surface_not_ready_failures(self) -> bool:
        """Whether to surface write drops that upstream kept silent."""
        return self._env_bool("SYNAPSE_SURFACE_WRITE_FAILURES") or self._retry_queue_dir() is not None

    def _timeout_multiplier(self) -> float:
        try:
            return float(os.environ.get("SYNAPSE_TIMEOUT_MULTIPLIER", "1"))
        except ValueError:
            return 1.0

    def _timeout_seconds(self, base_seconds: int) -> int:
        return int(base_seconds * self._timeout_multiplier())

    def _build_write_request(
        self,
        operation: str,
        *,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        timeout: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the exact serializable write request submitted to Graphiti."""
        request: Dict[str, Any] = {
            "operation": operation,
            "name": name,
            "episode_body": episode_body,
            "source_description": source_description,
            "reference_time": reference_time.isoformat(),
            "source": "message",
            "group_id": self._group_id,
            "timeout": timeout,
        }
        if metadata is not None:
            request["metadata"] = metadata
        return request

    def _submit_write_request(self, request: Dict[str, Any]) -> None:
        """Submit a queued/current write request to Graphiti.

        Raises on failure so callers can route through _record_write_failure().
        """
        if not self._graphiti or not self._loop:
            raise RuntimeError("graphiti/loop not ready")

        from graphiti_core.nodes import EpisodeType

        future = asyncio.run_coroutine_threadsafe(
            self._graphiti.add_episode(
                name=request["name"],
                episode_body=request["episode_body"],
                source_description=request["source_description"],
                reference_time=datetime.fromisoformat(request["reference_time"]),
                source=EpisodeType.message,
                group_id=request["group_id"],
            ),
            self._loop,
        )
        future.result(timeout=request["timeout"])

    def _queue_write_failure(
        self,
        operation: str,
        error: BaseException,
        request: Dict[str, Any],
    ) -> None:
        retry_queue_dir = self._retry_queue_dir()
        if retry_queue_dir is None:
            return
        try:
            queue_path = Path(retry_queue_dir)
            queue_path.mkdir(parents=True, exist_ok=True)

            failed_at = datetime.now(timezone.utc)
            canonical_request = json.dumps(
                request,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            digest = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()[:16]
            operation_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", operation).strip("_") or "write"
            filename = (
                f"{failed_at.strftime('%Y%m%dT%H%M%S%fZ')}_"
                f"{operation_safe}_{digest}_{uuid.uuid4().hex[:8]}.json"
            )
            final_path = queue_path / filename
            tmp_path = queue_path / f"{filename}.tmp"
            payload = {
                "failed_at": failed_at.isoformat(),
                "operation": operation,
                "error": str(error),
                "error_type": type(error).__name__,
                "request": request,
            }
            tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            os.replace(tmp_path, final_path)
        except Exception as queue_error:
            logger.error("Synapse failed-write queue error: %s", queue_error)

    def _record_write_failure(
        self,
        operation: str,
        error: BaseException,
        request: Dict[str, Any],
    ) -> None:
        """Surface a non-timeout write failure and optionally persist it."""
        log = logger.error if self._write_failure_log_level() == "error" else logger.warning
        log("Synapse %s failed: %s", operation, error)
        self._queue_write_failure(operation, error, request)

    def _record_timeout_failure(self, operation: str, request: Dict[str, Any]) -> None:
        """Record a timeout; timeout logging is controlled separately."""
        error = TimeoutError(f"add_episode exceeded {request['timeout']}s")
        level = self._timeout_log_level()
        if level == "warn":
            logger.warning("Synapse %s timed out: %s", operation, error)
        elif level == "error":
            logger.error("Synapse %s timed out: %s", operation, error)
        self._queue_write_failure(operation, error, request)

    def system_prompt_block(self) -> str:
        """Static info for the system prompt — brain-aware."""
        if not self._initialized:
            return ""

        if self._native_memory_active:
            return "Synapse memory active. Use synapse_query for past memories.\n"

        lines = [
            "Synapse memory active — you have a persistent temporal knowledge graph.",
            "Use synapse_remember to save important facts (replaces the memory tool).",
            "Use synapse_query to recall past conversations and facts.",
        ]

        profile_facts = [
            f for f in self._remembered_facts
            if f.get("category") == "user_profile"
        ]
        if profile_facts:
            lines.append("")
            lines.append("What you know about your user:")
            for fact in profile_facts[:10]:
                lines.append(f"- {fact['content']}")

        env_facts = [
            f for f in self._remembered_facts
            if f.get("category") == "environment"
        ]
        if env_facts:
            lines.append("")
            lines.append("What you know about your environment:")
            for fact in env_facts[:10]:
                lines.append(f"- {fact['content']}")

        return "\n".join(lines) + "\n"

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached context for the current query (zero blocking)."""
        if not self._initialized or not self._retrieval:
            return ""
        return self._retrieval.prefetch(query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Run background search and cache result for next turn."""
        if self._initialized and self._retrieval:
            self._retrieval.queue_prefetch(query)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Buffer a completed turn for batch episode ingestion."""
        if not self._initialized or not self._turn_buffer:
            return
        self._turn_buffer.add(user_content, assistant_content)
        if self._hippocampus:
            self._hippocampus.tick()
        if self._turn_buffer.should_flush():
            self._ingest_episode()

    def _ingest_episode(self) -> None:
        """Flush the turn buffer and ingest as a batch episode."""
        if not self._turn_buffer:
            return
        if (not self._graphiti or not self._loop) and not self._surface_not_ready_failures():
            return

        episode = self._turn_buffer.flush()
        if not episode:
            return

        request = self._build_write_request(
            "episode_ingest",
            name=episode["name"],
            episode_body=episode["body"],
            source_description="Hermes agent conversation",
            reference_time=episode["reference_time"],
            timeout=self._timeout_seconds(120),
            metadata={"turn_count": episode.get("turn_count")},
        )

        def _bg_ingest():
            try:
                self._submit_write_request(request)
            except TimeoutError:
                self._record_timeout_failure("episode_ingest", request)
                return
            except Exception as e:
                self._record_write_failure("episode_ingest", e, request)
                return

            if self._hippocampus:
                try:
                    from synapse.falkor import FalkorHelper
                    helper = FalkorHelper(
                        host=self._config.falkordb_host,
                        port=self._config.falkordb_port,
                        password=self._config.falkordb_password,
                    )
                    recent_edges = helper.get_recent_edges(
                        self._group_id, limit=50
                    )
                    existing_entities = [
                        e.get("name", "") for e in helper.get_entities(self._group_id)
                    ]
                    self._hippocampus.on_episode_ingested(
                        new_entities=[],
                        existing_entities=existing_entities,
                        new_edges=[],
                        existing_edges=recent_edges,
                    )
                except Exception as e:
                    logger.warning("Synapse post-episode processing failed: %s", e)

        sync = self._config.sync_writes if self._config else False
        if sync:
            # Sequenced write — waits for Graphiti so slow local LLMs are not
            # flooded with parallel episode writes.
            _bg_ingest()
        elif self._graphiti and self._loop:
            threading.Thread(target=_bg_ingest, daemon=True).start()
        else:
            _bg_ingest()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return all Synapse tool schemas (synapse_query + synapse_remember).

        Tool schemas are static module-level constants in tools.py — they
        do not depend on Graphiti/FalkorDB being initialized. Hermes calls
        this method at registration time, before initialize(), so we must
        return schemas unconditionally.
        """
        from synapse.tools import get_all_tool_schemas
        return get_all_tool_schemas()

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle synapse_query and synapse_remember tool calls."""
        if not self._initialized:
            return json.dumps({
                "error": (
                    f"Cannot handle '{tool_name}' — Synapse is not initialized. "
                    "Ensure FalkorDB is running and SYNAPSE_* env vars are set."
                ),
            })

        if tool_name == "synapse_query" and self._retrieval:
            from synapse.tools import handle_tool_call
            result = handle_tool_call(args, self._retrieval)
            if self._hippocampus:
                try:
                    result_dict = json.loads(result)
                    bm25_results = result_dict.get("results", [])
                    entities = set()
                    for r in bm25_results:
                        entities.add(r.get("from_node", ""))
                        entities.add(r.get("to_node", ""))
                    entities.discard("")

                    neighborhood_edges: list[dict] = []

                    # Pattern completion: expand BM25 results into neighborhood
                    if entities and self._hippocampus:
                        from synapse.falkor import FalkorHelper
                        helper = FalkorHelper(
                            host=self._config.falkordb_host,
                            port=self._config.falkordb_port,
                            password=self._config.falkordb_password,
                        )
                        neighborhood_edges = helper.get_entity_neighborhood(
                            self._group_id, list(entities), depth=1, limit=20
                        )
                        if neighborhood_edges:
                            expanded = self._hippocampus.expand_recall(
                                args.get("query", ""), neighborhood_edges
                            )
                            expanded_facts = expanded.get("facts", [])
                            existing_facts = {r.get("fact", "") for r in bm25_results}
                            for ef in expanded_facts:
                                fact = ef.get("fact", "")
                                if fact and fact not in existing_facts:
                                    bm25_results.append(ef)
                                    existing_facts.add(fact)
                            result_dict["pattern_completion"] = {
                                "expanded_entities": expanded.get("entities", []),
                                "depth": expanded.get("depth", 0),
                            }

                    # Reconsolidation + RIF: pass neighborhood edges for Jaccard
                    if entities:
                        self._hippocampus.on_recall(
                            list(entities), edges=neighborhood_edges or None
                        )

                    # RIF ranking: penalized entities sink in results
                    if self._hippocampus and bm25_results:
                        def rif_sort_key(r):
                            from_p = self._hippocampus.get_rif_penalty(
                                r.get("from_node", "")
                            )
                            to_p = self._hippocampus.get_rif_penalty(
                                r.get("to_node", "")
                            )
                            # Penalty = max of from/to — both endpoints matter
                            return -(max(from_p, to_p))  # negative = lower rank

                        bm25_results.sort(key=rif_sort_key)
                        result_dict["results"] = bm25_results[:20]

                    result = json.dumps(result_dict)
                except Exception:
                    pass
            return result

        if tool_name == "synapse_remember":
            from synapse.tools import handle_remember
            return handle_remember(args, self._store_remembered_fact)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _store_remembered_fact(self, content: str, category: str) -> bool:
        """Store an explicit fact in the graph with maximum salience.

        Default (upstream) behavior: cache in RAM, detach the graph write on
        a background thread, always return True. With SYNAPSE_SYNC_WRITES
        enabled, waits for add_episode (no background detach) so failures
        return to the tool and agents cannot flood parallel Graphiti writes.
        """
        request = self._build_write_request(
            "remember",
            name=f"Explicit memory: {category}",
            episode_body=content,
            source_description=f"Agent explicit memory write ({category})",
            reference_time=datetime.now(timezone.utc),
            timeout=self._timeout_seconds(60),
        )

        def _write() -> bool:
            try:
                self._submit_write_request(request)
                return True
            except TimeoutError:
                self._record_timeout_failure("remember", request)
                return False
            except Exception as e:
                self._record_write_failure("remember", e, request)
                return False

        sync = self._config.sync_writes if self._config else False
        if sync:
            if (
                (not self._graphiti or not self._loop)
                and not self._surface_not_ready_failures()
            ):
                logger.warning("Synapse remember failed: graphiti/loop not ready")
                return False
            if not _write():
                return False
            self._remembered_facts.append({
                "content": content,
                "category": category,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return True

        self._remembered_facts.append({
            "content": content,
            "category": category,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if self._graphiti and self._loop:
            threading.Thread(target=_write, daemon=True).start()
        elif self._surface_not_ready_failures():
            _write()
        return True

    def shutdown(self) -> None:
        """Clean shutdown — flush remaining turns, close Graphiti."""
        if self._turn_buffer and len(self._turn_buffer) > 0:
            self._ingest_episode()

        if self._hippocampus:
            self._hippocampus.clear()

        if self._graphiti and self._loop:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._graphiti.close(), self._loop
                )
                future.result(timeout=self._timeout_seconds(10))
            except Exception:
                pass

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._initialized = False

    # -- Optional hooks -----------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Flush remaining turns on session end."""
        if self._turn_buffer and len(self._turn_buffer) > 0:
            self._ingest_episode()
        if self._hippocampus:
            self._hippocampus.clear()

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes to the graph."""
        if not self._initialized:
            return

        if not self._native_memory_active:
            self._native_memory_active = True
            logger.debug("Synapse: native memory detected — switching to supplementary mode")

        if (not self._graphiti or not self._loop) and not self._surface_not_ready_failures():
            return

        request = self._build_write_request(
            "memory_write",
            name=f"Memory: {action} {target}",
            episode_body=content,
            source_description=f"Hermes memory write ({action} on {target})",
            reference_time=datetime.now(timezone.utc),
            timeout=self._timeout_seconds(30),
            metadata=metadata,
        )

        def _bg_write():
            try:
                self._submit_write_request(request)
            except TimeoutError:
                self._record_timeout_failure("memory_write", request)
            except Exception as e:
                self._record_write_failure("memory_write", e, request)

        sync = self._config.sync_writes if self._config else False
        if sync:
            # Sequenced write — waits for Graphiti (no background detach).
            _bg_write()
        elif self._graphiti and self._loop:
            threading.Thread(target=_bg_write, daemon=True).start()
        else:
            _bg_write()

    # -- Config --------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Config fields for 'hermes memory setup' wizard."""
        return [
            {"key": "falkordb_host", "description": "FalkorDB host", "required": True,
             "env_var": "SYNAPSE_FALKORDB_HOST", "default": "localhost"},
            {"key": "falkordb_port", "description": "FalkorDB port", "required": False,
             "env_var": "SYNAPSE_FALKORDB_PORT", "default": "6379"},
            {"key": "falkordb_password", "description": "FalkorDB password", "required": False,
             "secret": True, "env_var": "SYNAPSE_FALKORDB_PASSWORD"},
            {"key": "llm_api_key", "description": "LLM API key (OpenAI-compatible)",
             "required": True, "secret": True, "env_var": "SYNAPSE_LLM_API_KEY"},
            {"key": "llm_base_url", "description": "LLM base URL", "required": True,
             "env_var": "SYNAPSE_LLM_BASE_URL"},
            {"key": "llm_model", "description": "LLM model", "required": False,
             "env_var": "SYNAPSE_LLM_MODEL", "default": "gpt-4o-mini"},
            {"key": "embedding_model", "description": "Embedding model", "required": False,
             "env_var": "SYNAPSE_EMBEDDING_MODEL", "default": "text-embedding-3-small"},
            {"key": "batch_size", "description": "Turns per episode (optimization)",
             "required": False, "env_var": "SYNAPSE_BATCH_SIZE", "default": "5"},
            {"key": "half_life_days", "description": "Forgetting curve half-life in days",
             "required": False, "env_var": "SYNAPSE_HALF_LIFE_DAYS", "default": "7.0"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to ~/.hermes/synapse.json."""
        config_path = Path(hermes_home) / "synapse.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        non_secret = {
            k: v for k, v in values.items()
            if k not in ("falkordb_password", "llm_api_key")
        }
        config_path.write_text(json.dumps(non_secret, indent=2))

    def backup_paths(self) -> List[str]:
        """No external paths — FalkorDB data is in Docker."""
        return []


def register(ctx):
    """Register the Synapse memory provider (Hermes plugin entry point)."""
    ctx.register_memory_provider(SynapseMemoryProvider())
