from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from ..models import ModelRegistry, get_model_config, get_model_tier

logger = logging.getLogger(__name__)


def _run(coro):
    """Run async coroutine, handling Jupyter/Colab event loop conflicts."""
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        # In Jupyter/Colab environments, there's already an event loop running
        if "asyncio.run() cannot be called" not in str(exc):
            raise
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


def _patch_json_parser():
    """Patch itext2kg's JSON parser to handle malformed LLM outputs.

    This monkey-patches the langchain JSON parsing to be more resilient
    to common LLM output errors.
    """
    try:
        from langchain_core.utils import json as lc_json
        from ..utils.json_repair import repair_json

        original_parse_json_markdown = lc_json.parse_json_markdown

        @functools.wraps(original_parse_json_markdown)
        def patched_parse_json_markdown(json_string: str, *, parser: Callable = json.loads):
            """Enhanced JSON parser with repair capabilities."""
            # First try the original parser
            try:
                return original_parse_json_markdown(json_string, parser=parser)
            except Exception:
                pass

            # Try our repair logic
            logger.debug("Original JSON parsing failed, attempting repair...")
            result = repair_json(json_string)
            if result is not None:
                logger.debug("JSON repair successful")
                return result

            # If repair also failed, raise the original error
            # by calling the original function
            return original_parse_json_markdown(json_string, parser=parser)

        # Apply the patch
        lc_json.parse_json_markdown = patched_parse_json_markdown
        logger.info("Applied JSON parser patch for better LLM output handling")
        return True

    except Exception as e:
        logger.warning(f"Could not patch JSON parser: {e}")
        return False


def _patch_itext2kg_provider_detection():
    """Patch itext2kg to recognize ChatOllama and OllamaEmbeddings as Ollama provider.

    itext2kg internally checks the type of LLM/embeddings objects to determine
    the provider. When it doesn't recognize ChatOllama (from langchain_ollama),
    it falls back to "Unknown" provider which may trigger OpenAI key requirements.

    This patch adds recognition for:
    - langchain_ollama.ChatOllama -> "ollama"
    - langchain_ollama.OllamaEmbeddings -> "ollama"
    - langchain_community.chat_models.ollama.ChatOllama -> "ollama"
    - langchain_community.embeddings.ollama.OllamaEmbeddings -> "ollama"
    """
    try:
        try:
            import itext2kg.utils.llm as llm_utils
        except ImportError:
            llm_utils = None

        if llm_utils is not None:
            # Get the original provider detection function
            original_get_provider = getattr(llm_utils, 'get_provider', None)
            if original_get_provider is not None:
                @functools.wraps(original_get_provider)
                def patched_get_provider(llm_model):
                    """Enhanced provider detection that recognizes ChatOllama."""
                    try:
                        from langchain_ollama import ChatOllama
                    except ImportError:
                        ChatOllama = None

                    try:
                        from langchain_community.chat_models.ollama import ChatOllama as CommunityChatOllama
                    except ImportError:
                        CommunityChatOllama = None

                    if ChatOllama and isinstance(llm_model, ChatOllama):
                        return "ollama"

                    if CommunityChatOllama and isinstance(llm_model, CommunityChatOllama):
                        return "ollama"

                    return original_get_provider(llm_model)

                llm_utils.get_provider = patched_get_provider
                logger.info("Applied itext2kg provider detection patch for ChatOllama")
                return True

        from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser, ProviderType

        original_detect_provider = LangchainOutputParser._detect_provider

        @functools.wraps(original_detect_provider)
        def patched_detect_provider(self):
            try:
                from langchain_ollama import ChatOllama, OllamaEmbeddings
            except ImportError:
                ChatOllama = None
                OllamaEmbeddings = None

            model = getattr(self, "model", None)
            embeddings_model = getattr(self, "embeddings_model", None)

            # Route Ollama to CLAUDE provider type — no rate limiting,
            # no API key check, high batch size
            if ChatOllama and isinstance(model, ChatOllama):
                return ProviderType.CLAUDE        # ← change ANTHROPIC to CLAUDE
            if OllamaEmbeddings and isinstance(embeddings_model, OllamaEmbeddings):
                return ProviderType.CLAUDE        # ← change ANTHROPIC to CLAUDE

            return original_detect_provider(self)

        LangchainOutputParser._detect_provider = patched_detect_provider
        logger.info("Applied itext2kg provider detection patch for newer LangchainOutputParser layout")
        return True

    except Exception as e:
        logger.warning(f"Could not patch itext2kg provider detection: {e}")
        return False


def _patch_itext2kg_for_empty_results():
    """Patch itext2kg to handle empty atomic KG lists gracefully.

    Patches three methods on the Atom class:
    1. parallel_atomic_merge  - handles empty KG lists (``current[0]`` bug)
    2. build_atomic_kg_from_quintuples - catches IndexError from entity
       look-ups / embedding failures and returns an empty KG instead
    3. build_graph - uses ``return_exceptions=True`` in asyncio.gather so
       that one bad quintuple doesn't kill the whole batch, and tolerates
       an all-empty atomic-KG list gracefully
    """
    try:
        import itext2kg.atom.atom as atom_module
        from itext2kg.atom.models.knowledge_graph import KnowledgeGraph

        # --- 1. parallel_atomic_merge: guard against empty kgs list -------
        original_merge = atom_module.Atom.parallel_atomic_merge

        @functools.wraps(original_merge)
        def safe_parallel_atomic_merge(
            self, kgs, existing_kg=None, rel_threshold=0.7, ent_threshold=0.8, max_workers=8
        ):
            if not kgs:
                logger.warning("No atomic KGs to merge (empty list). Returning empty KG.")
                return KnowledgeGraph()

            valid_kgs = [kg for kg in kgs if kg is not None]
            if not valid_kgs:
                logger.warning("All atomic KGs are None. Returning empty KG.")
                return KnowledgeGraph()

            return original_merge(
                self, valid_kgs, existing_kg, rel_threshold, ent_threshold, max_workers
            )

        atom_module.Atom.parallel_atomic_merge = safe_parallel_atomic_merge

        # --- 2. build_atomic_kg_from_quintuples: catch per-quintuple errors
        original_build_atomic = atom_module.Atom.build_atomic_kg_from_quintuples

        @functools.wraps(original_build_atomic)
        async def safe_build_atomic_kg_from_quintuples(self, relationships, *args, **kwargs):
            if not relationships:
                logger.debug("Empty relationships list for quintuple - returning empty KG.")
                return KnowledgeGraph()
            try:
                return await original_build_atomic(self, relationships, *args, **kwargs)
            except (IndexError, ValueError, TypeError) as exc:
                logger.warning(
                    "build_atomic_kg_from_quintuples failed for a quintuple "
                    "(likely malformed entities or embedding failure): %s", exc,
                )
                return KnowledgeGraph()

        atom_module.Atom.build_atomic_kg_from_quintuples = safe_build_atomic_kg_from_quintuples

        # --- 3. build_graph: fault-tolerant asyncio.gather ----------------
        original_build_graph = atom_module.Atom.build_graph

        @functools.wraps(original_build_graph)
        async def safe_build_graph(self, atomic_facts, obs_timestamp, **kwargs):
            """Wraps build_graph to survive individual quintuple failures."""
            try:
                return await original_build_graph(
                    self, atomic_facts, obs_timestamp, **kwargs
                )
            except (IndexError, ValueError, TypeError) as exc:
                logger.warning(
                    "build_graph failed for timestamp %s: %s. Returning empty KG.",
                    obs_timestamp, exc,
                )
                return KnowledgeGraph()

        atom_module.Atom.build_graph = safe_build_graph

        logger.info("Applied itext2kg patches (parallel_atomic_merge, "
                    "build_atomic_kg_from_quintuples, build_graph)")
        return True

    except Exception as e:
        logger.warning(f"Could not patch itext2kg for empty results: {e}")
        return False


# Apply patches when module loads
_patches_applied = False


def _ensure_patches():
    """Ensure all patches are applied (idempotent)."""
    global _patches_applied
    if not _patches_applied:
        _patch_json_parser()
        _patch_itext2kg_provider_detection()
        _patch_itext2kg_for_empty_results()
        _patches_applied = True


async def _build_async(
    *,
    atomic_facts_dict: Dict[str, List[str]],
    ollama_base_url: str,
    embeddings_ollama_base_url: Optional[str],
    llm_model: str,
    embeddings_model: str,
    temperature: float,
    ent_threshold: float,
    rel_threshold: float,
    max_workers: int,
    api_key: Optional[str] = None,
    embeddings_api_key: Optional[str] = None,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    embeddings_kwargs: Optional[Dict[str, Any]] = None,
):
    # Ensure patches are applied
    _ensure_patches()

    try:
        import itext2kg  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "itext2kg is not installed in this Python environment. "
            "Install dependencies first (see DynamicKGConstruction/requirements.txt)."
        ) from e

    from itext2kg.atom import Atom
    from itext2kg.atom.models.knowledge_graph import KnowledgeGraph

    # Get model-specific configuration via centralized registry
    model_config = get_model_config(llm_model)
    model_tier = get_model_tier(llm_model)
    logger.info(f"Using model tier '{model_tier}' configuration for {llm_model}")

    # Build LLM and embeddings via ModelRegistry (provider auto-detected)
    llm = ModelRegistry.create_llm(
        llm_model,
        temperature=temperature,
        api_key=api_key,
        ollama_base_url=ollama_base_url,
        **(llm_kwargs or {}),
    )
    embeddings = ModelRegistry.create_embeddings(
        embeddings_model,
        api_key=embeddings_api_key,
        ollama_base_url=embeddings_ollama_base_url or ollama_base_url,
        **(embeddings_kwargs or {}),
    )

    # Log what we're processing
    total_facts = sum(len(facts) for facts in atomic_facts_dict.values())
    logger.info(f"Building KG from {total_facts} atomic facts across {len(atomic_facts_dict)} timestamps")

    atom = Atom(llm_model=llm, embeddings_model=embeddings)

    # Retry logic for the entire KG building process
    max_attempts = model_config["max_retries"]
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            kg = await atom.build_graph_from_different_obs_times(
                atomic_facts_with_obs_timestamps=atomic_facts_dict,
                ent_threshold=ent_threshold,
                rel_threshold=rel_threshold,
                max_workers=max_workers,
            )

            # Validate result
            if kg and not kg.is_empty():
                logger.info(
                    f"Successfully built KG with {len(kg.entities)} entities "
                    f"and {len(kg.relationships)} relationships"
                )
                return kg
            else:
                logger.warning(f"Attempt {attempt}: KG is empty, may retry...")
                if attempt < max_attempts:
                    await asyncio.sleep(model_config["retry_delay"])
                    continue
                return kg

        except IndexError as e:
            last_error = e
            logger.warning(
                f"Attempt {attempt}/{max_attempts}: IndexError during atomic KG "
                f"building - likely no valid entities/relations extracted. "
                f"This can happen when: (1) LLM returns quintuples that can't be "
                f"parsed into entities, (2) embeddings fail to match entities, "
                f"(3) the model provider is not recognized by itext2kg (check for "
                f"'Unknown provider' warnings above). Error: {e}"
            )
            if attempt < max_attempts:
                await asyncio.sleep(model_config["retry_delay"])
                continue

        except Exception as e:
            last_error = e
            error_msg = str(e)

            # Check for JSON parsing errors (common with smaller models)
            if "json" in error_msg.lower() or "parse" in error_msg.lower():
                logger.warning(
                    f"Attempt {attempt}: JSON parsing error (model may need repair): {e}"
                )
                if attempt < max_attempts:
                    await asyncio.sleep(model_config["retry_delay"])
                    continue
            else:
                logger.error(f"Attempt {attempt}: Unexpected error: {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(model_config["retry_delay"])
                    continue

    # All retries exhausted
    logger.warning(
        f"All {max_attempts} attempts failed. Returning empty KG. Last error: {last_error}"
    )
    return KnowledgeGraph()


def build_kg_from_atomic_facts(
    *,
    atomic_facts_dict: Dict[str, List[str]],
    ollama_base_url: str,
    embeddings_ollama_base_url: Optional[str] = None,
    llm_model: str,
    embeddings_model: str,
    temperature: float = 0.0,
    ent_threshold: float = 0.8,
    rel_threshold: float = 0.7,
    max_workers: int = 4,
    api_key: Optional[str] = None,
    embeddings_api_key: Optional[str] = None,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    embeddings_kwargs: Optional[Dict[str, Any]] = None,
):
    """Build a KnowledgeGraph using itext2kg ATOM (async under the hood).

    Provider is determined automatically from *llm_model* and *embeddings_model*
    by :class:`~skgb.models.ModelRegistry`.  Supported combinations:

    * Ollama LLM + Ollama embeddings (fully local)
    * Claude LLM + Ollama embeddings  (quality + privacy)
    * OpenAI LLM + OpenAI embeddings
    * Any mix of the above

    Args:
        atomic_facts_dict: Dict mapping observation timestamps to lists of atomic facts.
        ollama_base_url: Base URL for the Ollama LLM server.
        embeddings_ollama_base_url: Optional base URL for Ollama embeddings. When
            omitted, defaults to ``ollama_base_url``.
        llm_model: LLM model name (e.g. ``"claude-sonnet-4-6"``, ``"qwen2.5:32b"``).
        embeddings_model: Embeddings model name (e.g. ``"nomic-embed-text"``).
        temperature: LLM temperature (0.0 for deterministic).
        ent_threshold: Entity similarity threshold for deduplication.
        rel_threshold: Relation similarity threshold for deduplication.
        max_workers: Number of parallel workers.
        api_key: API key for cloud LLM providers (Anthropic / OpenAI).
        embeddings_api_key: API key for cloud embeddings providers (OpenAI).
        llm_kwargs: Optional provider-specific kwargs forwarded to the LLM factory.
        embeddings_kwargs: Optional provider-specific kwargs forwarded to the embeddings factory.

    Returns:
        KnowledgeGraph object (may be empty if extraction failed).
    """
    try:
        return _run(
            _build_async(
                atomic_facts_dict=atomic_facts_dict,
                ollama_base_url=ollama_base_url,
                embeddings_ollama_base_url=embeddings_ollama_base_url,
                llm_model=llm_model,
                embeddings_model=embeddings_model,
                temperature=temperature,
                ent_threshold=ent_threshold,
                rel_threshold=rel_threshold,
                max_workers=max_workers,
                api_key=api_key,
                embeddings_api_key=embeddings_api_key,
                llm_kwargs=llm_kwargs,
                embeddings_kwargs=embeddings_kwargs,
            )
        )
    except IndexError as e:
        # Catch IndexError that propagates through the event loop
        logger.warning(
            f"itext2kg IndexError (caught at sync level): {e}. "
            "No valid knowledge graph could be built from the atomic facts. "
            "Returning empty KnowledgeGraph."
        )
        from itext2kg.atom.models.knowledge_graph import KnowledgeGraph
        return KnowledgeGraph()
