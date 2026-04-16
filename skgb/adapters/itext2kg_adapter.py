from __future__ import annotations

import asyncio
import functools
import json
import logging
import shutil
import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..models import ModelRegistry, get_model_config, get_model_tier

logger = logging.getLogger(__name__)


def _run(coro):
    """Run async coroutine, handling Jupyter/Colab event loop conflicts."""
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called" not in str(exc):
            raise
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


def _patch_json_parser():
    try:
        from langchain_core.utils import json as lc_json
        from ..utils.json_repair import repair_json

        original_parse_json_markdown = lc_json.parse_json_markdown

        @functools.wraps(original_parse_json_markdown)
        def patched_parse_json_markdown(json_string: str, *, parser: Callable = json.loads):
            try:
                return original_parse_json_markdown(json_string, parser=parser)
            except Exception:
                pass
            logger.debug("Original JSON parsing failed, attempting repair...")
            result = repair_json(json_string)
            if result is not None:
                logger.debug("JSON repair successful")
                return result
            return original_parse_json_markdown(json_string, parser=parser)

        lc_json.parse_json_markdown = patched_parse_json_markdown
        logger.info("Applied JSON parser patch for better LLM output handling")
        return True

    except Exception as e:
        logger.warning(f"Could not patch JSON parser: {e}")
        return False


def _patch_itext2kg_provider_detection():
    try:
        try:
            import itext2kg.utils.llm as llm_utils
        except ImportError:
            llm_utils = None

        if llm_utils is not None:
            original_get_provider = getattr(llm_utils, 'get_provider', None)
            if original_get_provider is not None:
                @functools.wraps(original_get_provider)
                def patched_get_provider(llm_model):
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
                return ProviderType.CLAUDE
            if OllamaEmbeddings and isinstance(embeddings_model, OllamaEmbeddings):
                return ProviderType.CLAUDE

            return original_detect_provider(self)

        LangchainOutputParser._detect_provider = patched_detect_provider
        logger.info("Applied itext2kg provider detection patch for newer LangchainOutputParser layout")
        return True

    except Exception as e:
        logger.warning(f"Could not patch itext2kg provider detection: {e}")
        return False


def _patch_itext2kg_for_empty_results():
    try:
        import itext2kg.atom.atom as atom_module
        from itext2kg.atom.models.knowledge_graph import KnowledgeGraph

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
            return original_merge(self, valid_kgs, existing_kg, rel_threshold, ent_threshold, max_workers)

        atom_module.Atom.parallel_atomic_merge = safe_parallel_atomic_merge

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

        original_build_graph = atom_module.Atom.build_graph

        @functools.wraps(original_build_graph)
        async def safe_build_graph(self, atomic_facts, obs_timestamp, **kwargs):
            try:
                return await original_build_graph(self, atomic_facts, obs_timestamp, **kwargs)
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
    global _patches_applied
    if not _patches_applied:
        _patch_json_parser()
        _patch_itext2kg_provider_detection()
        _patch_itext2kg_for_empty_results()
        _patches_applied = True


def _save_checkpoint(kg, checkpoint_dir: Path, group_index: int) -> None:
    """Save a checkpoint of the current KG state to disk and Google Drive."""
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = "final" if group_index == -1 else f"group{group_index:03d}"
        checkpoint_name = f"checkpoint_{label}_{timestamp}"

        entities = []
        relationships = []

        if kg and not kg.is_empty():
            for ent in kg.entities:
                entities.append({
                    "name": getattr(ent, "name", str(ent)),
                    "label": getattr(ent, "label", ""),
                })
            for rel in kg.relationships:
                relationships.append({
                    "source": getattr(rel.startNode, "name", "") if hasattr(rel, "startNode") else "",
                    "target": getattr(rel.endNode, "name", "") if hasattr(rel, "endNode") else "",
                    "relation": getattr(rel, "name", ""),
                })

        checkpoint_data = {
            "group_index": group_index,
            "timestamp": timestamp,
            "entity_count": len(entities),
            "relation_count": len(relationships),
            "entities": entities,
            "relationships": relationships,
        }

        checkpoint_file = checkpoint_dir / f"{checkpoint_name}.json"
        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2, ensure_ascii=False))
        logger.info(
            f"✅ Checkpoint saved: {label} — "
            f"{len(entities)} entities, {len(relationships)} relations → {checkpoint_file.name}"
        )

        # Also zip and save to Google Drive if available
        try:
            drive_path = Path("/content/drive/MyDrive/skgb_checkpoints")
            drive_path.mkdir(parents=True, exist_ok=True)
            archive = shutil.make_archive(
                str(drive_path / checkpoint_name), "zip",
                str(checkpoint_dir.parent), str(checkpoint_dir.name)
            )
            logger.info(f"✅ Checkpoint also saved to Google Drive: {Path(archive).name}")
        except Exception:
            pass  # Drive not mounted — local save is enough

    except Exception as e:
        logger.warning(f"Checkpoint save failed at group {group_index}: {e}")


def _merge_kgs(kg_a, kg_b):
    """Merge two KnowledgeGraph objects by combining entities and relationships."""
    try:
        from itext2kg.atom.models.knowledge_graph import KnowledgeGraph

        if kg_a is None or kg_a.is_empty():
            return kg_b
        if kg_b is None or kg_b.is_empty():
            return kg_a

        merged = KnowledgeGraph()
        seen_entities = set()
        seen_relations = set()

        for ent in list(kg_a.entities) + list(kg_b.entities):
            key = getattr(ent, "name", str(ent)).lower().strip()
            if key not in seen_entities:
                seen_entities.add(key)
                merged.entities.append(ent)

        for rel in list(kg_a.relationships) + list(kg_b.relationships):
            src = getattr(rel.startNode, "name", "") if hasattr(rel, "startNode") else ""
            tgt = getattr(rel.endNode, "name", "") if hasattr(rel, "endNode") else ""
            name = getattr(rel, "name", "")
            key = f"{src}|{name}|{tgt}".lower().strip()
            if key not in seen_relations:
                seen_relations.add(key)
                merged.relationships.append(rel)

        return merged

    except Exception as e:
        logger.warning(f"KG merge failed: {e}. Returning kg_a.")
        return kg_a


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
    checkpoint_dir: Optional[Path] = None,
    facts_per_group: int = 25,
):
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

    model_config = get_model_config(llm_model)
    model_tier = get_model_tier(llm_model)
    logger.info(f"Using model tier '{model_tier}' configuration for {llm_model}")

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

    total_facts = sum(len(facts) for facts in atomic_facts_dict.values())
    logger.info(f"Building KG from {total_facts} atomic facts across {len(atomic_facts_dict)} timestamps")

    atom = Atom(llm_model=llm, embeddings_model=embeddings)
    merged_kg = KnowledgeGraph()

    for t_obs, facts in atomic_facts_dict.items():
        total_groups = (len(facts) + facts_per_group - 1) // facts_per_group
        logger.info(
            f"Processing {len(facts)} facts in {total_groups} groups of ~{facts_per_group} "
            f"(checkpoint every 5 groups)"
        )

        for group_idx in range(total_groups):
            group_facts = facts[group_idx * facts_per_group:(group_idx + 1) * facts_per_group]
            group_dict = {t_obs: group_facts}

            logger.info(
                f"🔄 Group {group_idx + 1}/{total_groups} "
                f"({len(group_facts)} facts)..."
            )

            max_attempts = model_config["max_retries"]
            group_kg = KnowledgeGraph()

            for attempt in range(1, max_attempts + 1):
                try:
                    group_kg = await atom.build_graph_from_different_obs_times(
                        atomic_facts_with_obs_timestamps=group_dict,
                        ent_threshold=ent_threshold,
                        rel_threshold=rel_threshold,
                        max_workers=max_workers,
                    )
                    if group_kg and not group_kg.is_empty():
                        logger.info(
                            f"   ✓ Group {group_idx + 1}: "
                            f"{len(group_kg.entities)} entities, "
                            f"{len(group_kg.relationships)} relations"
                        )
                    else:
                        logger.warning(f"   ⚠ Group {group_idx + 1} attempt {attempt}: empty KG")
                        if attempt < max_attempts:
                            await asyncio.sleep(model_config["retry_delay"])
                            continue
                    break

                except Exception as e:
                    logger.warning(f"   ✗ Group {group_idx + 1} attempt {attempt} failed: {e}")
                    if attempt < max_attempts:
                        await asyncio.sleep(model_config["retry_delay"])

            # Merge this group into running total
            merged_kg = _merge_kgs(merged_kg, group_kg)
            logger.info(
                f"   Running total: {len(merged_kg.entities)} entities, "
                f"{len(merged_kg.relationships)} relations"
            )

            # Checkpoint every 5 groups
            if checkpoint_dir and (group_idx + 1) % 5 == 0:
                logger.info(f"💾 Saving checkpoint after group {group_idx + 1}...")
                _save_checkpoint(merged_kg, checkpoint_dir, group_idx + 1)

    # Final checkpoint
    if checkpoint_dir:
        logger.info("💾 Saving final checkpoint...")
        _save_checkpoint(merged_kg, checkpoint_dir, -1)

    logger.info(
        f"✅ KG build complete: {len(merged_kg.entities)} entities, "
        f"{len(merged_kg.relationships)} relations"
    )
    return merged_kg


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
    checkpoint_dir: Optional[Path] = None,
    facts_per_group: int = 25,
):
    """Build a KnowledgeGraph using itext2kg ATOM (async under the hood).

    Processes atomic facts in groups of `facts_per_group` and saves a
    checkpoint to `checkpoint_dir` every 5 groups. This ensures partial
    results are preserved even if the runtime disconnects mid-run.

    Args:
        atomic_facts_dict: Dict mapping observation timestamps to lists of atomic facts.
        ollama_base_url: Base URL for the Ollama LLM server.
        embeddings_ollama_base_url: Optional base URL for Ollama embeddings.
        llm_model: LLM model name (e.g. ``"qwen2.5:32b"``).
        embeddings_model: Embeddings model name (e.g. ``"nomic-embed-text"``).
        temperature: LLM temperature (0.0 for deterministic).
        ent_threshold: Entity similarity threshold for deduplication.
        rel_threshold: Relation similarity threshold for deduplication.
        max_workers: Number of parallel workers.
        api_key: API key for cloud LLM providers.
        embeddings_api_key: API key for cloud embeddings providers.
        llm_kwargs: Optional provider-specific kwargs for the LLM factory.
        embeddings_kwargs: Optional provider-specific kwargs for embeddings factory.
        checkpoint_dir: Directory to save checkpoints. Defaults to
            skgb_output/checkpoints if None.
        facts_per_group: Number of atomic facts per processing group.
            Default 25 ≈ 5 itext2kg batches of 5 requests each.

    Returns:
        KnowledgeGraph object.
    """
    if checkpoint_dir is None:
        checkpoint_dir = Path("skgb_output/checkpoints")

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
                checkpoint_dir=checkpoint_dir,
                facts_per_group=facts_per_group,
            )
        )
    except IndexError as e:
        logger.warning(
            f"itext2kg IndexError (caught at sync level): {e}. "
            "Returning empty KnowledgeGraph."
        )
        from itext2kg.atom.models.knowledge_graph import KnowledgeGraph
        return KnowledgeGraph()