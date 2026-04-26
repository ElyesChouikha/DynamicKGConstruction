# DynamicKGConstruction

**Document-to-knowledge-graph construction and Neo4j retrieval — extended with a Supply Chain Resilience pipeline**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-PhD%20Research-orange.svg)](LICENSE)

DynamicKGConstruction is a research codebase centered on `skgb/`.

It currently has three working flows:

- **Construction (itext2kg)**: documents → Docling → header-aware chunks → itext2kg ATOM → JSON/CSV/GraphML/HTML/Neo4j export
- **Retrieval**: imported Neo4j graph → `skgb.retrieval` → evidence search and grounded answers from notebooks
- **Supply Chain Resilience Pipeline**: PDFs → Docling → semantic chunks → Claude (Anthropic API) structured extraction → ontology-aligned Neo4j KG

```text
Documents
  -> Docling markdown conversion
  -> semantic chunking
  -> itext2kg ATOM graph construction (original flow)
     OR
  -> Claude structured ontology extraction (resilience flow)
  -> kg_output/ exports
  -> Neo4j import
  -> notebook retrieval with skgb.retrieval
```

## Current Status

- `python3 -m skgb run ...` works end to end and writes `build_docling/`, `chunks_output/`, and `kg_output/`
- Neo4j export is generated automatically as `kg_output/neo4j_load.cypher`
- `skgb.retrieval` works against an imported Neo4j graph
- Default retrieval mode is `entity_graph`, which works directly on the current entity graph export
- Optional `vector` retrieval is also supported when your Neo4j database already contains chunk embeddings and a vector index
- Supply chain resilience ontology and SHACL shapes are in `ontology/`

## Highlights

- Multi-format document ingestion by default: PDF, DOCX, PPTX, XLSX, HTML, Markdown, AsciiDoc, CSV, and common image formats
- Working semantic chunking with header paths, stable chunk IDs, optional metadata chunk, and configurable overlap
- Working knowledge graph construction with itext2kg ATOM and export to JSON, CSV, GraphML, HTML, and Neo4j Cypher
- Construction provider auto-detection from model names: Ollama, Anthropic, and OpenAI LLMs are supported; embeddings can use Ollama or OpenAI
- Working notebook retrieval API: `build_rag()`, `search_context()`, and `ask_graph()`
- Docker Compose stack for Neo4j + Jupyter retrieval work
- Domain-specific supply chain resilience ontology (OWL/RDF + SHACL) with 20 competency questions

## Repo Usage

- Real source lives in `skgb/`
- Run commands from the repository root with `python3 -m skgb ...`
- Import from the repository root with `import skgb`
- `notebooks/archive/` mostly contains older benchmark and experiment artifacts, not the main implementation

## Installation

### Construction

```bash
git clone https://github.com/ElyesChouikha/DynamicKGConstruction.git
cd DynamicKGConstruction
python3 -m pip install -r requirements.txt
pip install langchain-anthropic neo4j pyshacl rdflib
```

### Retrieval Extras

`skgb.retrieval` depends on Neo4j GraphRAG packages that are kept separate from the construction stack. If you use the provided Jupyter Docker image, those packages are installed there already.

```bash
python3 -m pip install -r skgb/retrieval/requirements.txt
```

## Construction Quick Start

### Local Ollama Example

If you want a fully local run, start Ollama and pull a chat model plus an embeddings model:

```bash
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

Then run the pipeline from the repo root:

```bash
python3 -m skgb run \
  --input "path/to/document-or-folder" \
  --out "runs/demo" \
  --recursive
```

### Claude + Ollama Embeddings Example (Supply Chain Resilience Pipeline)

```bash
export ANTHROPIC_API_KEY="your-key-here"
ollama pull nomic-embed-text
```

Then configure with Claude as LLM and nomic-embed-text for embeddings:

```python
from skgb import SKGBConfig

cfg = SKGBConfig.from_out_dir(
    "skgb_output",
    llm_model="claude-sonnet-4-6",
    embeddings_model="nomic-embed-text",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    ollama_base_url="http://localhost:11434",
    temperature=0.0,
)
```

See `supply_chain_resilience_pipeline.ipynb` for the full end-to-end notebook.

### Python API

```python
from pathlib import Path
from skgb import SKGBConfig, run_pipeline

cfg = SKGBConfig.from_out_dir(
    "runs/demo",
    llm_model="qwen2.5:14b",
    embeddings_model="nomic-embed-text",
    ollama_base_url="http://localhost:11434",
)

result = run_pipeline(Path("path/to/document.pdf"), cfg)

print(result.build_docling_dir)
print(result.chunks_json_path)
print(result.kg_output_dir)
print(result.neo4j_cypher_path)
```

## Construction Providers

Construction provider selection is inferred from model names.

| Example model | Detected provider | Usage |
|---|---|---|
| `qwen2.5:14b` | Ollama | local or Ollama-compatible endpoint |
| `claude-sonnet-4-6` | Anthropic | LLM only |
| `gpt-4o` | OpenAI | LLM |
| `text-embedding-3-small` | OpenAI | embeddings |
| `nomic-embed-text` | Ollama | embeddings |

Important:

- Anthropic embeddings are not supported; pair Claude with Ollama or OpenAI embeddings
- If you run from the repo root, use `import skgb`, not `import DynamicKGConstruction.skgb`
- The construction pipeline enables a focused set of Docling formats by default; audio/XML helpers exist in the adapter but are not enabled by default

## Construction Output Layout

Each run writes this layout under `--out`:

```text
runs/demo/
  build_docling/
  chunks_output/
    all_chunks.json
  kg_output/
    knowledge_graph.json
    kg_nodes.csv
    kg_edges.csv
    knowledge_graph.graphml
    kg_visualization.html
    construction_report.txt
    neo4j_load.cypher
```

## Supply Chain Resilience Extension

This fork extends the base pipeline with an ontology-guided extraction layer for supply chain resilience knowledge graph construction, developed as part of a Bachelor thesis at Constructor University Bremen.

### What was added

- `ontology/resilience_ontology.ttl` — OWL/RDF ontology covering 6 classes: Supplier, Facility, Location, Product, RiskEvent, Certification
- `ontology/resilience_shapes.ttl` — SHACL validation shapes enforcing ontology constraints
- Claude structured output extraction mapped directly to ontology classes using Pydantic schemas
- 20 competency questions evaluable via Cypher against the resulting Neo4j graph
- Evaluation framework: SHACL validation pass rate, P/R/F1 on annotated gold set, CQ answerability

### Domain

- Electronics sector (EMS firms, smartphone OEMs, semiconductor assemblers)
- Geography: India and Vietnam operations in the context of China+1 diversification
- Document corpus: annual reports and sustainability reports (2022–2025)
- Companies: Hon Hai/Foxconn, Samsung, Apple supply chain, Dixon Technologies, Pegatron

### Key results

- 2507 chunks processed across 6 documents, 0 failures
- 347 suppliers, 818 products, 975 risk events, 214 facilities extracted
- 84.8% SHACL pass rate
- 10/20 competency questions answerable via Cypher (50%)
- Main finding: public annual reports capture qualitative resilience signals (collaboration, digitalization, adaptability) but rarely disclose quantitative metrics (TTR, TTS, lead times)

## Neo4j Export And Import

The construction pipeline exports `kg_nodes.csv`, `kg_edges.csv`, and `neo4j_load.cypher`.

To regenerate the Cypher loader for an existing run:

```bash
python3 -m skgb export-neo4j --kg-output "runs/demo/kg_output"
```

To load into the provided Docker Neo4j setup:

1. Start Neo4j with `docker compose up -d`
2. Copy `kg_nodes.csv` and `kg_edges.csv` into `./neo4j/import/`
3. Run the generated `neo4j_load.cypher` in Neo4j Browser or `cypher-shell`

The generated importer uses a generic `:REL` relationship type and stores the actual predicate in `r.relation`.

## Retrieval Quick Start

Retrieval is notebook-first. There is no retrieval CLI yet.

### 1. Configure Neo4j And Retrieval

Start from the example file:

```bash
cp .env.example .env
```

For the current exported entity graph, the recommended setup is `entity_graph` retrieval:

```env
NEO4J_AUTH=none
NEO4J_URI=neo4j://localhost:7687
NEO4J_DATABASE=neo4j
RETRIEVAL_STRATEGY=entity_graph
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:14b
RETRIEVER_TOP_K=5
LLM_TEMPERATURE=0.0
```

Use `vector` retrieval only when your Neo4j database already has chunk embeddings and a vector index:

```env
RETRIEVAL_STRATEGY=vector
NEO4J_VECTOR_INDEX=chunkEmbeddings
OLLAMA_EMBEDDINGS_MODEL=nomic-embed-text
```

### 2. Start Neo4j And Jupyter

```bash
docker compose up -d
```

This starts:

- `neo4j` on `http://localhost:7474` and `neo4j://localhost:7687`
- `jupyter` on `http://localhost:8888`

### 3. Import The Graph Into Neo4j

Run the construction pipeline first, then copy the exported CSVs into `./neo4j/import/`, and execute the generated `neo4j_load.cypher`.

### 4. Query From Python Or Notebooks

```python
from skgb.retrieval import ask_graph, build_rag, search_context

with build_rag(env_file=".env", validate=True) as runtime:
    context = search_context(
        "What are the main findings?",
        runtime=runtime,
        top_k=5,
    )
    answer = ask_graph(
        "What are the main findings?",
        runtime=runtime,
        top_k=5,
        return_context=True,
    )

print(answer.answer)
print(context.metadata)
```

Notebook demo:

- `notebooks/skgb_retrieval_evidence_demo.ipynb`

## Retrieval Modes

### `entity_graph`

- Default mode
- Recommended for the current exported Neo4j entity graph
- Does not require chunk nodes, embeddings, or a Neo4j vector index
- Ranks graph nodes using lexical matching plus graph salience, then builds evidence from connected facts

### `vector`

- Uses Neo4j GraphRAG `VectorCypherRetriever` plus `GraphRAG`
- Requires a Neo4j vector index and chunk embeddings already stored in the database
- Best when your Neo4j graph includes chunk-level retrieval structures beyond the default entity export

## Main Modules

```text
skgb/
  cli.py
  config.py
  pipeline.py
  adapters/
    docling_adapter.py
    chunking_adapter.py
    itext2kg_adapter.py
  export/
    file_export.py
    neo4j_export.py
  retrieval/
    config.py
    factory.py
    query.py
    entity_graph.py
ontology/
  resilience_ontology.ttl
  resilience_shapes.ttl
```

## Known Constraints

- `python3 -m skgb --help` still imports package code, so missing construction dependencies such as `docling` will fail before help output
- `construction_report.txt` currently labels LLM and embeddings as `(Ollama)` even when a run used a different provider; trust config or logs for the real provider
- Chunk overlap is applied after chunks from all documents are flattened, so overlap can cross document boundaries in multi-document runs
- The default Neo4j export is entity-centric; vector retrieval needs extra graph/index preparation beyond the default export
- Anthropic embeddings are not supported; always pair Claude with Ollama or OpenAI embeddings

## Acknowledgments

- [Docling](https://github.com/docling-project/docling) for robust multi-format document parsing
- [itext2kg](https://github.com/AuvaLab/itext2kg) for the ATOM-based graph construction workflow
- [Neo4j GraphRAG Python](https://github.com/neo4j/neo4j-graphrag-python) for the retrieval building blocks used by `skgb.retrieval`
- [Anthropic Claude](https://www.anthropic.com) for structured ontology-guided extraction

## License

This project is part of ongoing PhD research. See [LICENSE](LICENSE) for details.
