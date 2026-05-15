# sc_resilience_kg

**Supply Chain Resilience Knowledge Graph — PDF to Neo4j pipeline for electronics firms operating in India and Vietnam**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-PhD%20Research-orange.svg)](LICENSE)

This repository contains the full implementation of an ontology-guided knowledge graph construction pipeline developed as part of a Bachelor thesis at Constructor University Bremen. The pipeline extracts supply chain resilience entities from corporate PDF documents (annual reports, sustainability reports) and loads them into a Neo4j knowledge graph aligned to a custom OWL/RDF ontology.

## What this does

```
Corporate PDFs
  → IBM Docling (PDF → structured Markdown)
  → Semantic chunking
  → Claude (Anthropic API) structured ontology extraction
  → Entity deduplication (nomic-embed-text via Ollama)
  → Neo4j AuraDB knowledge graph
  → SHACL validation + Competency Question evaluation
```

## Results

- 2,507 chunks processed across 6 documents, 0 failures
- 347 suppliers, 818 products, 975 risk events, 214 facilities extracted
- 84.8% SHACL validation pass rate
- 10/20 competency questions answerable via Cypher
- Main finding: public annual reports reliably capture qualitative resilience signals (collaboration, digitalization, adaptability) but do not disclose quantitative metrics (TTR, TTS, lead times)

## Repository structure

```
sc_resilience_kg/
├── skgb/                    ← Python pipeline package
├── ontology/                ← OWL/RDF ontology + SHACL shapes
├── supply_chain_docs/       ← 6 corporate PDF documents used in the thesis
├── skgb_output/             ← Pipeline outputs (extraction JSON, SHACL report, CQ results)
├── data/
│   └── kg_import_clean.cypher
├── skgb_results.zip         ← Zipped pipeline outputs
├── docker-compose.yml       ← Optional: local Neo4j + Jupyter stack
└── requirements.txt
```

## Running the pipeline

The pipeline is distributed as a standalone notebook. No manual cloning or folder setup required.

**1. Download the notebook**

Download `supply_chain_resilience_pipeline.ipynb` from the [latest release](https://github.com/ElyesChouikha/DynamicKGConstruction/releases/latest) and place it in any folder you want to use as your workspace.

**2. Set up requirements**

- Python 3.10+
- [Ollama](https://ollama.com) installed and running — pull the embeddings model:
  ```bash
  ollama pull nomic-embed-text
  ```
- Anthropic API key — get one at [console.anthropic.com](https://console.anthropic.com)
- Neo4j AuraDB instance — free tier at [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura)

**3. Run cell 1.a**

The first cell automatically clones this repo next to the notebook and installs all dependencies. Your workspace will look like this after:

```
your_workspace/
├── supply_chain_resilience_pipeline.ipynb
└── sc_resilience_kg/
    ├── skgb/
    ├── ontology/
    ├── supply_chain_docs/    ← PDFs already included
    ├── skgb_output/
    └── requirements.txt
```

**4. Run all cells top to bottom**

Anthropic and Neo4j credentials are prompted securely — never hardcoded.

**Every time after that:**
- Activate your venv: `source venv/bin/activate`
- Skip cell 1.a
- Run from cell 1.b onwards

## Ontology

The supply chain resilience ontology defines 6 classes aligned to the resilience framework of Soni et al. (2014):

| Class | Key properties | Resilience factor |
|---|---|---|
| Supplier | tierLevel, riskIndex, collaborationLevel | Collaboration, Diversification |
| Facility | timeToRecover, timeToSurvive, digitalizationLevel | Flexibility, Visibility |
| Location | locationName, country | Diversification |
| Product | leadTime, onTimeDeliveryRate, isDualSourced | Flexibility, Diversification |
| RiskEvent | revenueImpact, timeToAwareness, affectsCountry | Visibility |
| Certification | certificationName, issuingBody | Collaboration |

OWL/RDF serialised in Turtle: `ontology/resilience_ontology.ttl`
SHACL validation shapes: `ontology/resilience_shapes.ttl`

## Document corpus

| Company | Document | Year |
|---|---|---|
| Apple | Supply Chain Progress Report | 2024 |
| Apple | Supply Chain Progress Report | 2026 |
| Dixon Technologies | Annual Report | 2024–25 |
| Hon Hai (Foxconn) | Annual Report | 2024 |
| Pegatron | Annual Report | 2024 |
| Samsung Electronics | Sustainability Report | 2025 |

Documents selected based on publicly available PDF format and explicit coverage of India or Vietnam operations in the China+1 diversification context.

## Domain

- Electronics sector: EMS firms, smartphone OEMs, semiconductor assemblers
- Geography: India and Vietnam operations under the China+1 strategy
- Time period: 2022–2026

## Cost estimate

- ~$0.007 per chunk via Claude Sonnet (~$18–20 for 2,500 chunks)
- Gold set evaluation and ablation study: ~$0.05 total

## Local Neo4j (optional)

If you prefer running Neo4j locally instead of AuraDB:

```bash
docker compose up -d
```

This starts Neo4j on `http://localhost:7474` and `neo4j://localhost:7687`. Load the graph using `data/kg_import_clean.cypher`.

## Acknowledgments

- [IBM Docling](https://github.com/docling-project/docling) — PDF to structured Markdown
- [itext2kg](https://github.com/AuvaLab/itext2kg) — ATOM-based graph construction framework this pipeline extends
- [Anthropic Claude](https://www.anthropic.com) — structured ontology-guided extraction
- [Neo4j](https://neo4j.com) — graph database

## License

This project is part of ongoing PhD research. See [LICENSE](LICENSE) for details.