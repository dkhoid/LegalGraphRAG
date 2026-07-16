# Plan: Transition to Vietnamese Civil Law Context

## Overview
The goal is to transition the LegalGraphRAG repository to focus exclusively on Vietnamese Civil Law. The original Chinese dataset (CAIL) and related processing scripts will be backed up and removed from the active project structure. The new Vietnamese API, web interface, and modified core logic will become the main project baseline.

## Project Type
**BACKEND** (with a Web UI frontend component)

## Success Criteria
- All Chinese raw data and generated data are safely backed up in an archive folder.
- The `raw_data/` and `datas/` directories contain only Vietnamese civil law data.
- The repository documentation (`README.md`) accurately reflects the new Vietnamese context.
- The Vietnamese API server (`api_server.py`) and Graph visualization scripts function correctly.

## Tech Stack
- Python (FastAPI for API server)
- NetworkX / Pyvis (Graph Visualization)
- LegalGraphRAG Core Framework

## File Structure
```
LegalGraphRAG/
├── api_server.py          # Vietnamese Civil Law API
├── web/                   # Frontend UI
├── archive_cn_data/       # (NEW) Backup of original Chinese CAIL datasets
├── raw_data/              # Active Vietnamese legal documents
├── datas/                 # Processed Vietnamese graph features
├── scripts/
│   ├── fetch_vn_legal_data.py
│   ├── generate_sample_cases.py
│   └── visualize_graph.py
└── core/                  # RAG evaluation logic (modified for VN context)
```

## Task Breakdown

### Task 1: Backup Chinese Data
- **Agent**: `orchestrator`
- **Skills**: `bash-linux`
- **Priority**: P0
- **INPUT**: `raw_data_backup_cn/` and existing CAIL scripts.
- **OUTPUT**: Ensure all Chinese-specific data files (e.g., `final_test.json`, `judicial_explanations.json`) are safely archived in `archive_cn_data/` and removed from active `raw_data/` and `datas/` folders.
- **VERIFY**: `ls raw_data/` shows no Chinese dataset files.

### Task 2: Commit Vietnamese Baseline
- **Agent**: `orchestrator`
- **Skills**: `bash-linux`
- **Priority**: P1
- **INPUT**: Staged git changes (`api_server.py`, `web/`, modified `core/` files).
- **OUTPUT**: A git commit solidifying the Vietnamese Civil Law modifications as the new main baseline.
- **VERIFY**: `git status` is clean.

### Task 3: Remove CAIL-Specific Scripts
- **Agent**: `backend-specialist`
- **Skills**: `clean-code`
- **Priority**: P1
- **INPUT**: Scripts in the `scripts/` directory.
- **OUTPUT**: Deletion of scripts that are strictly for CAIL (e.g., `prepare_cail_data.py`).
- **VERIFY**: The scripts folder only contains VN-related or generic processing scripts.

### Task 4: Update Documentation
- **Agent**: `documentation-writer`
- **Skills**: `documentation-templates`
- **Priority**: P2
- **INPUT**: `README.md`.
- **OUTPUT**: A rewritten README explaining how to run the Vietnamese Civil Law API, fetch VN data, and use the web interface.
- **VERIFY**: `README.md` no longer references Chinese CAIL challenges as the primary usage.

## ✅ Phase X: Verification
- [ ] Build & Data Pipeline: Re-run `python scripts/fetch_vn_legal_data.py` and `scripts/generate_sample_cases.py` to ensure active folders are correctly populated.
- [ ] API Check: Run `api_server.py` and verify it serves the `/web/index.html` UI successfully.
- [ ] Graph Check: Run `python scripts/visualize_graph.py` and ensure `graph_view.html` is generated successfully.
- [ ] Security: Pass `security_scan.py` to ensure no API keys or secrets were exposed in the new scripts.
