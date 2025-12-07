# Dynamic Segment Graph (DSG)

This repository hosts a rebuilt implementation of Dynamic Segment Graph for range-filtered approximate nearest neighbor search. The previous repo was removed to clear redundant history; this one stays lean while we iterate. If anything is unclear, feel free to ask questions or suggest improvements.

## Status & Roadmap
- Static build + load: done (DFS compression, CSR storage).
- Insertion of new points: in progress.
- Densification for small query ranges: planned / in progress.
- Reducing the building time.

## Quick Start
```bash
mkdir build && cd build
cmake ..
make -j
```

Entry points for indexing/querying live in `apps/` (`build_index.cc`, `query_index.cc`). Helper scripts in `scripts/` show typical arguments; adjust flags or code constants as needed.

## Notes
- Targets C++17; uses STL and SIMD where helpful.
- Datasets are not bundled—point the CLI to your own data files.
- Expect rapid changes while insertion and densification land.
