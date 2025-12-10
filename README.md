# Fire‑Related Tree Cover Loss and the Hansen–FRA Discrepancy in the Amazon Basin and Southern Cone (2015–2023): Replication Package

## Overview (English)

This directory contains the self‑contained replication package for the manuscript on fire‑related tree‑cover loss and the discrepancy between Hansen Global Forest Change and FAO FRA forest area statistics for twelve South American countries (2015–2023). It provides the Python scripts and processed inputs needed to rebuild the country–year panel dataset and regenerate all figures and tables used in the article.

The code is designed for scientific transparency and reproducibility in collaboration with partner institutions; it is **not** released under an open‑source license. All rights remain with the university (see the repository‑level README for institutional details).

### Structure

- `src/` – Python and shell scripts for data processing, analysis, and plotting.
- `Data/` – Minimal processed inputs (regional boundaries, filtered WDPA, FRA interpolation outputs, tile lists, VIIRS download notes).
- `results/` – Final panel datasets, figures, tables, and statistical summaries.
- `requirements.txt` – Python dependencies for reproducing the analysis.
- `MANIFEST.md` – Detailed list of files and their intended location within this package.

### Setup and execution

1. Create and activate a Python environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure that Hansen GFC v1.12 tiles, WDPA shapefiles, FRA 2025 CSVs, and VIIRS active fire CSVs are available following the guidance in `Data/viirs_download_instructions.md` and the tile lists at the package root.

3. From the replication package root, run the main V3 pipeline:

   ```bash
   cd src
   ./run_pipeline_v3.sh
   ```

   For the dual VIIRS metric variant, use:

   ```bash
   ./run_pipeline_v3_viirsdual.sh
   ```

4. The main outputs (country–year panel, figures, and statistical summaries) are written to `results/`.

## Resumen (Español)

Este directorio contiene el paquete de replicación del artículo sobre pérdida de cobertura arbórea asociada al fuego y la discrepancia entre Hansen Global Forest Change y las estadísticas de superficie forestal de la FAO (FRA) para doce países de Sudamérica entre 2015 y 2023. Incluye los scripts de Python y los insumos procesados necesarios para reconstruir el panel país‑año y regenerar las figuras y tablas utilizadas en el manuscrito.

El código se publica con fines de transparencia científica y colaboración académica, pero **no** se distribuye bajo una licencia de software libre. Todos los derechos patrimoniales pertenecen a la Corporación Universidad de la Costa CUC. Cualquier uso distinto a la revisión académica y la replicación de los resultados —incluida la modificación, redistribución o integración en otros sistemas— requiere autorización escrita previa de la Universidad.

## Resumo (Português)

Este diretório contém o pacote de replicação do artigo sobre perda de cobertura arbórea relacionada ao fogo e sobre a discrepância entre o produto Hansen Global Forest Change e as estatísticas de área florestal da FAO (FRA) para doze países da América do Sul no período de 2015 a 2023. O pacote disponibiliza scripts em Python e insumos processados que permitem reconstruir o painel país‑ano e reproduzir as figuras e tabelas do manuscrito.

O código é disponibilizado para fins de transparência científica e colaboração acadêmica, mas **não** é licenciado como software de código aberto. Todos os direitos patrimoniais pertencem à Corporación Universidad de la Costa CUC. Qualquer uso além da revisão acadêmica e da replicação dos resultados —incluindo modificação, redistribuição ou incorporação em outros softwares— exige autorização prévia e por escrito da Universidade.
