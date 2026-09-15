## Reproducible environment with Pixi

This project uses [Pixi](https://pixi.sh/) as the authoritative environment manager.

The environment is configured for:

- `linux-64`
- `win-64`

Python and project dependencies are defined in `pixi.toml`, while `pixi.lock` records the exact resolved environment for reproducibility.

### Install the environment

From the repository root, run:

```bash
pixi install
```

This creates the local Pixi environment in `.pixi/`.

The `.pixi/` directory is generated locally and is not committed to Git.

### Verify the environment

Two lightweight smoke tests are included.

To verify that the main UrbanFeather dependencies and modules import correctly:

```bash
pixi run smoke-imports
```

Expected output:

```text
UrbanFeather imports: OK
```

To verify that the bundled FEATHER implementation can execute on a small test graph:

```bash
pixi run smoke-feather
```

Expected output:

```text
shape: (4, 4)
finite: True
```

The FEATHER smoke test uses:

- 4 nodes
- 1 input feature
- 2 characteristic-function evaluation points
- random-walk order 1

This gives an embedding dimension of:

```text
1 feature × 2 evaluation points × 2 (cosine/sine) × 1 order = 4
```

### Python version

The project currently targets Python 3.10.

This was chosen based on the existing notebook metadata in the repository, where Python 3.10 appears repeatedly across the OSMnx, Pandana, and city-based workflows.

### Dependency management

Do not install project dependencies manually with `pip` or create a separate `venv`.

Use Pixi instead.

For example, to add a new dependency:

```bash
pixi add PACKAGE_NAME
```

After changing dependencies, commit both:

```text
pixi.toml
pixi.lock
```

The lockfile should remain under version control because it is what makes the environment reproducible across machines.

### Running Python inside the environment

Python commands should normally be run through Pixi:

```bash
pixi run python your_script.py
```

For example:

```bash
pixi run python src/Utils/OSM2FeatherConverter.py
```

Some of the bundled FEATHER code retains the original script-style import structure and may require its `src` directory to be on the Python import path when imported as part of the larger UrbanFeather project. The included smoke tests handle this explicitly.

### Development principle

The environment setup follows the project principle:

> Reproduce first, improve second.

The current goal is to reproduce the existing UrbanFeather codebase reliably before modernizing package versions, restructuring modules, or changing the FEATHER implementation.

## Acknowledgments

This thesis project builds on the work of Magnus Rolin, Morten Bønneland, Teis Friberg, and Jonas Henriksen, who developed the original project *15-Minute City Accessibility Analysis using FEATHER*.

Their work provided the foundation for the UrbanFeather codebase used and extended in this thesis.

The project also builds on the original FEATHER method by Benedek Rozemberczki and Rik Sarkar.

For citation details of the predecessor project, see the included `CITATION.cff` file.
