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


## Old README.md
<img src="feather_order_4.png" alt="This image shows a map with magnitude values calculated from the Feather ML-Algorithm. on the right side is the moving (transport) to the left is all Amenity nodes combined" >

This repository contains a research oriented project exploring **accessibility analysis based on the 15-minute city concept**, with a focus on comparing traditional accessibility approched with a FEATHER based approch.

## Problem Statement

The **15-minute city** is an urban planning concept introduced by Carlos Moreno in 2016. It describes a city where residents can access all essential daily services such as work, education, healthcare, shopping, and social activities within a **definitive walking distance** from their home.

The concept emphasises:

- Improved **accessibility**
- Reduced **dependence on cars**
- Increased **social equity**, regardless of mobility or socioeconomic status

In this project, accessibility is defined as the **ability to reach essential services and amenities within a given threshold**, which serves as a continuous value rather than a strict cutoff.

## Project Goal

The goal of this project is to explore and evaluate **new methods for urban accessibility analysis** by leveraging modern graph algorithms specifically the **FEATHER algorithm** and comparing them with established tools such as **GOAT (Geo Open Accessibility Tool)**.

The project aims to deliver a **FEATHER-based solution** capable of processing **OpenStreetMap (OSM)** data to produce accessibility analyses comparable to GOAT’s current capabilities.

## Research Objectives

1. **Comparative Analysis**
   Analyze FEATHER and GOAT for accessibility mapping, focusing on strengths, limitations, and practical differences through case studies or scenarios.

2. **Performance Evaluation**
   Evaluate FEATHER’s performance, robustness, and accuracy when calculating accessibility metrics within an urban environment.

3. **Methodological Extensions**
   Explore potential extensions to existing accessibility measurement methodologies that could be implemented using FEATHER.

## Methodology Overview

- Utilize **OpenStreetMap (OSM)** data as the primary spatial data source
- Apply the **FEATHER algorithm** for graph-based accessibility analysis
- Compare outputs and metrics against those produced by **Pandana**
- Document implementation details, assumptions, and limitations

## Expected Deliverables

- A FEATHER-based accessibility analysis
- Detailed documentation of the methodology and implementation
- Comparative results between FEATHER and PANDANA
- Discussion of limitations and future improvements

## How to run codebase Examples

This example demonstrates how to generate accessibility graph summaries for Daegu, South Korea using OpenStreetMap data.

```bash
#1. Clone the Repository
git clone https://github.com/MRollin03/FGSAM-FEATHER-Graph-summaries-of-accessibility-maps.git

#2. Navigate to the Utility Scripts
cd FGSAM-FEATHER-Graph-summaries-of-accessibility-maps/src/Utils

#3. Run the OSM to Feather Converter
# The following command downloads and processes map data for Daegu, South Korea and stores the generated project files in the projects directory.


```

If you are having trouble with the module utils not getting reconized/found write this command into the terminal
```bash
  # make current terminal's python session reconize your internal modules
  export PYTHONPATH=$PYTHONPATH:$(pwd)/FEATHER/src
```

>[!NOTE]
>Ensure you have Python and all required dependencies installed before running the script.
>Internet access is required to fetch OpenStreetMap data.
>Large locations may take additional processing time depending on system performance. and Overpass API status


## User Guide

### Automation.sh and or OSM2FeatherConverter.py

`OSM2FeatherConverter.py` handles the full OSM processing pipeline, including querying OpenStreetMap data through the Overpass API, formatting the data, computing FEATHER metrics, and generating plots. But a eaiser way of using it and get a overview would be using the Automation.sh scripts in the Utils directory. src/Utils/Automation.sh

 

Pipeline overview:

```
Input → Overpass API → Formatting → FEATHER Computing → Plotting
```
Run the whole pipine with the bash script
You can edit diffrent parameters easily in the script
```bash
bash Automation.sh
```

### Arguments

```bash
--title STR
    Name of the project.

--type STR
    Input mode:
    PLACE        #Use a single place name
    BBOX         #Use a bounding box
    MULTI_PLACE  #Use multiple place names

--bbox FLOAT FLOAT FLOAT FLOAT
    #Bounding box coordinates:
    #west south east north
    #Only used when --type BBOX

--place STR
    #Name of a single location/place.
    #Only used when --type PLACE

--places STR [STR ...]
    #Multiple locations/place names.
    #Only used when --type MULTI_PLACE

--solo STR
    #Process only a single feature/category.
    #If NONE, all categories are processed.

--distance INT
    #Maximum distance used when searching for POIs/Amenities.

--output STR
    #Output directory for the generated project folder.

--pandana BOOL
    #Enable or disable Pandana network calculations.

--order INT
    #FEATHER order value.

--plotorders STR
    #Comma-separated list of plot orders to generate.
    #Example:
    #"1,2,3,4"

--thetamax FLOAT
    #Maximum theta value used in FEATHER calculations.

--evalpoints INT
    #Number of evaluation points used during computation.
```

### Example Usage

#### PLACE mode

```bash
python OSM2FeatherConverter.py \
    --type PLACE \
    --place "Odense Municipality" \
    --title "example_project"
```

#### BBOX mode

```bash
python OSM2FeatherConverter.py \
    --type BBOX \
    --bbox 12.24358 55.91420 12.34091 55.95679 \
    --title "example_project"
```

#### MULTI_PLACE mode

```bash
python OSM2FeatherConverter.py \
    --type MULTI_PLACE \
    --places \
        "Copenhagen Municipality, Denmark" \
        "Frederiksberg Municipality, Denmark" \
    --title "example_project"
```


The options for FEATHER can be found here: https://github.com/benedekrozemberczki/FEATHER/blob/master/README.md

### Graphs
OSM to feather conversion pipeline
<img src="OSM Network Processing.png" alt="This is a diagram over the pipline process of the convertion" >

## Literature

- Moreno, C. (2021). _Definition of the 15-minute city: What is the 15-minute city?_
  [https://www.researchgate.net/publication/362839186_Definition_of_the_15-minute_city_WHAT_IS_THE_15_MINUTE_CITY](https://www.researchgate.net/publication/362839186_Definition_of_the_15-minute_city_WHAT_IS_THE_15_MINUTE_CITY)

- OpenStreetMap Contributors. _Planet OSM._
  [https://planet.osm.org/](https://planet.osm.org/) (Accessed: 26-01-2026)

- Rozenberczki, B., & Sarkar, R. (2020). _Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models._
  arXiv:2005.07959, [https://arxiv.org/pdf/2005.07959.pdf](https://arxiv.org/pdf/2005.07959.pdf) (Accessed: 26-01-2026)

## License

This project is intended for academic and research purposes. Licensing details will be added  once finalised.





