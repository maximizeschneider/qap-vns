# Quadratic Assignment Problem (QAP) Solver

## Overview
* Implementation of a solver for the Quadratic Assignment Problem using Variable Neighborhood Search (VNS)
* Optimizes facility-to-location assignments by minimizing total transportation costs
* Includes multiple constructive heuristics and VNS variants

## Features
* Fast numpy-based objective function implementation
* Multiple constructive heuristics:
  - Random assignment
  - Ordered Greedy
  - Dynamic Ordered Greedy
  - Increasing Degree of Freedom
* VNS implementation with variants:
  - Base VNS
  - General VNS
  - Reduced VNS
  - Skewed VNS

## Project Structure
* `documentation.ipynb` - Documentation and analysis in German
* `Solver.py` - Main interface for users
* `InputData.py` - Handles JSON input parsing
* `Solution.py` - Represents and evaluates solutions
* `ConstructiveHeuristic.py` - Initial solution generators
* `VariableNeighborhoodSearch.py` - VNS implementation
* `Neighborhood.py` - Different neighborhood implementations
* `Logger.py` - Logging functionality
* `Timer.py` - Time tracking utilities

## Input Format
* JSON files containing:
  - n: number of facilities/locations
  - F: flow matrix between facilities
  - D: distance matrix between locations
  - M: minimum distance requirements (optional)