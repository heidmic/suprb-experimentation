#!/usr/bin/env bash

export archive_pop=true

export experiment="runs/rule_discovery/novelty_search.py"
export dataset="german.numer"
export ns_type="MCNS"
sbatch slurm/rule_discovery.sbatch

export ns_type="NSLC"
sbatch slurm/rule_discovery.sbatch

export dataset="chscase_foot"
export ns_type="MCNS"
sbatch slurm/rule_discovery.sbatch

export ns_type="NSLC"
sbatch slurm/rule_discovery.sbatch

export dataset="energy_cool"
sbatch slurm/rule_discovery.sbatch




# export experiment="runs/rule_discovery/evolution_strategy.py"
# export dataset="meta"
# sbatch slurm/rule_discovery.sbatch


