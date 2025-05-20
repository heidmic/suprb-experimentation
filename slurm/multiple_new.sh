export filename="runs/comparisons/suprb_all_tuning.py"
echo $filename

datasets=("concrete_strength" "energy_cool" "combined_cycle_power_plant" "airfoil_self_noise")
fitness_funcs=("PseudoBIC" "ComplexityWu")

for fit in ${fitness_funcs[@]}; do
    for dset in ${datasets[@]}; do
        export experiment_name=${dset}
        export problem=${dset}
        export fitness_func=${fit}
        echo $experiment_name
        echo $problem
        echo $fitness_func
        sbatch slurm/default_licca.sbatch
        sleep 2
    done 
done

