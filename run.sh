echo "Generating the configuration database. This may take a while..."
python exp/run_join_exp.py -s exp/join_exp_config.csv
echo "[Success] The configuration database is generated."

echo "Run microbenchmarks from Section 5.2.1 to 5.2.7"
python exp/run_join_exp.py \
        -b ./bin/volcano/join_exp_4b4b \
        -c exp/join_exp_config.csv \
        -y exp/join_runs.yaml \
        -e 0 1 2 3 4 \
        -r $1 \
        -p exp_results/gpu_join \
        -d $2

python exp/run_join_exp.py \
        -b ./bin/volcano/join_exp_8b8b \
        -c exp/join_exp_config.csv \
        -y exp/join_runs.yaml \
        -e 5 \
        -r $1 \
        -p exp_results/gpu_join \
        -d $2
    
python exp/run_join_exp.py \
        -b ./bin/volcano/join_exp_4b8b \
        -c exp/join_exp_config.csv \
        -y exp/join_runs.yaml \
        -e 6 \
        -r $1 \
        -p exp_results/gpu_join \
        -d $2

echo "Run the sequence of joins from Section 5.2.8"
for a in SMJ SMJI SHJ PHJ
do 
    for r in {1..$1}
    do 
        ./bin/volcano/join_pipeline $2 25 27 $a
    done
done