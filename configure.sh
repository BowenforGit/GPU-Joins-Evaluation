
echo "Installing the required packages"
python -m pip install -r requirements.txt

echo "Compiling the code base"
make clean
make setup

sed "/#define [A-Z]\+_T_8B/d" src/volcano/join_exp.cu > src/volcano/join_exp_4b4b.cu
sed "s/using namespace std;/using namespace std;\n#define COL_T_8B/" src/volcano/join_exp_4b4b.cu > src/volcano/join_exp_4b8b.cu
sed "s/using namespace std;/using namespace std;\n#define KEY_T_8B\n#define COL_T_8B/" src/volcano/join_exp_4b4b.cu > src/volcano/join_exp_8b8b.cu

for target in join_exp_4b4b join_exp_4b8b join_exp_8b8b join_pipeline tpch_q7 tpch_q18 tpch_q19 tpcds_q64 tpcds_q95;
do
    echo "Compiling $target"
    make bin/volcano/$target
done