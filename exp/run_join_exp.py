import subprocess
import numpy as np
import itertools
import os
import time
import duckdb
import pandas as pd
import io
import argparse

def create_join_command(binary, nr, ns, pr, ps, algo, jtype = 'pkfk', dist='uniform', z = 0, sel = 1, p = 9, q = 6, uk = 1):
    command = f"{binary} -l -r {nr} -s {ns} -m {pr} -n {ps} -t {jtype} -d {dist} -z {z} -e {sel} -i {algo} -p {p} -q {q} -u {uk}"
    return command

def create_config_database(config_path = None, save_path = None):
    if config_path is None:
        nr = range(20, 29)
        ratio = range(0,8)
        pr = range(1, 7)
        ps = range(1, 7)
        dist = ['uniform', 'zipf']
        join_type = ['pkfk']
        unique_keys = range(21, 30)
        zipf_factor = np.linspace(0, 2, 9)
        selectivity = [1, 2, 4, 8, 16]
        algo = ['smj','phj','shj','smji']
        p = [9]
        q = [6,7]
        all_config = itertools.product(nr, ratio, pr, ps, dist, join_type, unique_keys, zipf_factor, selectivity, algo, p, q)
        all_config = itertools.filterfalse(lambda x: x[4] == 'uniform' and x[7] != 0, all_config)
        all_config = itertools.filterfalse(lambda x: x[4] == 'zipf' and x[8] != 1, all_config) # don't mix selectivity and distribution
        all_config = itertools.filterfalse(lambda x: x[5] == 'pfkf' and x[6] != x[0], all_config)
        all_config = itertools.filterfalse(lambda x: (x[9] == 'smj' or x[9] == 'smji') and (x[10] != np.min(p) or x[11] != np.min(q)), all_config)
        all_config = itertools.filterfalse(lambda x: x[0]+x[1] >= 31, all_config)
        df = pd.DataFrame(all_config, columns=['nr', 'ratio', 'pr', 'ps', 'dist', 'join_type', 'unique_keys', 'zipf_factor', 'selectivity', 'algo', 'p', 'q'])
        if save_path is not None:
            df.to_csv(save_path, index=False)
        return df
    else:
        return pd.read_csv(config_path)
    
def run_join_exp_from_query(binary, df, query, log_path, res_path, out_path, data_path, repeat = 5):
    config = duckdb.query(query).to_df()
    config.reset_index(inplace=True)
    total = config.shape[0]
    f = open(log_path, 'a')
    for r in range(repeat):
        cnt = 0
        fail = 0
        for t in config.itertuples(index=False):
            print(f"\n+=========+ [Round {r+1} of {repeat}] {cnt+1} / {total} +=========+")
            command = create_join_command(binary, t.nr, t.nr+t.ratio, t.pr, t.ps, t.algo, t.join_type, t.dist, t.zipf_factor, t.selectivity, t.p, t.q, t.unique_keys)
            command += ((' -f ' + data_path) if data_path is not None else '') + ' -o ' + res_path
            f.write(f"[{cnt}] {command}\n")
            cnt += 1
            try:
                with open(out_path, 'a') as out_file:
                    subprocess.run(command, shell=True, check=True, stdout=out_file)
                    # print(command)
            except subprocess.CalledProcessError as e:
                f.write(f"[[[fails]]]\n{repr(e)}\n")
                fail += 1
        f.write(f"[Round {r+1} of {repeat}] {cnt} experiments. {fail} fails.\n")
    f.close()

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--binary', type=str, required=False, help='binary path')
    parser.add_argument('-l', '--log', type=str, required=False, default='log.txt', help='log file path')
    parser.add_argument('-r', '--repeat', type=int, required=False, default=5, help='repeat times')
    parser.add_argument('-o', '--output', type=str, required=False, default='/dev/null', help='stdout file path')
    parser.add_argument('-c', '--config', type=str, required=False, help='available configuration path')
    parser.add_argument('-s', '--save_path', type=str, required=False, help='configurations save path')
    parser.add_argument('-e', '--exp', nargs='+', required=False, default=[], help='list of experiments to run')
    parser.add_argument('-y', '--yaml', type=str, required=False, help='yaml file that configures the experiment settings')
    parser.add_argument('-p', '--prefix', type=str, required=False, default=".", help='prefix file path of the output file')
    parser.add_argument('-d', '--data', type=str, required=False, default=None, help='For the first time, the program will store generated data in this path and use it for later experiments. If the data is already generated, the program will directly use the data in this path.')

    return parser

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    df = create_config_database(args.config, args.save_path)

    if args.yaml is not None:
        assert args.binary is not None
        import yaml
        with open(args.yaml, 'r') as f:
            root = yaml.load(f, Loader=yaml.Loader)
            for i in args.exp:
                i = int(i)
                exp = root['experiments'][i]
                print("+++++ Running experiment: ", exp['name'], f" for {args.repeat} times +++++")
                res_path = os.path.join(args.prefix, exp['output'])
                subprocess.run(f"mkdir -p {os.path.dirname(res_path)}", shell=True, check=True)
                run_join_exp_from_query(args.binary, df, exp['query'], args.log, res_path, args.output, args.data, args.repeat)
