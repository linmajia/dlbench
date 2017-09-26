# Set up ENV
cwd=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )
logdir="$cwd/logs/dlbench_$(date +%Y-%m-%d-%H-%M-%S)"
config="$logdir/benchmark.config"
mkdir -p $logdir
src="$cwd/configs/benchmark.config"
gpu=$(python -m pip show tensorflow-gpu)
if [[ "$gpu" == "" ]]; then # CPU
    awk '!/^gpu;/' $src >$config
else # GPU 
    awk '!/^cpu;/' $src >$config
fi

# Run Python
logpath="$logdir/benchmark.log"
echo "excuting: PYTHONPATH=$cwd:$PYTHONPATH python -u $cwd/benchmark.py -config $config -result $logdir 2>&1 | tee $logpath"
PYTHONPATH=$cwd:$PYTHONPATH python -u $cwd/benchmark.py -config $config -result $logdir 2>&1 | tee $logpath
