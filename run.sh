# Set up ENV
cwd=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )
src="$cwd/configs/benchmark.config"
mkdir -p "$cwd/logs"
config="$cwd/logs/$(date +%Y%m%d%H%M).config"
gpu=$(python -m pip show tensorflow-gpu)
if [[ "$gpu" == "" ]]; then # CPU
    awk '!/^gpu;/' $src >$config
else # GPU 
    awk '!/^cpu;/' $src >$config
fi

# Run Python
echo "excuting: PYTHONPATH=$cwd:$PYTHONPATH python -u $cwd/benchmark.py -config $config"
PYTHONPATH=$cwd:$PYTHONPATH python -u $cwd/benchmark.py -config $config
if [ $? -eq 0 ]; then
    rm -f $config
fi
