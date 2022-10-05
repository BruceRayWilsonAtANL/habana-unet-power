chsh -s
chsh -s bash
echo $SHELL
vim ~/.profile
exit
source aevard_venv/bin/activate
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache False --cache-path ./kaggle_128_cache --weights-file ./128.pt
export PYTHON=/home/aevard/aevard_venv/bin/python
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache False --cache-path ./kaggle_128_cache --weights-file ./128.pt
vim unet.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache False --cache-path ./kaggle_128_cache --weights-file ./128.pt
vim unet.py
vim dataset.py
vim unet.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache False --cache-path ./kaggle_128_cache --weights-file ./128.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache True --cache-path ./kaggle_128_cache --weights-file ./128.pt
vim dataset.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache True --cache-path ./kaggle_128_cache --weights-file ./128.pt
vim dataset.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache True --cache-path ./kaggle_128_cache --weights-file ./128.pt
vim dataset.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache True --cache-path ./kaggle_128_cache --weights-file ./128.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache False --cache-path ./kaggle_128_cache --weights-file ./128.pt
vim dataset.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache False --cache-path ./kaggle_128_cache --weights-file ./128.pt
vim dataset.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache False --cache-path ./kaggle_128_cache --weights-file ./128.pt
vim dataset.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache False --cache-path ./kaggle_128_cache --weights-file ./128.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache False --cache-path ./kaggle_128_cache --weights-file ./128.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache --cache-path ./kaggle_128_cache --weights-file ./128.pt
vim unet.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache --cache-path ./kaggle_128_cache --weights-file ./128.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache False --cache-path ./kaggle_128_cache --weights-file ./128.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache True --cache-path ./kaggle_128_cache --weights-file ./128.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache --cache-path ./kaggle_128_cache --weights-file ./128.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --cache-path ./kaggle_128_cache --weights-file ./128.pt
ls
source aevard_venv/bin/activate
cd apps/unet_bench/
ls
echo $PYTHON
export PYTHON=/home/aevard/aevard_venv/bin/python
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256
ls
vim dataset.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --cache-path ./kaggle_256_cache --weights-file ./256.pt
source aevard_venv/bin/activate
export PYTHON=`which python`
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --cache-path ./kaggle_256_cache --weights-file ./256.pt
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --cache-path ./kaggle_256_cache --weights-file ./256.pt
ls
source aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --cache-path ./kaggle_256_cache --weights-file ./256.pt
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --cache-path ./kaggle_256_cache --weights-file ./256.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --cache-path ./kaggle_256_cache --weights-file ./256.pt --run-name size-256-3
vim unet.py
ls
pwd
vim unet.py
ls
mkdir performance
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --cache-path ./kaggle_256_cache --weights-file ./256.pt --run-name size-256-3
vim unet.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --cache-path ./kaggle_256_cache --weights-file ./256.pt --run-name size-256-3
ls
rm performance/*
ls
ls ~/.habana
ls
ls ~/.habana
cd ~/.habana
ls fuser/
ls post_graph/
source aevard_venv/bin/activate
export PYTHON=`which python`
echo $PYTHON
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im---cache-path ./kaggle_128_cache --weights-file ./128.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --cache-path ./kaggle_128_cache --weights-file ./128.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache --cache-path ./kaggle_128_cache --weights-file ./128.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache --cache-path ./kaggle_128_cache --weights-file ./128.pt load-size-128
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache --cache-path ./kaggle_128_cache --weights-file ./128.pt --run-name load-size-128
vim performance/load-size-128
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache --cache-path ./kaggle_128_cache --weights-file ./128.pt --run-name load-size-128
vim performance/load-size-128
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache --cache-path ./kaggle_128_cache --weights-file ./128.pt --run-name load-size-128
vim performance/load-size-128
vim unet.py
vim performance/load-size-128
rm performance/*
ls
ls performance/
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache --cache-path ./kaggle_128_cache --weights-file ./128.pt --run-name load-size-128
vim performance/load-size-128.txt
cat performance/load-size-128.txt pbcopy
cat performance/load-size-128.txt | pbcopy
xclip
pbcopy
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 32 --im-cache --cache-path ./kaggle_cache --run-name load-size-32
top
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache --cache-path ./kaggle_128_cache --weights-file ./128.pt --run-name load-size-128
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 256 --im-cache --cache-path ./kaggle_256_cache --weights-file ./128.pt --run-name load-size-256-2
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 32 --im-cache --cache-path ./kaggle_cache --run-name load-size-32
top
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-1 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256.pt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 32  --cache-path kaggle_cache --run-name load-size-32
top
hl-smi -Q utilization.aip
hl-smi
hl-smi -h
echo $HABANA_LOGS
ls /var/log/habana_logs/
echo $HABANA_PROFILE
export HABANA_PROFILE=1
hl-smi
vim /var/log/habana_logs/hl-smi_log.
vim /var/log/habana_logs/hl-smi_log.txt
export HABANA_LOGS=~/habana_logs
hl-smi
ls
ls /tmp
ls /tmp | grep evard
top
source aevard_venv/bin/activate
ls
export PYTHON=/home/aevard/aevard_venv/bin/python
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 32 --im-cache --cache-path ./kaggle_cache --weights-file ./32.pt --run-name load-size-32
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 32 --im-cache --cache-path ./kaggle_cache --weights-file ./32.pt --run-name load-size-32
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 32 --cache-path ./kaggle_cache --weights-file ./32.pt --run-name load-size-32-cache
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --im-cache --cache-path ./kaggle_128_cache --weights-file ./128c.pt --run-name load-size-128-cache
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 128 --cache-path ./kaggle_128_cache --weights-file ./128c.pt --run-name load-size-128-cache
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 32 --cache-path ./kaggle_cache --weights-file ./32.pt --run-name load-size-32-cache
vim unet.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 32 --cache-path ./kaggle_cache --weights-file ./32.pt --run-name load-size-32-cache
mkdir weights
ls
mkdir data
;s
ls
mv kaggle_128_cache data
mv kaggle_cache data
mv kaggle_3m data
mv kaggle_256_cache data
ls data
ls
mv *.pt weights
ls weights
ls
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32.pt --run-name size-32-check
ls
ls data
vim dataset.py
vim unet.py
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32.pt --run-name size-32-check
ls
vim performance/size-32-check.txt
$PYTHON unet.py --hpu --use_lazy_mode --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32.pt --run-name size-32-check
vim performance/size-32-check.txt
git status
ls
hl-prof-config
hl-prof-config -h
cd
ls -a
ls ~/.habana
ls ~/.habana/post_graph/
ls ~/.habana/fuser
hl-prof-config -v
hl-prof-config --list-templates
ls ~/.habana
hl-smi -L
hl-smi -q
hl-smi -Q timestamp bus_id
hl-smi -Q timestamp,bus_id,memory.used
hl-smi -Q timestamp,bus_id,memory.used -f csv
hl-smi -Q timestamp,bus_id,memory.used,power.draw -f csv
hl-smi -Q timestamp,bus_id,memory.free,memory.used,power.draw -f csv -l
hl-smi -Q timestamp,bus_id,memory.free,memory.used,power.draw,utilization.aip -f csv -l
hl-prof-config -s test32 -o ~/.habana
ls ~/.habana
vim ~/.habana/prof_config.json
ls ~/.habana/fuser/
ls ~/.habana/post_graph/
ls ~/.habana/post_graph/797891/
ls ~/.habana/post_graph/73
ls ~/.habana/post_graph/738364/
ls ~/.habana/post_graph/789413/
ls
ls ~/.habana/post_graph/789413/
ls ~/.habana
ls
cd habana
cd habana_logs/
ls
vim synapse_log.txt
source aevard_venv/bin/activate
ls
export PYTHON=/home/aevard/aevard_venv/bin/python
export PYTHONPATH=/home/aevard/Model-References:/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
ls
cd habana
cd habana_logs/
ls
vim hl-smi_log.txt
hl-smi
echo $HABANA_PROFILE
export HABANA_PROFILE=1
hl-prof-configs --list-templates
hl-prof-config --list-templates
hl-prof-config -c
hl-prof-config -p
hl-prof-config -p enq
cd
cd apps/unet_bench/
ls
ls ~/.habana
ls
vim habana_log.livealloc.log_0
ls
vim default_profiling_hl0_011.json
vim default_profiling_hl0_052.json
rm *.json
ls
rm default_profiling_808958.hltv
rm default_profiling_810565.hltv
rm default_profiling_815711.hltv
ls
vim default_profiling_1117070.csv
ls
vim default_profiling_1118544.csv
ls
llss
ls
rm *.hltv
rm *.csv
ls
vim default_profiling_hl0_1.json
ls
rm *.json
ls
rm *.hltv
ls
rm *.jsonr
rm *.json
ls
rm *.json
ls
rm *.hltv
ls
rm *.json
vim unet.py
$PYTHON test_json.py
ls
ls -l
vim test_json.py
$PYTHON test_json.py
ls
ls hide_data/
source aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
cd apps
cd unet_bench/
ls
vim default_profiling_8
vim default_profiling_810565.hltv
ls
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-1 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256.pt
echo $HANABA_LOGS
echo $HABANA_PROFILE
export HABANA_PROFILE=1
export HABANA_PROF_CONFIG=~/.habana/prof_config.json
hl-prof-config -gaudi -e off -g 1-100
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-1 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256.pt
hl-prof-config -gaudi -e off -g 1-100 --json
hl-prof-config -gaudi -e off -g 1-100 --json on
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-prof-1 --epochs 50 --image-size 32 --cache-path kaggle_cache
hl-prof-config -gaudi -e off -g 1-100 --json off
hl-prof-config -gaudi -e off -g 1-100 --hltv all
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-prof-1 --epochs 50 --image-size 32 --cache-path kaggle_cache
hl-prof-config -gaudi -e off -g 1-100 --hltv off
hl-prof-config -gaudi -e off -g 1-100 --csv all
vim ~/.habana/prof_config.json
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-prof-1 --epochs 50 --image-size 32 --cache-path kaggle_cache
hl-prof-config -gaudi -p mem
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-prof-1 --epochs 50 --image-size 32 --cache-path kaggle_cache
ls ~/.habana
rm ~/.habana/prof_config.json
hl-prof-config --csv all --trace-analylzer on -p enq -gaudi -g 1-100
hl-prof-config --json-compressed all --trace-analyzer on -p enq -gaudi -g 1-100
hl-prof-config
hl-prof-config -h
hl-prof-config --invoc json
hl-prof-config --c gaudi
hl-prof-config -c gaudi
hl-prof-config --trace-analyzer
hl-prof-config --trace-analyzer on
hl-prof-config --memory-reuse on
hl-prof-config --list-templates
hl-prof-config -p
hl-prof-config -p mem
hl-prof-config -h
hl-prof-config -p enq
vim ~/.habana/prof_config.json
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-prof-1 --epochs 50 --image-size 32 --cache-path kaggle_cache
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-prof-2 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-2.pt
vim ~/.habana/prof_config.json
hl-prof-config -gaudi -g 1-100
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-prof-2 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-2.pt
ls
vim ~/.habana/prof_config.json
hl-prof-config -gaudi -g 1-120
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-prof-3 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-3.pt
ls
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-prof-3 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-3.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-prof-4 --epochs 1 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-4.pt --batch-size 960
ls
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-prof-4 --epochs 1 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-4.pt --batch-size 960
ls
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-prof-5 --epochs 1 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-5.pt
l
rm *.json
ls
mkdir hide_data
cd hide_data/
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 1 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
vim unet.py
vim ../unet.py
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 1 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
vim ../unet.py
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 1 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
source aevard_venv/bin/activate
ls
export PYTHON=/home/aevard/aevard_venv/bin/python
export PYTHONPATH=/home/aevard/Model-References:/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
hl-smi -Q utilization.aip
hl-smi -Q utilization.aip -f csv
hl-smi -Q uuid,utilization.aip -f csv
hl-smi -Q bus_id,uuid,utilization.aip -f csv
hl-smi -Q serial,uuid,utilization.aip -f csv
hl-smi -Q name,,uuid,utilization.aip -f csv
hl-smi -Q bus_id,uuid,utilization.aip -f csv
hl-smi -Q bus_id,uuid,utilization.aip,memory.used -f csv
hl-smi -Q bus_id,uuid,utilization.aip,memory.used -f csv -l
hl-smi -Q timestamp,bus_id,uuid,utilization.aip,memory.used -f csv -l
hl-smi -d PRODUCT
hl-smi
hl-smi -n ports  -i  0000:08:00.0
hl-smi -n ports
hl-smi -n ports -i 0000:08:00.0
hl-smi -n ports -i 0000:01:00.0
hl-smi -n stats
hl-smi -d ROW_REPLACEMENT
top
hl-smi -q memory.free,memory.used -f csv
hl-smi -Q memory.free,memory.used -f csv
hl-smi -Q uuid.memory.free,memory.used -f csv
hl-smi -Q uuid,memory.free,memory.used -f csv
hl-smi -Q timestamp,bus_id,memory.free,memory.used -f csv
top
hl-smi -Q timestamp,bus_id,memory.free,memory.used -f csv
hl-smi -Q timestamp,bus_id,memory.free,memory.used -f csv -l
hl-smi topo -c -N
hl-smi -Q timestamp,bus_id,memory.free,memory.used -f csv
top
hl-smi -Q timestamp,bus_id,memory.free,memory.used -f csv
ls
top
ls
vim ~/.habana/prof_config.json
top
cd apps/unet_bench/
ls
cd hide_data/
ls
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
ls
exit
ls
cd apps
ls
cd unet_bench/
ls
cd
source aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export PYTHONPATH=/home/aevard/Model-References:/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
ls
ls habana_logs/
cd apps/unet_bench/
ls
cd hide_data/
ls
cd ..
hl-prof-config -gaudi -g 1-6000
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 1 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
ls
cd hide_data/
ls
rm *.json
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 1 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
ls
ls ..
ls ~/.habana
hl-prof-config -h
hl-prof-config -p enq
hl-prof-config --trace-analyzer on
hl-prof-config --invoc json
hl-prof-config -h
hl-prof-config -g 1-6000
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
hl-prof-config --invoc json
hl-prof-config -g 1-6000 --trace-analyzer on
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
rm ~/.habana/prof_config.json -o ~/apps/unet_bench/hide_data --gaudi2 --chip gaudi
rm ~/.habana/prof_config.json -o ~/apps/unet_bench/hide_data --gaudi2
hl-prof-config -o ~/apps/unet_bench/hide_data --gaudi2
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
hl-prof-config -h
hl-prof-config -p mem
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
hl-prof-config -h
hl-prof-config --gaudi2 -o ~/apps/unet_bench/hide_data --invoc json -p enq -g 1-6000
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
hl-prof-config --invoc json
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
hl-prof-config
ls
vim default_profiling_1152394.hltv
hl-prof-config -h
hl-prof-config -c gaudi
hl-prof-config --gaudi
hl-prof-config -h
hl-prof-config --fuser on
hl-prof-config --memory-reuse on
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
export HABANA_PROFILE=1
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
ls
$PYTHON test_json.py
vim test_json.py
$PYTHON test_json.py
ls
rm *.json
hl-prof-config -h
hl-prof-config --invoc csv
$PYTHON test_json.py
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
hl-prof-config
hl-prof-config -h
hl-prof-config --invoc json
hl-prof-config -p mem
hl-prof-config --merged json
hl-prof-config --memory-reuse off
hl-prof-config --fuser off
hl-prof-config -g 1-50
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
hl-prof-config -p device-acq
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
hl-prof-config -p enq
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
hl-prof-config -p multi-enq
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
hl-prof-config -g 1-10
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
hl-prof-config -g 1-5,11-15
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 32-prof-hidden --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-prof-hidden.pt
hl-prof-config -p enq -g 1-2000
$PYTHON ../unet.py --hpu --use_lazy_mode --run-name 256-prof-1-epoch --epochs 1 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-prof-1-epoch.pt
time $PYTHON ../unet.py --hpu --use_lazy_mode --run-name 256-prof-2-epoch --epochs 2 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-prof-2-epoch.pt
cd apps/unet_bench/
ls
ls -l
vim unet.py
ls hide_data/
ls
ls hide_data/
la
ls hide_data/
hl-prof-config -h
ls ~/.habana_logs/
ls ~/habana_logs/
ls hide_data/
vim ~/.habana/prof_config.json
ls hide_data/
ls
ls ~
rm ~/.habana/prof_config.json
ls ~/habana_logs/
ls
ls hide_data/
hl-prof-config
hl-prof-config -v
hl-prof-config -h
hl-prof-config -s
hl-prof-config --merged json
ls
ls hide_data/
ls ~/.habana
vim ~/.habana/prof_config.json
ls /home/aevard/apps/unet_bench/hide_data
vim ~/.habana/prof_config.json
hl-prof-config --trace-analyzer
hl-prof-config --trace-analyzer on
vim ~/.habana/prof_config.json
ls hide_data/
ls
mv test_json.py hide_data/
mv hide_data/test_json.py .
ls
ls hide_data/
vim ~/.habana/prof_config.json
ls hide_data/
rm hide_data/*.csv
ls hide_data/
vim test_json.py
python3 test_json.py
vim test_json.py
python3 test_json.py
vim test_json.py
python3 test_json.py
vim test_json.py
python3 test_json.py
hl-prof-config -h
ls -l hide_data/
hl-prof-config -h
ls -l hide_data/
ls hide_data/
rm hide_data/*.json
ls hide_data/
time $PYTHON unet.py --hpu --use_lazy_mode --run-name fake --epochs 1 --image-size 32 --cache-path kaggle_cache --weights-file fake.pt
source ~/aevard_venv/bin/activate
time $PYTHON unet.py --hpu --use_lazy_mode --run-name fake --epochs 1 --image-size 32 --cache-path kaggle_cache --weights-file fake.pt
export PYTHON=/home/aevard/aevard_venv/bin/python
time $PYTHON unet.py --hpu --use_lazy_mode --run-name fake --epochs 1 --image-size 32 --cache-path kaggle_cache --weights-file fake.pt
ls hide_data/
top
hl-smi -Q timestamp,module_id
hl-smi -Q timestamp,module_id -f csv
hl-smi -Q timestamp,module_id,utilization.aip -f csv
touch books.txt
vim books.txt
echo stuff >> books.txt
vim books.txt
echo stuff >> books.txt
vim books.tx
vim books.txt
touch books.txt
rm books.txt
ls
cd apps/unet_bench/
ls
mkdir bash_scripts
cd bash_scripts
vim build-hl-smi-csv
ls
./build-hl-smi-csv basic_try
chmod build-hl-smi-csv
chmod build-hl-smi-csv 771
chmod 771 build-hl-smi-csv1
ls
chmod 771 build-hl-smi-csv
./build-hl-smi-csv basic_try
rm basic_try
./build-hl-smi-csv basic_try
rm basic_try
./build-hl-smi-csv basic_try
vim build-hl-smi-csv
./build-hl-smi-csv basic_try-2
cd apps/unet_bench/
ls
cd bash_scripts/
ls
./build-hl-smi-csv basic_try-2
./build-hl-smi-csv basic_try_try-3
source aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
$PYTHON unet-5.py --hpu --use_lazy_mode --run-name 128-5-layer-1 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-5-l-1.pt
vim unet-5.py
$PYTHON unet-5.py --hpu --use_lazy_mode --run-name 128-5-layer-1 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-5-l-1.pt
$PYTHON unet-5.py --hpu --use_lazy_mode --run-name 128-5-layer-2 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-5-l-2.pt
$PYTHON unet-5.py --hpu --use_lazy_mode --run-name 128-5-layer-3 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-5-l-3.pt$PYTHON unet-3.py --hpu --use_lazy_mode --run-name 256-3-layer-1 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-3-l-1.pt
vim unet-5.py
ls weights/
rm weights/*.pt
$PYTHON unet-3.py --hpu --use_lazy_mode --run-name 256-3-layer-1 --layers 3 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-3-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-3-layer-1 --layers 3 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-3-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-3-layer-2 --layers 3 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-3-l-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-3-layer-3 --layers 3 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-3-l-3.pt
ls
ls performance/
source aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-4-layer-1 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-3-l-1.ptrm
rm performance/128-4-layer-1.txt
rm performance/128-5-layer-1.txt
$PYTHON unet-3.py --hpu --use_lazy_mode --run-name 128-3-layer-1 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-3-l-1.pt
$PYTHON unet-3.py --hpu --use_lazy_mode --run-name 128-2-layer-1 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-3-l-2.pt
$PYTHON unet-3.py --hpu --use_lazy_mode --run-name 128-2-layer-3 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-3-l-3.pt
ls bash_scripts/
cd bash_scripts/
vim build-hl-smi-csv
vim basic_trey
vim basic_try
rm basic_try
cd ..
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-4-layer-1 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-4-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-4-layer-2 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-4-l-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-4-layer-3 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-4-l-3.pt
source aevard_venv/bin/activate
ls
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
ls
cd apps/unet_bench/
ls
ls hide_data/
ls
rm *.hltv
rm performance32-prof-hidden.txt
ls
vim unet.py
ls performance/
cd performance/
mkdir pre-csv
mv *.txt pre-csv/
ls
cd ..
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-csv-check --epochs 25 --image-size 32 --cache-path kaggle_cache --weights-file 32-csv.pt
vim performance/32-csv-check.txt
top
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-4-layer-1 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-3-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-4-layer-1 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-3-l-1.pt$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-4-layer-1 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-4-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-4-layer-1 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-4-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-4-layer-2 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-4-l-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-4-layer-3 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 256-4-l-3.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-csv-check --epochs 25 --image-size 32 --cache-path kaggle_cache --weights-file 32-csv.pt
$PYTHON unet-5.py --hpu --use_lazy_mode --run-name 256-5-layer-1 --layers 5 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-5-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-5-layer-1 --layers 5 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-5-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-5-layer-2 --layers 5 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-5-l-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-5-layer-3 --layers 5 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-5-l-3.pt
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-no-cache-1 --im-cache False --epochs 50 --image-size 256 --cache-path kaggle_1_256_cache --weights-file 256-no-cache-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-no-cache-1 --im-cache --epochs 50 --image-size 256 --cache-path kaggle_1_256_cache --weights-file 256-no-cache-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-3-layer-3 --layers 3 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-3-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-3-layer-2 --layers 3 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-3-l-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-5-layer-3 --layers 5 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-5-l-1.pt
`$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-5-layer-1 --layers 5 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-5-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-5-layer-1 --layers 5 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-5-l-1.pt
exit
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-no-cache-3 --im-cache --epochs 50 --image-size 256 --cache-path kaggle_1_256_cache --weights-file 256-no-cache-3.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-4-layer-3 --epochs 50 --image-size 32 --cache-path kaggle_32_cache --weights-file 256-4-l-31.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-4-layer-3 --epochs 50 --image-size 32 --cache-path kaggle_32_cache --weights-file 32-4-l-32.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-4-layer-3 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-4-l-31.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-4-layer-2 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-4-l-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-3-layer-3 --layers 3 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-3-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-3-layer-1 --layers 3 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-3-l-1.pt
exit
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-no-cache-2 --im-cache --epochs 50 --image-size 256 --cache-path kaggle_2_256_cache --weights-file 256-no-cache-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-5-layer-1 --layers 5 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-5-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-5-layer-2 --layers 5 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-5-l-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-4-layer-3 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-4-l-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-4-layer-1 --epochs 50 --image-size 32 --cache-path kaggle_cache --weights-file 32-4-l-1.pt
exit
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
ls
ls data
ls performance/
ls
cd bash_scripts/
ls
vim build-hl-s
./build-hl-smi-csv no-caching
./build-hl-smi-csv no-caching-real
vim build-hl-smi-csv
./build-hl-smi-csv
./build-hl-smi-csv size_small-1
./build-hl-smi-csv size_small-2
./build-hl-smi-csv size_small-3
exit
source aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
top
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-b-64-1 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-64-1.pt
source  aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-b-128-1 --batch-size 128 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-128-1.pt
source aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-b-32-1 --batch-size 32 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-32-1.pt
source aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
cd bash_scripts/
ls
vim build-hl-smi-csv
./build-hl-smi-csv add_mem_data
ls -l
export HABANA_LOGS=~/habana_logs
./build-hl-smi-csv add_mem_data
cd apps/unet_bench/bash_scripts/
ls -l
export HABANA_PROFILE=1
source aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
ls
top
ls
ls weights/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-5-l-1 --layers 5 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-5-l-1.pt
vim unet.py
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-5-layer-1 --layers 5 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-5-l-1.pt
top
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-5-layer-1 --layers 5 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-5-l-1.pt
export PYTHON=/home/aevard/aevard_venv/bin/python
source aevard_venv/bin/activate
export HABANA_LOGS=~/habana_logs
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-5-layer-2 --layers 5 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-5-l-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-5-layer-2 --layers 5 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-5-l-2.pt
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/bash_scripts/
ls
./build-hl-smi-csv redo-5-layers.txt
vim build-hl-smi-csv
./build-hl-smi-csv redo-5-layers.txt
export PYTHON=/home/aevard/aevard_venv/bin/python
source aevard_venv/bin/activate
export HABANA_LOGS=~/habana_logs
cd apps/unet_bench/
ls performance/
rm performance/256-5-layer-1.txt
rm performance/256-5-layer-2.txt
rm performance/256-5-layer-3.txt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-5-layer-3 --layers 5 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-5-l-3.pt
top
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-5-layer-4 --layers 5 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-5-l-4.pt
top
cd apps/unet_bench/
ls
cd bash_scripts/
ls
vim power2.txt
ls
touch bla
vim bla
touch bla
vim bla
rm bla
top
export PYTHON=/home/aevard/aevard_venv/bin/python
source aevard_venv/bin/activate
cd apps/unet_bench/
topexport HABANA_PROFILE=1
export HABANA_LOGS=~/habana_logs
cd bash_scripts/
vim build-hl-smi-csv
hl-smi -Q timestamp,module_id,utilization.aip,power.draw,memory.used -f csv
vim build-hl-smi-csv
ls
rm size_small-2
rm basic_try
rm basic_try_try-3
rm basic_try-2
ls
rm size_small-
rm size_small-1
ls
ls -l
rm size_small-3
ls -l
vim build-hl-smi-csv
./build-hl-smi-csv power-test-256
ls
top
vim build-hl-smi-csv
vim power-test-256
vim power-test-256
vim build-hl-smi-csv
hl-smi -Q timestamp,module_id,utilization.aip,power.draw,memory.used -f csv
./build-hl-smi-csv corrupt?
vim corrupt\?
./build-hl-smi-csv corrupt?
vim build-hl-smi-csv
./build-hl-smi-csv corrupt?
vim build-hl-smi-csv
top
vim build-hl-smi-csv
./build-hl-smi-csv
./build-hl-smi-csv power2.txt
vim build-hl-smi-csv
./build-hl-smi-csv power2.txt
ls
source aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
ls
cd apps/unet_bench/bash_scripts/
ls
vim build-hl-smi-csv
vim build-hl-smi-csv
cd ..
$HABANA_PROFILE
echo $HABANA_PROFILE
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-power-1 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-p-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-power2-1 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-p2-1.pt
export PYTHON=/home/aevard/aevard_venv/bin/python
source aevard_venv/bin/activate
cd apps/unet_bench
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-power-2 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-p-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-power2-2 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-p2-2.pt
source aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
echo $HABANA_PROFILE
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-power-3 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-p-3.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-power2-3 --epochs 50 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-p2-3.pt
lls
ls
cd apps/unet_bench/
ls
ls data
ls data/kaggle_duped/
ls data/kaggle_duped/TCGA_DU_7013_19860523
ls data/kaggle_duped
rm -rf data/kaggle_duped
mkdir data/kaggle_duped
ls data/kaggle_duped/
ls
ls data
du
du -s
du -s data/kaggle_duped/
cd
source aevard_venv/bin/activate
ls
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
ls
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128_d --data-path kaggle_duped --epochs 1 --im-cache --image-size 128 --cache-path kaggle_duped_cache --weights-file 128_duped.pt
top
cd apps/unet_bench/
ls
cd data/
ls
ls -l
rm -rf kaggle_2_256_cache/
rm -rf kaggle_1_256_cache/
cp kaggle_3m kaggle_duped
cp -r kaggle_3m kaggle_duped
cd kaggle_duped/
ls
cd TCGA_DU_6408_19860521/
ls
cd ..
ls
cd ..
cd bash_scripts/
ls
python3 duplicate.py
ls
vim build-hl-smi-csv
./build-hl-smi-csv dup-cache.txt
export HABANA_LOGS=~/habana_logs
./build-hl-smi-csv dup-cache.txt
exit
cd apps/unet_bench/bash_scripts/
ls
rm dup-cache.txt
rm 'corrupt?'
./build-hl-smi-csv dup-cache.txt
export HABANA_LOGS=~/habana_logs
export PYTHON=/home/aevard/aevard_venv/bin/python
./build-hl-smi-csv dup-cache.txt
ls
./build-hl-smi-csv duped-batch.txt
./build-hl-smi-csv normal-batch.txt
./build-hl-smi-csv nohalf-duped-batch.txt
export PYTHON=/home/aevard/aevard_venv/bin/python
source aevard_venv/bin/activate
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-batch-256-2 --epochs 50 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 128-b-256-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-batch-64-2 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 128-b-64-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-duped-128-2 --epochs 50 --image-size 128 --cache-path kaggle_duped_cache --weights-file 128-b-128-2.pt
cd apps/unet_bench/
cd
source aevard_venv/bin/activate
cd apps/unet_bench/
export PYTHON=/home/aevard/aevard_venv/bin/python
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128_d --data-path kaggle_duped --epochs 1 --im-cache --image-size 128 --cache-path kaggle_duped_cache --weights-file 128_duped.pt
ls
ls data
`$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-batch-256-1 --epochs 50 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 128-b-256-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-batch-256-1 --epochs 50 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 128-b-256-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-batch-64-1 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 128-b-64-1.pt
ls data
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-duped-128-1 --epochs 50 --image-size 128 --cache-path kaggle_duped_cache --weights-file 128-b-128-1.pt
export PYTHON=/home/aevard/aevard_venv/bin/python
source aevard_venv/bin/activate
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-batch-256-3 --epochs 50 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 128-b-256-3.pt
top
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-batch-64-3 --epochs 50 --image-size 128 --cache-path kaggle_128_cache --weights-file 128-b-64-3.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-duped-128-3 --epochs 50 --image-size 128 --cache-path kaggle_duped_cache --weights-file 128-b-128-3.pt
top
cd apps/unet_bench/
ls
cd data
ls
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-normal-1 --epochs 50
cd ..
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-normal-1 --epochs 50 &
top
jobs
psps T
ps T
top
exit
top
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-normal-3 --epochs 50 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-normal-3.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-normal-3 --epochs 50 --image-size 32 --batch-size 64 --cache-path kaggle_32_cache --weights-file 32-normal-3.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-normal-3 --epochs 50 --image-size 128 --batch-size 64 --cache-path kaggle_128_cache --weights-file 128-normal-3.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-normal-3 --epochs 25 --image-size 256 --batch-size 64 --cache-path kaggle_256_cache --weights-file 256-normal-3.pt
exit
export HABANA_LOGS=~/habana_logs
cd apps/unet_bench/bash_scripts/
ls
ls ../data
top
ls ../data
ls
rm *.txt
ls
rm add_mem_data
rm no-caching
rm power-test-256
rm no-caching-real
ls
vim hl-smi-csv
vim build-hl-smi-csv
top
ls
./build-hl-smi-csv normal-64.txt
./build-hl-smi-csv normal-32.txt
./build-hl-smi-csv normal-128.txt
./build-hl-smi-csv normal-256.txt
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-normal-1 --epochs 50 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-normal-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-normal-1 --epochs 50 --image-size 32 --batch-size 64 --cache-path kaggle_32_cache --weights-file 32-normal-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-normal-1 --epochs 50 --image-size 128 --batch-size 64 --cache-path kaggle_128_cache --weights-file 128-normal-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-normal-1 --epochs 25 --image-size 256 --batch-size 64 --cache-path kaggle_256_cache --weights-file 256-normal-1.pt
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-normal-2 --epochs 50 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-normal-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-normal-2 --epochs 50 --image-size 32 --batch-size 64 --cache-path kaggle_32_cache --weights-file 32-normal-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-normal-2 --epochs 50 --image-size 128 --batch-size 64 --cache-path kaggle_128_cache --weights-file 128-normal-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-normal-2 --epochs 25 --image-size 256 --batch-size 64 --cache-path kaggle_256_cache --weights-file 256-normal-2.pt
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
$PYTHON unet.py --hpu --use_lazy_mode --run-name 32-cache --epochs 1 --image-size 32 --batch-size 64 --im-cache --cache-path kaggle_32_cache --weights-file 32-cache.pt
ls ../data
ls data
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-duped --epochs 1 --image-size 64 --data-path kaggle_duped --batch-size 256 --cache-path kaggle_duped_64_cache --weights-file 256-cache.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-duped --epochs 1 --image-size 64 --data-path kaggle_duped --batch-size 256 --cache-path kaggle_duped_64_cache --im-cache --weights-file 256-cache.pt
cd apps/unet_bench/
aevard@habana-02:~$ cd apps/unet_bench/
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-duped-3 --epochs 25 --image-size 64 --batch-size 256 --cache-path kaggle_duped_64_cache --weights-file 64-d-3.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-duped-3 --epochs 10 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 256-d-3.pt
cd apps/unet_bench/
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-duped-1 --epochs 25 --image-size 64 --batch-size 256 --cache-path kaggle_duped_64_cache --weights-file 64-d-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-duped-1 --epochs 10 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 256-d-1.pt
cd apps/unet_bench/
export HABANA_LOGS=~/habana_logs
cd scripts
cd bash_scripts/
ls
ls ../data
./build-hl-smi-csv duped-64.txt
./build-hl-smi-csv duped-128.txt
top
cd apps/unet_bench/
aevard@habana-02:~$ cd apps/unet_bench/
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-duped-2 --epochs 25 --image-size 64 --batch-size 256 --cache-path kaggle_duped_64_cache --weights-file 64-d-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-duped-2 --epochs 10 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 256-d-2.pt
top
hl-smi -Q timestamp,module_id,utilization.aip,power.draw,memory.used -f csv
export HABANA_LOGS=~/habana_logs
top
hl-smi -Q timestamp,module_id,utilization.aip,power.draw,memory.used -f csv
hl-smi -Q timestamp,module_id,utilization.aip,power.draw,memory.used -f csv -l 1
top
hl-smi -Q timestamp,module_id,utilization.aip,power.draw,memory.used -f csv -l 1
cd apps/unet_bench/performance/
ls
vim distrib-test-
vim distrib-test-2.txt
ls
vim distrib-test-3.txt
top
vim distrib-test-3.txt
source aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
ls
cd ..
ls
cd ..
ls
cd apps/unet_bench/
ls
$PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-1 --weights-file d-test-1
$PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-1 --weights-file d-test-1.pt
echo $WORLD_SIZE
$PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-1 --weights-file d-test-1.pt --world-size 2
vim unet_distrib.py
$PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-1 --weights-file d-test-1.pt --world_size 2
$PYTHON pip install mpi4py
$PYTHON install mpi4py
$PYTHON -m pip install mpi4py
vim unet_
vim unet_distrib.py
$PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-1 --weights-file d-test-1.pt --world_size 2
MASTER_ADDR=localhost
echo $MASTER_ADDR
MASTER_PORT=12345
$PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-1 --weights-file d-test-1.pt --world_size 2
echo $MASTER_ADDR
$PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-1 --weights-file d-test-1.pt --world_size 2
$PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-1 --weights-file d-test-1.pt --world-size 2
ls
cd performance/
ls
vim distrib-test-1.txt
$PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-2 --weights-file d-test-2.pt --world-size 2
cd ..
$PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-2 --weights-file d-test-2.pt --world-size 2
vim performance/distrib-test-2.txt
mpirun -np 2 $PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-2 --weights-file d-test-2.pt --world-size 2
export export HABANA_LOGS=~/habana_logs
export HABANA_LOGS=~/habana_logs
mpirun -np 2 $PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-2 --weights-file d-test-2.pt --world-size 2
mpirun -np 2 $PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-2 --3eights-file d-test-3.pt --world-size 2
mpirun -np 2 $PYTHON unet_distrib.py --hpu --distributed --use_lazy_mode --run-name distrib-test-3 --weights-file d-test-3.pt --world-size 2
source ~/aevard_venv/bin/activate
export HABANA_LOGS=~/habana_logs
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
ls
ls performance/
mpirun -n 2 --bind-to core --map-by slot:PE=7 --rank-by core  --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --run-name distrib-test-4 --epochs 1 --kaggle_cache --weights-file d-t-4.pt
ls
mpirun -n 2 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name distrib-test-4 --epochs 1 --kaggle_cache --weights-file d-t-4.pt
mpirun -n 2 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name distrib-test-4 --epochs 1 --kaggle_cache --weights-file d-t-4.ptmpirun -n 2 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name distrib-test-4 --epochs 1 --kaggle-cache --weights-file d-t-4.pt
mpirun -n 2 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name distrib-test-4 --epochs 1 --kaggle-cache --weights-file d-t-4.pt
mpirun -n 2 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt
mpirun -n 2 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2`
mpirun -n 2 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
$RANK
echo $RANK
vim unet_distrib.py
mpirun -n 2 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
vim unet_distrib.py
mpirun -n 2 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
mpirun -n 2 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2 --distributed
vim unet_distrib.py
mpirun -n 2 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2 --distributed
mpirun -np 2 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
mpirun -np 1 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 1
mpirun -np 1 --bind-to core --map-by slot:PE=7 --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
cd ..
ls
cd ..
ls
cd Model-References/
ls
cd PyTorch/
ls
cd com
cd computer_vision/
ls
cd classification/
ls
cd torchvision/
ls
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --print-freq=1 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=False
$PYTHON -m install requirements.txt
$PYTHON -m pip install requirements.txt
export PYTHONPATH=/home/aevard/Model-References:/home/aevard/aevard_venv/bin/python
$PYTHON -m pip install -r requirements.txt
top
ls
cd ..
cd
cd apps/unet_bench/
ls
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
/etc/init.d/ssh restart '-p 560'
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
vim utils.py
echo $MASTER_PORT
export MASTER_PORT=12300
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
hostname -I
export MASTER_ADDR=140.221.80.102
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
ls
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 1
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
top
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
export HABANA_LOGS=~/habana_logs
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-4 --epochs 1 --cache-path kaggle_cache --weights-file d-t-4.pt --world-size 2
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-5 --epochs 10 --cache-path kaggle_cache --weights-file d-t-5.pt --world-size 2
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-6 --epochs 10 --cache-path kaggle_cache --weights-file d-t-6.pt --world-size 2
$PYTHON unet.py --hpu --use_lazy_mode --run-name distrib-compare --epochs 10 --cache-path kaggle_cache --weights-file d-c-1.pt
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-6 --epochs 10 --cache-path kaggle_cache --weights-file d-t-6.pt --world-size 2
$PYTHON unet.py --hpu --use_lazy_mode --run-name distrib-compare --epochs 10 --cache-path kaggle_cache --weights-file d-c-1.pt
mpirun -np 4 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name distrib-test-6 --epochs 10 --cache-path kaggle_cache --weights-file d-t-6.pt --world-size 4
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 32-normal-cards-2 --epochs 50 --cache-path kaggle_cache --weights-file 32-n-c-2.pt --world-size 2
export HABANA_LOGS=~/habana_logs
cd apps/unet_bench/
vim unet_distrib.py
cd performance/
ls
vim distrib-test-4.txt
ls
vim distrib-test-5.txt
vim ../unet_distrib.py
vim distrib-test-5.txt
vim distrib-test-6.txt
top
vim distrib-test-6.txt
vim ../unet_distrib.py
vim ../unet.oy
vim ../unet.py
vim ../unet_distrib.py
top
ls
rm *.txt
cd ../scripts
cd ../bash_scripts/
ls
vim build-hl-smi-csv
./build-hl-smi-csv 32-normal-cards-2.txt
rm 32-normal-cards-2.txt
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
cd apps/unet_bench/
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 32-normal-cards-2 --epochs 50 --cache-path kaggle_cache --weights-file 32-n-c-2.pt --world-size 2
mpirun -np 1 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 32-normal-cards-1 --epochs 50 --cache-path kaggle_cache --weights-file 32-n-c-1.pt --world-size 1
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 64-normal-cards-2 --epochs 50 --cache-path kaggle_cache --weights-file 64-n-c-2.pt --world-size 2
mpirun -np 128 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 128-normal-cards-2 --epochs 50 --cache-path kaggle_cache --weights-file 128-n-c-2.pt --world-size 2
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 128-normal-cards-2 --image-size 128 --epochs 50 --cache-path kaggle_128_cache --weights-file 128-n-c-2.pt --world-size 2
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 256-normal-cards-2 --image-size 256 --epochs 50 --cache-path kaggle_256_cache --weights-file 256-n-c-2.pt --world-size 2
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 256-normal-cards-2 --image-size 256 --epochs 25 --cache-path kaggle_256_cache --weights-file 256-n-c-2.pt --world-size 2
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 64-duped-cards-2 --epochs 25 --image-size 64 --batch-size 256 --cache-path kaggle_duped_64_cache --weights-file 64-d-c-2.pt --world-size 2mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 128-duped-cards-2 --epochs 10 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 128-d-c-2.pt --world-size 2
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 128-duped-cards-2 --epochs 10 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 128-d-c-2.pt --world-size 2
mpirun -np 4 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 128-duped-cards-4 --epochs 10 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 128-d-c-4.pt --world-size 4
ls data
$PYTHON unet.py --run-name 256-cacher --epochs 1 --image-size 256 --im-cache --cache-path kaggle_duped_256_cache --weights-file 256-c.pt
export HABANA_LOGS=~/habana_logs
cd apps/unet_bench/
ls
rename
mv -r bash_scripts/ scripts
mv  bash_scripts scripts
ls
cd scripts
ls
./build-hl-smi-csv 32-normal-cards-2.txt
./build-hl-smi-csv 64-normal-cards-2.txt
./build-hl-smi-csv 128-normal-cards-2.txt
./build-hl-smi-csv 256-normal-cards-2.txt
./build-hl-smi-csv 64-duped-cards-2.txt
./build-hl-smi-csv 128-duped-cards-2.txt
./build-hl-smi-csv 128-duped-cards-4.txt
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
cd apps/unet_bench/
cd scripts/
ls
./build-hl-smi-csv 256-duped-cards-4.txt
cd ..
mpirun -np 4 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 256-duped-cards-4 --epochs 10 --image-size 256 --batch-size 256 --cache-path kaggle_duped_256_cache --weights-file 256-d-c-4.pt --world-size 4
export HABANA_LOGS=~/habana_logs
cd apps/unet_bench/
ls
ls data
mpirun -np 4 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 256-duped-cards-4 --epochs 10 --image-size 256 --batch-size 256 --cache-path kaggle_duped_256_cache --weights-file 256-d-c-4.pt --world-size 4
cd scripts/
./build-hl-smi-csv 256-duped-cards-4.txt
ls
cd apps/unet_bench/
ls
ls data
rm data/kaggle_max
rm -rf data/kaggle_max
cd scripts
ls
vim duplicateCache.py
python3 duplicateCache.py
mkdir ../data/kaggle_max_cache
python3 duplicateCache.py
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-max-3 --image-size 64 --batch-size 64 --epochs 2 --cache-path kaggle_max_cache --weights-file 64-m-3.pt
export HABANA_LOGS=~/habana_logs
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
cd ..
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-max-3 --image-size 64 --batch-size 64 --epochs 2 --cache-path kaggle_max_cache --weights-file 64-m-3.pt
cd apps/unet_bench/
cd scripts/
ls
vim duplicate.py
python3 duplicate.py
ls ../data
ls
cd ../data
ls kaggle_
ls kaggle_3
ls kaggle_3m
cd ../scripts/
ls
vim duplicate.py
python3 duplicate.py
mkdir ../data/kaggle_max
python3 duplicate.py
top
ls -l ../data
du -hs ../data/kaggle_max_128_cache/
cd ..
ls
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-max-3 --image-size 128 --batch-size 64 --epochs 2 --cache-path kaggle_max_128_cache --weights-file 128-m-3.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-max-2 --image-size 64 --batch-size 64 --epochs 2 --cache-path kaggle_max_cache --weights-file 64-m-2.pt
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
cd apps/unet_bench/
mpirun -np 4 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 256-duped-cards-4 --epochs 10 --image-size 256 --batch-size 256 --cache-path kaggle_duped_256_cache --weights-file 256-d-c-4.pt --world-size 4
mpirun -np 8 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 256-duped-cards-8 --epochs 10 --image-size 128 --batch-size 256 --cache-path kaggle_duped_cache --weights-file 256-d-c-8.pt --world-size 8
ls data
ls data/kaggle_
ls data/kaggle_max
mpirun -np 4 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 64-duped-cards-4 --epochs 25 --image-size 64 --batch-size 256 --cache-path kaggle_duped_64_cache --weights-file 64-d-c-4.pt --world-size 4
top
ls data/kaggle_duped_cache/
ls
cd scripts
ls
python3 duplicateCache.py
vim dup
vim duplicateCache.py
python3 duplicateCache.py
vim duplicateCache.py
python3 duplicateCache.py
./build-hl-smi-csv 128-max-1.txt
cd ..
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-max-1 --image-size 128 --batch-size 64 --epochs 2 --cache-path kaggle_max_128_cache --weights-file 128-m-1.pt
ls data/kaggle_max_128_cache/
vim dataset.py
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-max-1 --image-size 128 --batch-size 64 --epochs 2 --cache-path kaggle_max_128_cache --weights-file 128-m-1.pt
rm -rf data/kaggle_max_128_cache/
mkdir data/kaggle_max_128_cache
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-max-1 --image-size 128 --batch-size 64 --epochs 2 --cache-path kaggle_max_128_cache --weights-file 128-m-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-max-2 --image-size 128 --batch-size 64 --epochs 2 --cache-path kaggle_max_128_cache --weights-file 128-m-2.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-max-1 --image-size 64 --batch-size 64 --epochs 2 --cache-path kaggle_max_cache --weights-file 64-m-1.pt
export HABANA_LOGS=~/habana_logs
cd apps/unet_bench/
cd scripts/
ls
rm 256-duped-cards-4.txt
./build-hl-smi-csv 256-duped-cards-4.txt
ls
rm 256-duped-cards-4.txt
top
./build-hl-smi-csv 128-duped-cards-8.txt
top
vim build-hl-smi-csv
hl-smi -Q timestamp,module_id,utilization.aip,power.draw,memory.used -f     csv
ls
./build-hl-smi-csv 64-duped-cards-4.txt
ls
ls ../data
mkdir ../data/max_128_cache
cd ..
ls data
mv data/max_128_cache kaggle_max_128_cache
ls data
mv kaggle_max_128_cache data/kaggle_max_128_cache
ls data
ls data/kaggle_max_128_cache/
ls -l data
ls data
$PYTHON unet.py --hpu --use_lazy_mode --run-name 128-max-1 --image-size 128 --batch-size 64 --epochs 2 --cache-path kaggle_max_128_cache --weights-file 128-m-1.pt
cd scripts
/build-hl-smi-csv 128-max-1.txt
./build-hl-smi-csv 128-max-1.txt
rm 128-max-1.txt
./build-hl-smi-csv 128-max-1.txt
vim duplicateCache.py
python3 duplicateCache.py
ls ../data
vim duplicateCache.py
ls ../data
python3 duplicateCache.py
rm 128-max-1.txt
./build-hl-smi-csv 128-max-1.txt
./build-hl-smi-csv 64-max-1.txt
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
cd apps/unet_bench/
ls performance/
rm performance/*.txt
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 256 --cache-path kaggle_cache --weights-file 64-test-c-d.pt --world-size 2
vim unet_distrib.py
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 256 --cache-path kaggle_cache --weights-file 64-test-c-d.pt --world-size 2
$PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name 64-test-cards-1 --epochs 5 --image-size 64 --batch-size 256 --cache-path kaggle_cache --weights-file 64-test-c-1.pt --world-size 1
mpirun -np 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet_distrib.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 256 --cache-path kaggle_cache --weights-file 64-test-c-d.pt --world-size 2
$PYTHON unet_distrib.py --hpu --use_lazy_mode --run-name 64-test-cards-1 --epochs 5 --image-size 64 --batch-size 256 --cache-path kaggle_cache --weights-file 64-test-c-1.pt --world-size 1
top
ls
cd apps
git@github.com:Adrave/habana-unet-power.git
git clone git@github.com:Adrave/habana-unet-power.git
ls
cd habana-unet-power/
ls
source ~/aevard_venv/bin/activate
top
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
ls
cd unet_bench/
ls
top
git pull
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-batch-64 --epochs 50 --cache-path kaggle_cache --weights-file 64-b-64.pt
mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
vim unet.py
mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
git pull
git status
git restore unet.p
git restore unet.py
git status
git pull
echo $SLURM_PROCID
echo $RANK
vim unet.py
mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
vim unet.py
mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
top
git status
vim unet.py
:q
git status unet.py
git diff unet.py
git status
git restore unet.py
ls
git status
rm habana_log.livealloc.log_0
rm habana_log.livealloc.log_1
ls
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
git status
git pull
cd apps/habana-unet-power/unet_bench/
ls
git status
top
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
ls
exit
source ~/aevard_venv/bin/activate
top
cd apps/habana-unet-power/
ls
top
ls
top
git -b checkout
git checkout
git pull
git checkout -b feature
git show-branch -a
git branch -a
git checkout -b feature/bruce
git checkout feature/bruce
git show-branch -a
git branch -a
git checkout main
git branch -a
git pull
git pull origin main
git fetch
git branch -a
git checkout --track origin/newsletter
git checkout --track origin/feature/bruce
git checkout --track remotes/origin/feature/bruce
git branch -r
git branch -v -a
git switch bruce
git switch feature/bruce
git switch feature
git branch -a
git status
git diff unet.py
git diff unet_bench/
git diff unet_bench/unet.py
cd unet_bench/
git status
git diff unet
git diff unet.py
git switch main
git status
git branch -a
git branch -d feature
git branch -a
git fetch
git branch -a
git switch bruce
git switch feature/bruce
git branch -a
git status
vim unet.py
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
vim unet.py
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-batch-64 --epochs 50 --cache-path kaggle_cache --weights-file 64-b-64.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-batch-64 --epochs 5 --cache-path kaggle_cache --weights-file 64-b-64.pt
exit
top
cd apps/habana-unet-power/
ls
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python'
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
git pull
git branch
git branch main
git checkout main
git stauts
git status
git diff unet_bench/unet.py
git status
git pull
git status
git diff unet_bench/unet.py
:q
cd unet_bench/
vim unet.py
git status
restore unet.py
git restore unet.py
git status
mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
git branch
git checkout mainA
git pull
mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
git pull
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
git branch
vim unet.py
git pull
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
top
vim unet.py
git pull
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
vim unet.py
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
vim unet.py
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
vim unet.py
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2

mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
git status
git pull
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 256-test-cards-2 --epochs 5 --image-size 256 --batch-size 64 --cache-path kaggle_256_cache --weights-file 256-t-c-2.pt --world-size 2
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 256-test-cards-2 --epochs 5 --image-size 256 --batch-size 64 --cache-path kaggle_256_cache --weights-file 256-t-c-2.pt --world-size 2
ls data
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 256-test-cards-2 --epochs 5 --image-size 256 --batch-size 64 --cache-path kaggle_256_cache --weights-file 256-t-c-2.pt --world-size 2
top
exit
top
cd apps/habana-unet-power/
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
top
cd unet_bench
ls
top
git status
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-batch-64 --epochs 50 --cache-path kaggle_cache --weights-file 64-b-64.pt
git checkout main
git branch
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-batch-64 --epochs 50 --cache-path kaggle_cache --weights-file 64-b-64.pt
vim unet.py
git push
git pull
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-batch-64 --epochs 50 --cache-path kaggle_cache --weights-file 64-b-64.pt
vim unet.py
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-batch-64 --epochs 50 --cache-path kaggle_cache --weights-file 64-b-64.pt
git status
git restore unet.py
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-batch-64 --epochs 50 --cache-path kaggle_cache --weights-file 64-b-64.pt
THON unet.py --hpu --use_lazy_mode --run-name 64-batch-64 --epochs 5 --cache-path kaggle_cache --weights-file 64-b-64.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-batch-64 --epochs 5 --cache-path kaggle_cache --weights-file 64-b-64.pt
git status
git restore unet.py
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-batch-64 --epochs 5 --cache-path kaggle_cache --weights-file 64-b-64.pt
git status
$PYTHON unet.py --hpu --use_lazy_mode --run-name 64-batch-64 --epochs 5 --cache-path kaggle_cache --weights-file 64-b-64.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-batch-64 --epochs 5 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-b-64.pt
ls data
cp ../../unet_bench/data/kaggle_256_cache kaggle_256_cache
cp -r ../../unet_bench/data/kaggle_256_cache kaggle_256_cache
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-batch-64 --epochs 5 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-b-64.pt
ls
mv kaggle_256_cache data/kaggle_256_cache
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-batch-64 --epochs 5 --image-size 256 --cache-path kaggle_256_cache --weights-file 256-b-64.pt
exit
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export PYTHONPATH=/home/aevard/Model-References:/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
cd Model-References/
ls
cd TensorFlow/examples/
ls
cd hello_world/
ls
./run_hvd_16gaudi.sh
./run_hvd_8gaudi.sh
exit
/lambda_stor/homes/aevard/aevard_venv/bin/python -m pip freeze | grep scikit-image
top
exit
pwd
ls
ls /lambda_stor/
aevard_venv/bin/python -m pip freeze
aevard_venv/bin/python -m pip freeze | grep scikit-image
/lambda_stor/homes/aevard/aevard_venv/bin/python -m pip freeze | grep scikit-image
/lambda_stor/homes/aevard/aevard_venv/bin/python -m pip freeze | grep scikit-learn
top
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
cd apps/habana-unet-power/
ls
cd unet_bench/
ls
git pull
git checkout feature/bruce
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2 --num-workers 10
vim unet.py
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2 --num-workers 10
top
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2-5 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2 --num-workers 5
top
time mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-b-16 --epochs 5 --image-size 64 --batch-size 16 --cache-path kaggle_cache --weights-file 64-t-b-16.pt --world-size 2
$PYTHON -m pip freeze > bruce.txt
ls
pwd
vim bruce.txt
deactivate
source ~/PT_venv/bin/activate
export PYTHON=~/aevard_venv/bin/python3
export PYTHON=~/PT_venv/bin/python3
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-6 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-w-6.pt --world-size 2 --num-workers 6
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-6 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-w-6.pt --world-size 2 --num-workers 6
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-w-2.pt --world-size 2 --num-workers 2
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-10 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-cw-10.pt --world-size 2 --num-workers 10
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-1 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-w-1.pt --world-size 2 --num-workers 1
echo $PYTHON
git pull
git branch
git checkout main
git diff unet.py
git restore unet.py
git checkout main
git pull
git status
git diff utils.py
git restore utils.py
git branch
git pull
git status
mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-4 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-n-4.pt --world-size 4 --num-workers 4
time mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-4 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-n-4.pt --world-size 4 --num-workers 4
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-4-4 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-w-4.pt --world-size 2 --num-workers 4
time mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-b-128 --epochs 5 --image-size 64 --batch-size 128 --cache-path kaggle_cache --weights-file 64-t-b-128.pt --world-size 2
time mpirun -n 4 --rank-by core python3 unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-4 --epochs 5 --image-size 256 --batch-size 128 --cache-path kaggle_cache --weights-file 64-t-w-6.pt --world-size 4 --num-workers 4
mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-4-64 --epochs 5 --image-size 64 --batch-size 1 --cache-path kaggle_cache --weights-file 64-t-n-4.pt --world-size 4 --num-workers 4
mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-b-16 --epochs 5 --image-size 64 --batch-size 16 --cache-path kaggle_cache --weights-file 64-t-b-16.pt --world-size 2
time mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-b-16 --epochs 5 --image-size 64 --batch-size 16 --cache-path kaggle_cache --weights-file 64-t-b-16.pt --world-size 2
time mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-4 --epochs 5 --image-size 256 --batch-size 16 --cache-path kaggle_cache --weights-file 64-t-w-4-16.pt --world-size 4 --num-workers 4
time mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-4 --epochs 5 --image-size 256 --batch-size 128 --cache-path kaggle_cache --weights-file 64-t-w-4-128.pt --world-size 4 --num-workers 4
$PYTHON unet.py --hpu --use_lazy_mode --run-name 256-test-workers-1-128 --epochs 5 --image-size 256 --batch-size 128 --cache-path kaggle_cache --weights-file 256-t-w-1-128.pt --world-size 1
time $PYTHON unet.py --hpu --use_lazy_mode --run-name 256-test-workers-1-128 --epochs 5 --image-size 256 --batch-size 128 --cache-path kaggle_cache --weights-file 256-t-w-1-128.pt --world-size 1
time $PYTHON unet.py --hpu --use_lazy_mode --run-name 256-test-workers-2-128 --epochs 5 --image-size 256 --batch-size 128 --cache-path kaggle_cache --weights-file 256-t-w-1-128.pt --num-workers 2
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 256-test-workers-2-128 --epochs 5 --image-size 256 --batch-size 128 --cache-path kaggle_cache --weights-file 256-t-w-4-128.pt --world-size 2 --num-workers 2
mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 256-test-workers-2-16 --epochs 5 --image-size 256 --batch-size 16 --cache-path kaggle_cache --weights-file 256-t-w-2-16.pt --world-size 2 --num-workers 2
time mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 256-test-workers-2-16 --epochs 5 --image-size 256 --batch-size 16 --cache-path kaggle_cache --weights-file 256-t-w-2-16.pt --world-size 2 --num-workers 2
cp -r /lambda_stor/habana/apps/1.5.0/Model-References ~/new_models
time mpirun -n 2 --rank-by core python3 unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-4 --epochs 5 --image-size 256 --batch-size 128 --cache-path kaggle_cache --weights-file 64.pt --world-size 2 --num-workers 2
time mpirun -n 2 --rank-by core python3 unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-4 --epochs 5 --image-size 64 --batch-size 128 --cache-path kaggle_cache --weights-file 64.pt --world-size 2 --num-workers 2
time mpirun -n 2 --rank-by core python3 unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-4 --epochs 5 --image-size 256 --batch-size 128 --cache-path kaggle_256_cache --weights-file 64.pt --world-size 2 --num-workers 2
time mpirun -n 2 --rank-by core python3 unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-4 --epochs 5 --image-size 256 --batch-size 64 --cache-path kaggle_256_cache --weights-file 64.pt --world-size 2 --num-workers 2
top
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2 --num-workers 15 &
cd apps/unet_bench/
ls
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2-15 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2 --num-workers 15 &
git branch
cd ..
ls
cd habana-unet-power/
ls
cd unet_bench/
ls
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2-15 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2 --num-workers 15 &
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2-5 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2 --num-workers 5 &
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2 --num-workers 2 &
vim utils.py
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2-5 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2 --num-workers 5 &
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2 --num-workers 2 &
vim utils.py
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2-5 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2 --num-workers 5 &
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-cards-2-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-c-2.pt --world-size 2 --num-workers 2 &
time mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-b-32 --epochs 5 --image-size 64 --batch-size 32 --cache-path kaggle_cache --weights-file 64-t-b-32.pt --world-size 2
top
cd ..
ls
cd ..
ls
cd ..
ls
cd Model-References/
ls
ctlr
aevard_venv
deactivate
cd ..
python3 -m venv --system-site-packages ~/PT_venv
source ~/PT_venv/bin/activate
ls
which python
python -m pip install apps/habana-unet-power/unet_bench/bruce.txt
python -m pip -r install apps/habana-unet-power/unet_bench/bruce.txt
python -m pip install -r apps/habana-unet-power/unet_bench/bruce.txt wh
pip list
pip freeze
which python
top
cd apps/habana-unet-power/unet_bench/
ls
ls performance/
mkdir performance/pre-bruce-env
mv performance/*.txt performance/pre-bruce-env/
ls performance/
export PYTHON=~/PT_venv/bin/python
export PYTHON=~/PT_venv/bin/python3
top
mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-workers-2 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file 64-t-w-2.pt --world-size 2 --num-workers 2
ls
time mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-b-16 --epochs 5 --image-size 64 --batch-size 16 --cache-path kaggle_cache --weights-file 64-t-b-16.pt --world-size 2
mpirun -n 2 --bind-to core --rank-by core --allow-run-as-root $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name 64-test-b-16 --epochs 5 --image-size 64 --batch-size 16 --cache-path kaggle_cache --weights-file 64-t-b-16.pt --world-size 2
top
cd
ls
cd new_models/
ls
cd PyTorch/
ls
cd examples
ls
cd computer_vision/
cd hello_world/
ls
top
pip freeze
top


source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
cd apps/unet_bench/




Begin OK

Open Terminal 1 from dev machine

```bash
ssh -J wilsonb@homes.cels.anl.gov wilsonb@habana-01.ai.alcf.anl.gov
```

Continue

```bash
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
source /home/aevard/aevard_venv/bin/activate


xxcd /home/aevard/apps/habana-unet-power/unet_bench/
cd DL/BruceRayWilsonAtANL/habana-unet-power/unet_bench/

# This only gets done once of course.
cp -r /home/aevard/apps/unet_bench/data/kaggle_duped_cache/ data/kaggle_duped_cache


cd /home/wilsonb/DL/BruceRayWilsonAtANL/habana-unet-power/unet_bench/scripts

# This only gets done once of course.
chmod 755 build-hl-smi-csv
```

Start the power monitoring script with a specified output file.

```bash
./build-hl-smi-csv post-git-test.txt
```

# Start Terminal 2 from your dev machine

Open a new terminal.

```bash
ssh -J wilsonb@homes.cels.anl.gov wilsonb@habana-01.ai.alcf.anl.gov
```

Continue

```bash
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
source /home/aevard/aevard_venv/bin/activate
cd DL/BruceRayWilsonAtANL/habana-unet-power/unet_bench/
```


```bash
time $PYTHON unet.py --hpu --use_lazy_mode --run-name habana-worker-1-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-1-d.pt
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-2-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-2-d.pt --world-size 2 --num-workers 2
time mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-4-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-4-d.pt --world-size 4 --num-workers 4
```

When the runs have finished switch to terminal 1

<Ctrl+c> out of the power monitoring scripts


```bash
cat performance/load-size-128.txt | pbcopy
xclip
pbcopy
```










ls
mpirun -n 2 --rank-by core $PYTHON hvd_unet.py --hpu --use_lazy_mode --distributed --run-name hvd-test-1 --epochs 5 --image-size 64 --batch-size 64 --cache-path kaggle_cache --weights-file h-t-1.pt --world-size 2 --num-workers 2
vim unet.py
$PYTHON unet.py --hpu --use_lazy_mode --run-name mem-print-1 --epochs 5 --cache-path kaggle_cache --weights-file m-p-1.pt
vim unet.py
$PYTHON unet.py --hpu --use_lazy_mode --run-name mem-print-1 --epochs 5 --cache-path kaggle_cache --weights-file m-p-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name mem-print-2 --epochs 5 --cache-path kaggle_256_cache --image-size 256 --weights-file m-p-1.pt
$PYTHON unet.py --hpu --use_lazy_mode --run-name mem-print-2 --epochs 5 --cache-path kaggle_256_cache --image-size 256 --weights-file m-p-2.pt
top
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
cd apps/habana-unet-power/
git pull
ls
cd unet_bench/
ls
$PYTHON unet.py --hpu --use_lazy_mode --run-name habana-worker-1-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-1-d.pt
time $PYTHON unet.py --hpu --use_lazy_mode --run-name habana-worker-1-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-1-d.pt
ls data
cp -r ../../unet_bench/data/kaggle_duped_cache/ data/kaggle_duped_cache
time $PYTHON unet.py --hpu --use_lazy_mode --run-name habana-worker-1-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-1-d.pt
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-2-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-2-d.pt --world-size 2 --num-workers 2
time mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-4-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-4-d.pt --world-size 4 --num-workers 4
top
time mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-4-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-4-d.pt --world-size 4 --num-workers 4
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-2-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-2-d.pt --world-size 2 --num-workers 2
time mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-4-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-4-d.pt --world-size 4 --num-workers 4
top
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-2-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-2-d.pt --world-size 2 --num-workers 2
time mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-4-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-4-d.pt --world-size 4 --num-workers 4
source ~/aevard_venv/bin/activate
export HABANA_LOGS=~/habana_logs
export PYTHON=/home/aevard/aevard_venv/bin/python
cd apps/unet_bench/
ls
cd scripts/
ls
rm *.txt
ssh ..
ls
cd ..
ls
cd habana-unet-power/
ls
cd unet_bench/
cd scripts/
ls
chown build-hl-smi-csv 755
chmod build-hl-smi-csv 755
chmod 755 build-hl-smi-csv
./build-hl-smi-csv post-git-test.txt
source ~/aevard_venv/bin/activate
export PYTHON=/home/aevard/aevard_venv/bin/python
export HABANA_LOGS=~/habana_logs
top
cd apps/
ls
cd habana-unet-power/
git pull
ls
cd unet_bench/
ls
top
ls
$PYTHON unet.py --hpu --use_lazy_mode --run-name mem-print-1 --epochs 5 --cache-path kaggle_cache --weights-file m-p-1.pt
ls performance/


cd performance/
ls
rm *.txt
ls
git status
ls
mkdir unsorted l
ls
rm unsorted/
rm -r unsorted/
mkdir unsorted-logs
cd ..
ls
$PYTHON unet.py --hpu --use_lazy_mode --run-name mem-print-1 --epochs 5 --cache-path kaggle_cache --weights-file m-p-1.pt
