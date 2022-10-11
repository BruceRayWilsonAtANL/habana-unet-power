#!/bin/bash
touch $1

for i in {1..3600}
do
    hl-smi -Q timestamp,module_id,utilization.aip,power.draw,memory.used -f csv 
    sleep 1
done
