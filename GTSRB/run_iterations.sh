#!/bin/bash
for i in {1..10}
do
    perf stat --append --field-separator=, -o output_10_1.csv -e instructions,branches,branch-misses,page-faults,L1-icache-load-misses,LLC-load-misses,LLC-store-misses,iTLB-load-misses,dTLB-load-misses python3 evaluate_model_clean.py final_model_clean.h5
done

for i in {1..10}
do
    perf stat --append --field-separator=, -o output_10_2.csv -e instructions,branches,branch-misses,page-faults,L1-icache-load-misses,LLC-load-misses,LLC-store-misses,iTLB-load-misses,dTLB-load-misses python3 evaluate_model_adversarial.py final_model_clean.h5
done

# for i in {1..10}
# do
#     perf stat --append --field-separator=, -o output_10_3.csv -e instructions,branches,branch-misses,page-faults,L1-icache-load-misses,LLC-load-misses,LLC-store-misses,iTLB-load-misses,dTLB-load-misses python3 evaluate_model_clean.py final_model_adv.h5
# done

# for i in {1..10}
# do
#     perf stat --append --field-separator=, -o output_10_4.csv -e instructions,branches,branch-misses,page-faults,L1-icache-load-misses,LLC-load-misses,LLC-store-misses,iTLB-load-misses,dTLB-load-misses python3 evaluate_model_adversarial.py final_model_adv.h5
# done