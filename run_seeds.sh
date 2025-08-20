# run_seeds.sh
set -e
DATASET="CIFAR-100"
epochs=200
Routing_No=3
for SEED in 101 202 303; do
  papermill main.ipynb "logs/${DATASET}/seed-${SEED}.ipynb" \
    -p args_dict '{
        "dataset": "'$dataset'", 
        "epochs": '$epochs',
        "Routing_N": "'$Routing_No'",
        "seed": "'$SEED'",
        "data_path": "/Dataset/Marine_tree" # Dataset path on 
    }'
done
