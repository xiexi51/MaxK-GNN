# Check if the correct number of arguments are provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <k> <seed> <gpu> <model>"
  exit 1
fi

# Assign the input arguments to variables
k="$1"
seed="$2"
gpu="$3"
model="$4"
export dataset=ogbn-products

if [ "$model" == "sage" ]; then
    selfloop=""
else
    selfloop=--selfloop
fi

mkdir -p ./log/${dataset}_seed${seed}/
nohup python -u maxk_gnn_dgl.py --dataset ${dataset} --model ${model} ${selfloop} \
 --hidden_layers 3 --hidden_dim 256 --nonlinear "maxk" --maxk ${k} \
 --dropout 0.5 --norm --w_lr 0.003 --seed ${seed} \
 --path experiment/${dataset}_seed${seed}/${model}_max${k} --epochs 500 --gpu ${gpu} <<< "y" \
 > ./log/${dataset}_seed${seed}/${model}_max${k}.txt &
