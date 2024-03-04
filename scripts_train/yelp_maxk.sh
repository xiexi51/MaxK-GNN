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
export dataset=yelp

mkdir -p ./log/${dataset}_seed${seed}/
nohup python -u maxk_gnn_dgl.py --dataset ${dataset} --model ${model} \
 --hidden_layers 4 --hidden_dim 384 --nonlinear "maxk" --maxk ${k} \
 --dropout 0.1 --norm --w_lr 0.001 --seed ${seed} \
 --path experiment/${dataset}_seed${seed}/${model}_max${k} --epochs 3000 --gpu ${gpu} \
 > ./log/${dataset}_seed${seed}/${model}_max${k}.txt &