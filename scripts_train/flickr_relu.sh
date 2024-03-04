if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <model> <gpu> <seed>"
  exit 1
fi

# Assign the input arguments to variables
model="$1"
gpu="$2"
seed="$3"
export dataset=flickr

mkdir -p ./log/${dataset}_seed${seed}/
nohup python -u maxk_gnn_dgl.py --dataset ${dataset}  --model ${model} --selfloop \
 --hidden_layers 3 --hidden_dim 256 --nonlinear "relu" \
 --dropout 0.2 --norm --w_lr 0.001 --seed ${seed} \
 --path experiment/${dataset}_seed${seed}/${model}_relu --epochs 400 --gpu ${gpu} \
 > ./log/${dataset}_seed${seed}/${model}_relu.txt &
