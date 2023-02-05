python app.py --dataset Earthquake --mode train --timesteps 200 --samplingsteps 200 --batch_size 64 --cuda_id 0 --total_epochs 2000

python app.py --dataset COVID19 --mode train --timesteps 200 --samplingsteps 200 --batch_size 64 --cuda_id 0 --total_epochs 2000

python app.py --dataset Pinwheel --mode train --timesteps 200 --samplingsteps 200 --batch_size 256 --cuda_id 0 --total_epochs 2000 

python app.py --dataset HawkesGMM --mode train --timesteps 200 --samplingsteps 200 --batch_size 256 --cuda_id 0 --total_epochs 2000

python app.py --dataset Mobility --mode train --timesteps 200 --samplingsteps 200 --batch_size 128 --cuda_id 0 --total_epochs 2000 

python app.py --dataset Citybikes --mode train --timesteps 200 --samplingsteps 200 --batch_size 128 --cuda_id 0 --total_epochs 2000 

python app.py --dataset Independent --mode train --timesteps 200 --samplingsteps 200 --batch_size 128 --cuda_id 0 --total_epochs 2000 
