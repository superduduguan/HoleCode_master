TIMEOUT /T 3600
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 450 --bs 256" 
TIMEOUT /T 12900
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 450 --bs 128"
TIMEOUT /T 12900
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 550 --bs 256"
TIMEOUT /T 12900
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 650 --bs 256 --xl True"
