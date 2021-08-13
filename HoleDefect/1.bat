start /max cmd /k "conda activate py36_tf1 & python run.py --ep 250 --bs 256" 
TIMEOUT /T 6900
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 300 --bs 128 --xl True"
TIMEOUT /T 6900
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 350 --bs 256"
TIMEOUT /T 6900
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 350 --bs 256 --xl True"
TIMEOUT /T 6900
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 400 --bs 128" 
TIMEOUT /T 6900
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 450 --bs 256 --xl True"