start /max cmd /k "conda activate py36_tf1 & python run.py --ep 80 --bs 64" 
TIMEOUT /T 750
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 90 --bs 64"
TIMEOUT /T 750
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 100 --bs 64"
TIMEOUT /T 750
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 110" 
TIMEOUT /T 750
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 90"
TIMEOUT /T 750
start /max cmd /k "conda activate py36_tf1 & python run.py --ep 100"
TIMEOUT /T 750
start /max cmd /k "conda activate py36_tf1 & python run.py"


