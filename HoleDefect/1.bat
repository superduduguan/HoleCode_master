start cmd /k "conda activate py36_tf1 & python run.py " 
TIMEOUT /T 500
start cmd /k "conda activate py36_tf1 & python run.py  --ep 90"
TIMEOUT /T 500
start cmd /k "conda activate py36_tf1 & python run.py  --ep 130"
TIMEOUT /T 500
start cmd /k "conda activate py36_tf1 & python run.py  --ep 115"
TIMEOUT /T 500
start cmd /k "conda activate py36_tf1 & python run.py  --ep 115 --bs 64"
TIMEOUT /T 500
start cmd /k "conda activate py36_tf1 & python run.py  --ep 90 --bs 64"

