import os
os.chdir('/content')
# confirma os arquivos
!ls *.csv

!python fincontrol_etl.py
