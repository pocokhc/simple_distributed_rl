REM Copy the srl folder into this directory and run it.
del /S *.pyc
tar -czvf submission.tar.gz main.py model.py parameter.dat srl
