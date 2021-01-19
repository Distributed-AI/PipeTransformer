# Text Classification (GLUE SST-2)

Large scale experiments:
```
nohup sh run_tc_pipetransformer.sh 8 2 0 192.168.11.2 22222 1 "ib0" > ./PipeTransformer-TC0.log 2>&1 &
nohup sh run_tc_pipetransformer.sh 8 2 1 192.168.11.2 22222 1 "ib0" > ./PipeTransformer-TC1.log 2>&1 &
```
Debugging:
```
nohup sh run_tc_pipetransformer.sh 4 1 0 192.168.1.73 11111 0 "wlx9cefd5fb3821" > ./PipeTransformer-TC.log 2>&1 &
```

kill all processes
```
kill $(ps aux | grep "main_tc.py" | grep -v grep | awk '{print $2}')
```