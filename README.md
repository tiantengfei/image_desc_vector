# image_desc_vector
> A distributed trainer program implements between-graph replication and synchronization training.

### On the ps0 server, type: 
    $:python image_distributed_train.py 
    --ps_hosts=www.ps0.com:2222 
    --worker_hosts=www.worker0.com:2222,www.woker1.com:2222
    --job_name=ps --task_id=0

###On the woker0 server, type:
    $:python image_distributed_train.py \
    --ps_hosts=www.ps0.com:2222 \
    --worker_hosts=www.worker0.com:2222,www.woker1.com:2222 \
    --job_name=woker --task_id=0


###On the worker1 server, type:
    $:python image_distributed_train.py 
    --ps_hosts=www.ps0.com:2222 \
    --worker_hosts=www.worker0.com:2222,www.woker1.co:2222 \
    --job_name=worker --task_id=1
