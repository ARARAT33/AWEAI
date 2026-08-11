# Scheduling

Job scheduling manages the allocation of cluster resources to training and inference jobs.

## Usage

```bash
# Submit job
aweai cluster submit my_job --nodes 4 --gpus 8

# View queue
aweai cluster queue

# Cancel job
aweai cluster cancel JOB_ID
```

```python
from aweai.cluster.scheduling import JobScheduler

scheduler = JobScheduler()
job_id = scheduler.submit("my_job", nodes=4, gpus=8)
scheduler.cancel(job_id)
```

## Related Pages

- [Manager](Manager.md) — Cluster manager
- [Deploy](Deploy.md) — Deployment
