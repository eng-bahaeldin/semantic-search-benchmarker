# Kubernetes Workload Orchestration & Networking Specification

## 1. Deployment Rollout & Pod Lifecycle Management

### 1.1 RollingUpdateDeployment Strategy Parameters
A `RollingUpdateDeployment` controls how Kubernetes transitions pods from an old `ReplicaSet` to a new `ReplicaSet` without interrupting client traffic. Two primary parameters govern surge capacity and availability floor: `maxSurge` and `maxUnavailable`.

- **`maxUnavailable`**: Specifies the maximum number of pods that can be unavailable during the update process. The value can be an absolute number (e.g., `5`) or a percentage of desired pods (e.g., `10%`). By default, `maxUnavailable` is set to **`25%`**.
- **`maxUnavailable` Constraint**: This value cannot be `0` if `maxSurge` is also set to `0`, as the controller would be deadlocked and unable to terminate or create any pods.
- **`maxSurge`**: Specifies the maximum number of pods that can be created above the desired number of `replicas`. The default value is **`25%`**. When `maxSurge` is `25%` and `maxUnavailable` is `0%`, Kubernetes scales up new pods first and waits for readiness probes before terminating old pods.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-ingress-worker
  namespace: prod-core
spec:
  replicas: 12
  revisionHistoryLimit: 5
  progressDeadlineSeconds: 600
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 0
  selector:
    matchLabels:
      app: payment-ingress
