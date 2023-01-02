
## Key terms

**minikube** - local version of k8s , makes it easy to learn and develop apps

**kubectl** - cli tools for k8s

**kubernetes pod** - smallest deployable units of computing that you can create and manage in Kubernetes. It's a group of one or more containers with shared network and storage resources, along with a specification of how to run the containers. People frequenctly use one container per pod, treating a pod like a wrapper around a single container.

**Deployment** - checks on the health of a pod and restarts the pod's container if it terminates. Recommended way to manage the creation and scaling of pods

**Service** - An abstract way to expose an application running on a set of pods as a netword service.

**Control pane** - 

## Install

to install minikube, follow the instructions here
https://minikube.sigs.k8s.io/docs/start/


start minikube: `minikube start`
get cluster details: `kubectl cluster-info`
get nodes in a cluster: `kubectl get nodes`

kubectl pattern: `kubectl action resource`

## Deploying an app
create a deployment: `kubectl create deployment <deployment name> --image=<docker image>`
other flags for create deployment:
	- ` -- <command>` creates a deployment with a specific command (e.g. `... --image=<image name> -- <command>`)
	-  `--replicas=<int>` runs the image with the specified number of replicas
	- `--port=<int>` exposes the specified port


## References
- tutorials found at https://kubernetes.io
