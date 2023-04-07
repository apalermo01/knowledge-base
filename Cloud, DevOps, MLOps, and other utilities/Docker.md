### key terms
- **dockerfile** - a script of instructions describing how to build a docker image
- **image** - read-only file with everything that an application needs to run. Represents an application and its environment at a specific point in time. 
- **Container** - the environment in which an image runs. 

### Build an image

```bash

docker build --tag <image name> .
```

### Run an image

```bash
docker run -d --name <name to give running container> <image name> -f /dev/null
```
- `-f /dev/null` keeps the container from exiting

### enter into container through bash
```bash
docker exec -it <name of running container> bash
```


# Running a docker container with persistent storage

## Storage options


**Bind mounts**
- mounts a fiile or directory on the host machine to the container
- relies on the structure of the host's filesystem
- the documentation suggests sticking to volumes

**Volume**
- completely managed by docker
- independent of system configuration


## Setting up persistent storage


- Docker suggests using the `--mount` flag when executing docker run
- arguments after mount are comma separated key-value pairs
	- `type` = type of mount, either `bind`, `volume`, or `tmpfs`
	- `source` = changes depending on the mount type used
		- `type=bind`: file path or directory in the host machine
			- NOTE: this directory must already exist on the host machine
		- `type=volume`: for named volumes, the name of the volume. For anonymous volumes, this may be ommitted
	- `destination` = file path / directory in container

**example of running a docker container with a bind mount**
```bash
docker run \
	-d \
	--name <image name> \
	--mount type=bind,\
			source=<path on host machine>,\
			destination=<path in container> \
	<name of container> \
	tail -f /dev/null
```


# References
- https://docs.docker.com/storage/volumes/
- https://docs.docker.com/storage/bind-mounts/