
**To build an image**
```<bash>
docker build --tag <name of image> .
```

**To run an image**
```<bash>
docker run -d <name of image>
```
(-d tells docker to run in detached mode)

**To bash into a docker container**

First, start a container that persists in the backgroud
```<bash>
docker run -d <name of image> tail -f /dev/null
```

now, bash into the container
```<bash>
docker exec -ti <container name> bash
```
where `container name` is the item under the 'NAMES' column when runningn `docker ps`

**To see a list of docker images**
```<bash>
docker images
```

**To see a list of running containers**
```<bash>
docker ps
```



## References
- https://stackoverflow.com/questions/63483902/docker-run-bin-bash