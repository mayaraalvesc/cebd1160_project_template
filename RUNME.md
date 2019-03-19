### To build:

The container can be built using the following command from within this directory:

```
docker build -t gkiar/cebd1160-project .
```

### To run:

The software can be run from within this directory with the following command:

```
docker run -ti -v ${PWD}/data:/data ${PWD}/figures:/figures gkiar/cebd1160-project /data /figures
```
