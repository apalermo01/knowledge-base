# working with c++ - python bindings [in progress]

1) created virtual environment, installed invoke via pip

## Make sure invoke is working
(using invoke's documentation https://docs.pyinvoke.org/en/latest/getting-started.html)

2) made a file called tasks.py containing this code snippet:
```python
from invoke import task 

@task
def build(c):
    print("Building!")
```

3) executed the task (in terminal): 
```bash
invoke build
```

## create c/c++ library
(following https://realpython.com/python-bindings-overview/)

4) create a file names `cmult.c` with this code:

```c
float cmult(int int_param, float float_param){
    float return_value = int_param * float_param;
    printf(" cmult: int: %d, float: %.1f, returning: %.1f\n", 
        int_param, float_param, return_value);
    return return_value;
}
```

5) make a new file called `ctypes_test.py`

6) load the library

```python
import ctypes
import pathlib

if __name__ == '__main__':
    # load the shared library into ctypes
    libname =pathlib.Path().absolute() / "libcmult.so"
    c_lib = ctypes.CDLL(libname)
```

7) 