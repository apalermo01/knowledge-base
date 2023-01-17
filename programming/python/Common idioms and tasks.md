
**add a path to sys**

```python
import os, sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)
```

Context: 
when working in a jupyter notebook, this lets you import modules that are one directory above the current notebook


**make the project an installable package**

- make a file called `setup.py` in the root directory of the project with these contents:

```python
from setuptools import find_packages
from setuptools import setup

setup(
	name='name_of_project',
	version='project_version',
	packages=find_packages(),
)
```

then run `pip install --editable .` in the shell. 