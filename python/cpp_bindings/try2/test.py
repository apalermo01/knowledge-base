from build.module_name import some_python_fn
import build

if __name__ == '__main__':
    print(dir(build.module_name))
    print(build.module_name.PySomeClass(5))
    m = build.module_name.PySomeClass(5)
    print(m.multiply(25))
    print(m.multiply_list([1, 2, 3]))
    print(some_python_fn(1, 2))