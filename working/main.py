import os
import fnmatch
import importlib

def exec_module(dirpath, file):
    with open(os.path.join(dirpath, file), 'r') as f:
        code = compile(f.read(), file, 'exec')
        exec(code, globals())


def global_import(dirpath, file):
    mod_name = file.split('.')[0]
    prefix = '.'.join(dirpath.split('/')[1:])
    full_name = '.'.join([prefix, mod_name]) if prefix else mod_name
    _tmp = importlib.import_module(full_name)
    for var in dir(_tmp):
        if var[:2] != '__':
            globals()[var] = getattr(_tmp, var)

# exec_module('working', 'imports.py')
global_import('.', 'imports.py')

for (dirpath, dirnames, filenames) in os.walk('.'):
    if dirpath.endswith('checkpoints'):
        continue
    for file in filenames:
        if fnmatch.fnmatch(file, '*.py'):
            if file not in ('main.py', 'france_hosp.py'):
                # exec_module(dirpath, file)
                global_import(dirpath, file)
