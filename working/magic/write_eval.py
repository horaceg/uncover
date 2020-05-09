from IPython.core.magic import Magics, magics_class, cell_magic


@magics_class
class WriteEval(Magics):

    @cell_magic
    def write_eval(self, line, cell):
        self.shell.run_cell_magic('writefile', line, cell)
        exec(cell, self.shell.user_global_ns)
        return 


def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    # You can register the class itself without instantiating it.  IPython will
    # call the default constructor on it.
    ipython.register_magics(WriteEval)
