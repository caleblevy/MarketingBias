from pathlib import Path
import shutil

import pslpython
from pslpython.model import Model as _Model
from pslpython.rule import Rule as _Rule
from pslpython.predicate import Predicate as _Predicate
from pslpython.partition import Partition

class Model(_Model):

    def __init__(self, name, predicate_dir, output_dir):
        self._predicate_dir = predicate_dir
        self._output_dir = output_dir
        self._java_path = shutil.which('java')
        if (self._java_path is None):
            raise ModelError("Could not locate a java runtime (via https://docs.python.org/dev/library/shutil.html#shutil.which). Make sure that java exists within your path.")
        self._rules = []
        self._predicates = {}
        self._name = name

    def add_predicate(self, *args, **kwargs):
        super().add_predicate(Predicate(*args, **kwargs))

    def _load_data(self, eval_or_learn):
        for predicate in self.get_predicates():
            predicate.clear_data()
            observations_file = self._predicate_dir / eval_or_learn / "observations" / f"{predicate._raw_name}.txt"
            targets_file = self._predicate_dir / eval_or_learn / "targets" / f"{predicate._raw_name}.txt"
            truth_file = self._predicate_dir / eval_or_learn / "truth" / f"{predicate._raw_name}.txt"
            if predicate.closed():
                predicate.add_data_file(partition.OBSERVATIONS, observations_file)
            else:
                if observations_file.exists():
                    predicate.add_data_file(partition.OBSERVATIONS, observations_file)
                predicate.add_data_file(partition.TARGETS, target_file)
                if truth_file.exists():
                    predicate.add_data_file(partition.TRUTH, truth_file)

    def learn(self, method = '', additional_cli_options = [], psl_config = {}, jvm_options = [], temp_dir=None, cleanup_temp=True):
        self._load_data("eval")



class Predicate(_Predicate):

    def __init__(self, raw_name: str, closed: bool, size: int = None, arg_types = None):
        super().__init__(raw_name, closed, size, arg_types)
        self._raw_name = raw_name


if __name__ == '__main__':
    a = Model("priors", Path("datasets/modcloth/predicates/4/"), Path("results"))
    