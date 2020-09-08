import asyncio
import csv
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile  # TODO: Remove the need for this import

import pandas

import pslpython
from pslpython.model import Model as _Model, ModelError
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
        for name, predicate in self.get_predicates().items():
            predicate.clear_data()
            observation_file = self._predicate_dir / eval_or_learn / "observations" / f"{name}.txt"
            target_file = self._predicate_dir / eval_or_learn / "targets" / f"{name}.txt"
            truth_file = self._predicate_dir / eval_or_learn / "truth" / f"{name}.txt"
            if predicate.closed():
                predicate.add_data_file(Partition.OBSERVATIONS, observation_file)
            else:
                if observation_file.exists():
                    predicate.add_data_file(Partition.OBSERVATIONS, observation_file)
                predicate.add_data_file(Partition.TARGETS, target_file)
                if truth_file.exists():
                    predicate.add_data_file(Partition.TRUTH, truth_file)

    def infer(self, method='', additional_cli_options=None, psl_config=None, jvm_options=None, temp_dir=None, cleanup_temp=True):
        """
        Run inference on this model.

        Args:
            method: The inference method to use.
            additional_cli_options: Additional options to pass direcly to the CLI.
                                   Here you would do things like select a database backend.
            psl_config: Configuration passed directly to the PSL core code.
                        https://github.com/eriq-augustine/psl/wiki/Configuration-Options
            jvm_options: Options passed to the JVM.
                         Most commonly '-Xmx' and '-Xms'.
            logger: An optional logger to send the output of PSL to.
                    If not specified (None), then a default INFO logger is used.
                    If False, only fatal PSL output will be passed on.
                    If no logging levels are sent via psl_config, PSL's logging level will be set
                    to match this logger's level.
            temp_dir: Where to write PSL files to for calling the CLI.
                      Defaults to Model.TEMP_DIR_SUBDIR inside the system's temp directory (tempfile.gettempdir()).
            cleanup_temp: Remove the files in temp_dir after running.

        Returns:
            The inferred values as a map to dataframe.
            {predicate: frame, ...}
            The frame will have columns names that match the index of the argument and 'truth'.
        """
        self._load_data("eval")
        self._output_dir.mkdir(exist_ok=True, parents=True)
        if additional_cli_options is None:
            additional_cli_options = []
        if psl_config is None:
            psl_config = {}
        if jvm_options is None:
            jvm_options = []

        # Start original
        temp_dir, data_file_path, rules_file_path = self._prep_run(temp_dir)

        cli_options = []

        cli_options.append('--infer')
        if (method != ''):
            cli_options.append(method)

        inferred_dir = os.path.join(temp_dir, Model.CLI_INFERRED_OUTPUT_DIR)
        cli_options.append('--output')
        cli_options.append(inferred_dir)

        cli_options += additional_cli_options

        self._run_psl(data_file_path, rules_file_path, cli_options, psl_config, jvm_options)
        results = self._collect_inference_results(inferred_dir)

        if (cleanup_temp):
            self._cleanup_temp(temp_dir)

        inferred_predicate_dir = self._output_dir / "inferred_predicates"
        inferred_predicate_dir.mkdir(exist_ok=True)
        for predicate in self.get_predicates().values():
            if (predicate.closed()):
                continue
            out_path = inferred_predicate_dir / (predicate.name()+'.txt')
            results[predicate].to_csv(out_path, sep = "\t", header = False, index = False)

        return results


    def _prep_run(self, temp_dir=None):
        """
        Run weight learning on this model.
        The new weights will be applied to this model.

        Args:
            logger: An optional logger to send the output of PSL to.
                    If not specified (None), then a default INFO logger is used.
                    If False, only fatal PSL output will be passed on.
            temp_dir: Where to write PSL files to for calling the CLI.
                      Defaults to Model.TEMP_DIR_SUBDIR inside the system's temp directory (tempfile.gettempdir()).

        Returns:
            A prepped logger, a usable temp_dir, the path to the CLI data file, and the path to the CLI rules file.
        """

        if (len(self._rules) == 0):
            raise ModelError("No rules specified to the model.")

        if (temp_dir is None):
            temp_dir = os.path.join(tempfile.gettempdir(), Model.TEMP_DIR_SUBDIR)
        # TODO: Remove the need for this
        temp_dir = os.path.join(temp_dir, self._name)
        os.makedirs(temp_dir, exist_ok=True)

        data_file_path = self._write_data(temp_dir)
        rules_file_path = self._write_rules(temp_dir)

        return temp_dir, data_file_path, rules_file_path


    def _run_psl(self, data_file_path, rules_file_path, cli_options, psl_config, jvm_options):
        command = [
            self._java_path
        ]

        for option in jvm_options:
            command.append(str(option))

        command += [
            '-jar',
            Model.CLI_JAR_PATH,
            '--model',
            rules_file_path,
            '--data',
            data_file_path,
        ]

        for option in cli_options:
            command.append(str(option))

        for (key, value) in psl_config.items():
            command.append('-D')
            command.append("%s=%s" % (key, value))

        "Running: `%s`." % (pslpython.util.shell_join(command))
        exit_status = execute(command, self._output_dir)

        if (exit_status != 0):
            raise ModelError("PSL returned a non-zero exit status: %d." % (exit_status))

    def _collect_inference_results(self, inferred_dir):
        """
        Get the inferred data written by PSL.

        Returns:
            A dict with the keys being the predicate and the value being a dataframe with the data.
            {predicate: frame, ...}
            The frame will have columns names that match the index of the argument and 'truth'.
        """

        results = {}

        for dirent in os.listdir(inferred_dir):
            path = os.path.join(inferred_dir, dirent)

            if (not os.path.isfile(path)):
                continue

            predicate_name = os.path.splitext(dirent)[0]
            predicate = None
            for possible_predicate in self._predicates.values():
                if (possible_predicate.name().upper() == predicate_name.upper()):
                    predicate = possible_predicate
                    break

            if (predicate is None):
                raise ModelError("Unable to find predicate that matches name if inferred data file. Predicate name: '%s'. Inferred file path: '%s'." % (predicate_name, path))

            columns = list(range(len(predicate))) + [Model.TRUTH_COLUMN_NAME]
            data = pandas.read_csv(path, delimiter = Model.CLI_DELIM, names = columns, header = None, skiprows = None, quoting = csv.QUOTE_NONE)

            # Clean up and convert types.
            for i in range(len(data.columns) - 1):
                if (predicate.types()[i] in Predicate.INT_TYPES):
                    data[data.columns[i]] = data[data.columns[i]].apply(lambda val: int(val))
                elif (predicate.types()[i] in Predicate.FLOAT_TYPES):
                    data[data.columns[i]] = data[data.columns[i]].apply(lambda val: float(val))

            data[Model.TRUTH_COLUMN_NAME] = pandas.to_numeric(data[Model.TRUTH_COLUMN_NAME])

            results[predicate] = data

        return results


class Predicate(_Predicate):

    def __init__(self, raw_name: str, closed: bool, size: int = None, arg_types = None):
        super().__init__(raw_name, closed, size, arg_types)
        self._name = raw_name


class RunOutput():
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def _read_stream(stream, callback):
    while True:
        line = await stream.readline()
        if line:
            callback(line)
        else:
            break

async def _stream_subprocess(cmd, stdin=None, quiet=False, echo=False) -> RunOutput:
    if os.name == 'nt':
        platform_settings = {'env': os.environ}
    else:
        platform_settings = {'executable': '/bin/bash'}

    if echo:
        print(cmd)

    p = await asyncio.create_subprocess_exec(*cmd,
                                              stdin=stdin,
                                              stdout=asyncio.subprocess.PIPE,
                                              stderr=asyncio.subprocess.PIPE)
    out = []
    err = []

    def tee(line, sink, pipe):
        line = line.decode('utf-8').rstrip()
        sink.append(line)
        if not quiet:
            print(line, file=pipe)

    await asyncio.wait([
        _read_stream(p.stdout, lambda l: tee(l, out, sys.stdout)),
        _read_stream(p.stderr, lambda l: tee(l, err, sys.stderr)),
    ])

    return RunOutput(await p.wait(), out, err)


def run(cmd, stdin=None, quiet=False, echo=False) -> RunOutput:
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        _stream_subprocess(cmd, stdin=stdin, quiet=quiet, echo=echo)
    )

    return result


def execute(command, output_dir):
    if isinstance(command, str):
        command = shlex.split(command)
    print("Running: `%s`." % (pslpython.util.shell_join(command)))
    proc = run(command)
    with open(output_dir / "stdout.log", 'w') as f:
        f.write('\n'.join(proc.stdout))
    with open(output_dir / "stderr.log", 'w') as g:
        g.write('\n'.join(proc.stderr))
    return proc.returncode
    