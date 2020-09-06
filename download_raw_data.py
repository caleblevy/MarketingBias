#!/usr/bin/env python3
from pathlib import Path

import sh

sh.cd(Path(__file__).parent.absolute())
sh.mkdir('-p', 'datasets/modcloth/raw', 'datasets/electronics/raw')
print(sh.aws('s3', 'cp', 's3://seshlabucsc/df_modcloth.csv', './datasets/modcloth/raw'))
print(sh.aws('s3', 'cp', 's3://seshlabucsc/df_electronics.csv', './datasets/electronics/raw'))
