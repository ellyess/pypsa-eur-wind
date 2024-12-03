# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# This script helps to generate a scenarios.yaml file for PyPSA-Eur.
# You can modify the template to your needs and define all possible combinations of config values that should be considered.

if "snakemake" in globals():
    filename = snakemake.output[0]
else:
    filename = "../config/scenarios.yaml"

import itertools

# Insert your config values that should be altered in the template.
# Change `config_section` and `config_section2` to the actual config sections.
template = """
scenario{scenario_number}:
    wake_effect:
        type: {config_value}
    renewable:
        offwind-near:
            correction_factor: {config_value3}
        offwind-far:
            correction_factor: {config_value3}
        offwind-float:
            correction_factor: {config_value3}
"""

# Define all possible combinations of config values.
# This must define all config values that are used in the template.
config_values = dict(
    config_value=["false", "glaum", "false"], 
    config_value2=[10e3, 2e3, 4e3, 6e3, 8e3, 12e3, 14e3, 16e3, 18e3],
    config_value3=[1.,0.906,0.8855]
    )

combinations = [
    dict(zip(config_values.keys(), values))
    for values in itertools.product(*config_values.values())
]

with open(filename, "w") as f:
    for i, config in enumerate(combinations):
        f.write(template.format(scenario_number=i, **config))
