"""CLI script to run the full experiment.

Will run the experiment pipeline over the full MedQA evaluation set and all
configured conditions, compute metrics, and generate the analysis report.

TODO: parse CLI args (config path, output dir).
TODO: main() -> loads config, builds ExperimentRunner, runs all conditions,
      computes metrics, generates report.
"""
