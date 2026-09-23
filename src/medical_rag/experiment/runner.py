"""Experiment runner.

Will drive execution of one or more experiment conditions over a dataset,
wiring together retrieval, modules, and generation, and collecting outputs
for later metrics/analysis.

TODO: class ExperimentRunner: takes config, retriever, llm client.
TODO: ExperimentRunner.run_condition(condition, dataset) -> list[dict] results.
TODO: ExperimentRunner.run_all(conditions, dataset) -> dict of results per condition.
"""
