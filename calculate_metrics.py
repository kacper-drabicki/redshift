from metrics_calculator import CalculateMetrics

evaluator = CalculateMetrics(
    data_dir="../dataframes/experiment2.1",
    output_dir="../metrics/experiment2.1",
    verbose=True
)
evaluator.run_all()