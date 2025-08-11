from metrics_calculator import CalculateMetrics

calculator = CalculateMetrics(
    data_dir="../dataframes/experiment2.1",
    output_dir="../metrics/experiment2.1",
    verbose=True
)
calculator.run_all()