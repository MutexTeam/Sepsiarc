STAGE 2 FINAL EXPORTS

Files:
- Final_Test_Results.csv: final saved-model test metrics
- Test_Patient_Predictions.csv: actual labels, predicted probabilities,
  predicted classes, and frozen threshold
- Confusion_Matrix.csv: test-set classification counts
- Selected_65_Features.csv: ordered model input features
- Model_Metadata.json: saved model metadata
- Final_Stage2_Summary.json: model configuration, metrics, and limitations

Model threshold: 0.29
Target: Risk_50pct

Important:
- This export does not retrain the model.
- The test patients were evaluated previously.
- Risk_50pct is a rule-derived proxy, not a clinically validated outcome.
