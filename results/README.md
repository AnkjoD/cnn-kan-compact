# Reported results

`paper_reported_metrics.csv` records the final values recovered from the saved
experiment outputs supplied with the FAIR 2026 project. It is a provenance
table, not a substitute for rerunning evaluation.

Important caveats:

- FastKAN seed 123 has conflicting transient notebook output. The table uses
  the later saved `metrics.json` values that match the manuscript. Verify the
  checkpoint hash before the camera-ready submission.
- The Master Evaluation notebook reconstructed a legacy split because the
  original frozen manifest was unavailable at runtime. Its diagnostic outputs
  must not replace the primary frozen-split results.
- DenseNet121 did not finish the controlled frozen-split queue and is therefore
  excluded from this table.
- Kermany is used only for zero-shot external evaluation in the experiments
  listed here. Target-aware DANN experiments require a separate protocol and
  are not mixed with these results.
- Linear and MLP seeds 123/2026 and FastKAN no-crop seeds 42/123/2026 are
  transcribed from completed Colab output. The ZIP with per-image predictions
  and JSON metrics was not available when preparing this release. These rows
  carry `notebook_log_pending_archive_check` and should be reconciled before
  tagging a paper release.
- The older no-crop result (Kermany AUC 0.8555) belongs to a diagnostic
  checkpoint with incomplete original split provenance; it is not part of the
  three-seed no-crop rows.
