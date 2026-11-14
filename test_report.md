# Test report

The tests below were run on 13 November 2026, using source commit `bedf68f`.
The automated result was 85 passed, 0 failed, 0 errors and 0 skipped.
The run took 7.81 seconds. Its recorded start time was
`2026-11-13T09:05:28.337034+05:30`.

The final checks were repeated on 14 November 2026. All 85 tests passed again,
along with the style checks, DVC freshness check and candidate model validation.

The command used was:

```bash
python -m pytest tests -q --junitxml=reports/tests/final.xml
```

The XML output is kept locally in `reports/tests/final.xml`. Below I have given
the counts and a brief overview of the application checks. The full executed
case list is at the end of this file.

| Test file | Passed | Failed | Errors | Skipped |
|---|---:|---:|---:|---:|
| [tests/test_api.py](tests/test_api.py) | 22 | 0 | 0 | 0 |
| [tests/test_baseline.py](tests/test_baseline.py) | 18 | 0 | 0 | 0 |
| [tests/test_frontend.py](tests/test_frontend.py) | 6 | 0 | 0 | 0 |
| [tests/test_monitoring.py](tests/test_monitoring.py) | 9 | 0 | 0 | 0 |
| [tests/test_operations.py](tests/test_operations.py) | 21 | 0 | 0 | 0 |
| [tests/test_pipeline.py](tests/test_pipeline.py) | 9 | 0 | 0 | 0 |
| Total | 85 | 0 | 0 | 0 |

There was one AnyIO/Starlette deprecation warning from a dependency. The code
style, formatting and installed dependency checks also passed in the final review.

## Application checks

These checks used the real dataset, running Docker services and browser. They
are separate from the 85 automated cases above. The eight points below follow
the acceptance criteria in section 4 of [REPORT.pdf](REPORT.pdf).

1. Data preparation: Passed. The sampling tests checked repeatability and separate
   splits. A fresh setup also regenerated the labelled batch with the same DVC
   checksum: `804a763ce3a9d1bfbfa9a37e352bca0c`. Ray recorded its batch count,
   time and throughput, as given below.
2. Saved model: Passed. The model loaded in a separate process, and the fresh
   setup reproduced the held-out metrics in a new MLflow store. The tests also
   checked Git lineage and restoring the saved source snapshot.
3. API: Passed. Valid requests returned a prediction with the pinned model
   identity. Invalid fields and model-service failures returned the specified errors.
4. Frontend: Passed. Blank forms, examples, repeated submissions and error recovery
   worked. The real browser checks covered desktop and mobile sizes, with no
   horizontal overflow or browser exceptions.
5. Pipeline and jobs: Passed. Cached job `3020ce3864774000972a18314e4c723d`
   reused all three stages. Failed jobs retained their errors. A stopped worker
   caused an alert and HTTP 503 for new jobs.
6. Deployment and rollback: Passed. The deliberately bad image in job
   `8cd35daa78e448668e079b6241d1ad0f` failed, and the previous model, image and
   settings were restored. That job stayed failed in the history. The API,
   frontend probe and Prometheus also worked with verified HTTPS. Rollback
   restored the previous HTTP settings exactly.
7. Monitoring: Passed. Shifted amounts produced PSI of about 9.318 and a firing
   alert after about 18.4 seconds. Normal traffic cleared it. Stopping the
   frontend produced an alert after about 15.5 seconds while the API stayed
   ready; restarting the frontend cleared it. Grafana loaded its saved dashboard.
8. Labelled maintenance: Passed. Job `e6d826f071394a329bff8ddd7d2e20db` reused
   preparation and ran training and evaluation. It added 20,000 rows, including
   75 fraud cases, to make 370,000 training rows. The validation and test file
   hashes stayed unchanged. Model 19's live reference stayed at 350,000 rows
   while model 20 was only a candidate. It changed to model 20's training parent
   only after release.

The main local release after these checks was model 20, with model 19 kept for
rollback. Model 20's comparison run was `b174e6cc36c44b4a851601a8288815ce`.
The run and model numbers in a fresh store will be different.

## Replay and fresh setup

The recorded Ray preparation processed 6,362,620 source rows in 26 batches using
two workers. The full preparation took 11.55 seconds, or about 550,759 rows per
second. This includes Ray startup, reading the CSV, making the splits and writing
Parquet files. The numbers are recorded in `data/prepared/preparation.json`.

The recorded replay used 2,000 rows and four concurrent clients. It measured
235.25 requests per second and 23.66 ms p95 client latency. There were only two
fraud labels in this replay, so its quality results are only a small example.
No minimum F1 or throughput was required. For the drift check, the amounts were
changed, so the old labels were not used to calculate quality.

The fresh setup check used a separate clone of `bedf68f`, with the deferred
monitoring files supplied locally. It regenerated the batch, ran Docker training
in a new MLflow store, started all six services and served predictions. All three
Prometheus targets were up and the Grafana dashboard was available. Only that
check's temporary services and volumes were removed afterwards.

These results show that the application acceptance checks passed locally.
The monitoring configuration and replay summaries are included with the project.
The video link can be found in the README.

## Executed automated cases

### tests/test_api.py

- `test_prediction_forwards_named_float_fields_and_returns_pinned_identity` — passed.
- `test_invalid_transaction_never_reaches_model[patch0]` — passed.
- `test_invalid_transaction_never_reaches_model[patch1]` — passed.
- `test_invalid_transaction_never_reaches_model[patch2]` — passed.
- `test_invalid_transaction_never_reaches_model[patch3]` — passed.
- `test_invalid_transaction_never_reaches_model[patch4]` — passed.
- `test_invalid_transaction_never_reaches_model[patch5]` — passed.
- `test_missing_field_is_rejected` — passed.
- `test_nonfinite_json_is_rejected_with_serializable_error[NaN]` — passed.
- `test_nonfinite_json_is_rejected_with_serializable_error[Infinity]` — passed.
- `test_invalid_model_output_cannot_become_a_prediction[reply0]` — passed.
- `test_invalid_model_output_cannot_become_a_prediction[reply1]` — passed.
- `test_invalid_model_output_cannot_become_a_prediction[reply2]` — passed.
- `test_invalid_model_output_cannot_become_a_prediction[reply3]` — passed.
- `test_invalid_model_output_cannot_become_a_prediction[reply4]` — passed.
- `test_invalid_model_output_cannot_become_a_prediction[reply5]` — passed.
- `test_upstream_failure_is_explicit[ConnectError-503]` — passed.
- `test_upstream_failure_is_explicit[ReadTimeout-504]` — passed.
- `test_readiness_and_liveness_are_separate` — passed.
- `test_examples_obey_the_api_contract` — passed.
- `test_unavailable_pipeline_evidence_is_not_replaced_with_fake_values` — passed.
- `test_model_version_alias_is_rejected_at_startup` — passed.

### tests/test_baseline.py

- `test_sampling_is_reproducible_disjoint_and_stratified` — passed.
- `test_sampling_does_not_take_only_first_rows` — passed.
- `test_insufficient_fraud_fails_before_training` — passed.
- `test_invalid_split_fractions_are_rejected` — passed.
- `test_weight_uses_the_supplied_training_labels` — passed.
- `test_invalid_numeric_inputs_rejected[-1.0]` — passed.
- `test_invalid_numeric_inputs_rejected[inf]` — passed.
- `test_invalid_numeric_inputs_rejected[nan]` — passed.
- `test_unknown_transaction_type_rejected` — passed.
- `test_strings_and_booleans_are_not_numeric_inputs[not-a-number]` — passed.
- `test_strings_and_booleans_are_not_numeric_inputs[True]` — passed.
- `test_missing_column_rejected` — passed.
- `test_target_cannot_be_passed_to_inference` — passed.
- `test_derived_feature_does_not_mutate_input` — passed.
- `test_csv_rejects_nonbinary_target` — passed.
- `test_csv_excludes_identifiers_and_post_transaction_columns` — passed.
- `test_model_serialization_keeps_preprocessing_and_predictions` — passed.
- `test_average_precision_uses_scores_not_thresholded_labels` — passed.

### tests/test_frontend.py

- `test_blank_form_does_not_submit_guessed_values` — passed.
- `test_example_populates_form_and_one_submit_produces_result` — passed.
- `test_failed_prediction_clears_previous_result` — passed.
- `test_operations_screen_displays_recorded_evidence` — passed.
- `test_missing_backend_configuration_is_visible` — passed.
- `test_pipeline_outputs_being_rebuilt_do_not_hide_operations_controls` — passed.

### tests/test_monitoring.py

- `test_psi_is_zero_for_the_reference_and_detects_a_large_shift` — passed.
- `test_invalid_amounts_do_not_create_a_drift_measurement[values0]` — passed.
- `test_invalid_amounts_do_not_create_a_drift_measurement[values1]` — passed.
- `test_invalid_amounts_do_not_create_a_drift_measurement[values2]` — passed.
- `test_invalid_amounts_do_not_create_a_drift_measurement[values3]` — passed.
- `test_malformed_reference_is_rejected` — passed.
- `test_labelled_maintenance_rejects_overlap_and_invalid_labels` — passed.
- `test_window_minimum_bounding_invalid_input_and_cached_reference` — passed.
- `test_metrics_count_real_statuses_and_bound_route_client_labels` — passed.

### tests/test_operations.py

- `test_api_refuses_jobs_without_a_worker` — passed.
- `test_queue_accepts_one_job_and_exposes_its_real_failure` — passed.
- `test_api_accepts_only_predefined_pipeline_actions[payload0]` — passed.
- `test_api_accepts_only_predefined_pipeline_actions[payload1]` — passed.
- `test_api_accepts_only_predefined_pipeline_actions[payload2]` — passed.
- `test_deployment_requires_a_positive_integer_version[-1]` — passed.
- `test_deployment_requires_a_positive_integer_version[0]` — passed.
- `test_deployment_requires_a_positive_integer_version[True]` — passed.
- `test_deployment_requires_a_positive_integer_version[9]` — passed.
- `test_api_submits_version_without_accepting_shell_or_image_input` — passed.
- `test_unknown_job_is_not_replaced_with_a_successful_record` — passed.
- `test_worker_exit_and_interruption_are_visible` — passed.
- `test_pipeline_command_failure_stops_downstream_stages` — passed.
- `test_rejected_candidate_never_changes_the_running_configuration` — passed.
- `test_failed_restart_restores_previous_settings_and_verifies_recovery` — passed.
- `test_successful_release_and_manual_rollback_preserve_both_identities` — passed.
- `test_api_keeps_saved_environment_contents_private` — passed.
- `test_push_checks_never_reproduce_build_or_deploy[False]` — passed.
- `test_push_checks_never_reproduce_build_or_deploy[True]` — passed.
- `test_release_accepts_constant_predictions_without_a_quality_gate[0.1-Not flagged]` — passed.
- `test_release_accepts_constant_predictions_without_a_quality_gate[0.9-Flag for review]` — passed.

### tests/test_pipeline.py

- `test_ray_batch_logic_preserves_source_identity` — passed.
- `test_ray_batch_logic_rejects_invalid_labels` — passed.
- `test_selection_uses_validation_and_has_deterministic_ties` — passed.
- `test_lineage_requires_the_actual_git_checkout` — passed.
- `test_tracking_registration_and_fresh_process_model_load` — passed.
- `test_training_rejects_missing_project_context[MLFLOW_RUN_ID]` — passed.
- `test_training_rejects_missing_project_context[MLFLOW_TRACKING_URI]` — passed.
- `test_source_snapshot_can_restore_tracked_and_untracked_draft_changes` — passed.
- `test_source_hash_keys_do_not_depend_on_container_mount_path` — passed.

