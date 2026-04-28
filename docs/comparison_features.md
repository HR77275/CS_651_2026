# Optimizer Comparison Features

## Primary quality metrics
- `val.miou`: primary comparison metric for semantic segmentation
- `train.miou`
- `val.pixel_accuracy`
- `train.pixel_accuracy`
- `val.mean_loss`
- `train.mean_loss`
- `val.mean_main_loss`
- `train.mean_main_loss`
- `val.mean_aux_loss`
- `train.mean_aux_loss`
- `val.per_class_iou_named`

## Efficiency metrics
- `train.mean_step_time_sec`
- `train.mean_forward_time_sec`
- `train.mean_backward_time_sec`
- `train.mean_optimizer_time_sec`
- `train.examples_per_sec`
- `val.mean_step_time_sec`
- `val.examples_per_sec`
- `train.epoch_time_sec`
- `val.epoch_time_sec`

## Memory metrics
- `train.optimizer_state_bytes`
- `train.parameter_bytes`

## Bookkeeping metrics
- `train.learning_rate`
- `train.epoch` / `val.epoch`
- `train.num_batches` / `val.num_batches`
- `train.num_examples` / `val.num_examples`
