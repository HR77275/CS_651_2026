#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import csv
import json
import time
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import matplotlib.pyplot as plt
import torch.nn.functional as F

import lpips
loss_fn_alex = lpips.LPIPS(net='vgg')

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()

def optimizer_state_nbytes(optimizer):
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += tensor_nbytes(value)
    return total

def parameter_nbytes(gaussians):
    params = [
        gaussians._xyz,
        gaussians._normal,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
    ]
    return sum(tensor_nbytes(param) for param in params)

def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0

def open_metric_writers(model_path):
    jsonl_path = os.path.join(model_path, "training_metrics.jsonl")
    csv_path = os.path.join(model_path, "training_metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    fieldnames = [
        "epoch",
        "iteration",
        "optimizer",
        "num_points",
        "mean_loss",
        "mean_l1_loss",
        "mean_train_psnr",
        "mean_step_time_ms",
        "mean_forward_time_ms",
        "mean_backward_time_ms",
        "mean_optimizer_time_ms",
        "epoch_time_sec",
        "examples_per_sec",
        "optimizer_state_bytes",
        "parameter_bytes",
        "cuda_allocated_bytes",
        "cuda_reserved_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    return jsonl_path, csv_file, csv_writer

def write_epoch_metrics(jsonl_path, csv_file, csv_writer, metrics):
    with open(jsonl_path, "a") as jsonl_file:
        jsonl_file.write(json.dumps(metrics) + "\n")
    csv_writer.writerow(metrics)
    csv_file.flush()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,normal_lr, finetune, load_iteration):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    # finetune = True
    scene = Scene(dataset, gaussians)#, load_iteration=load_iteration)  #, finetune=finetune)
    train_cameras = scene.getTrainCameras()
    epoch_size = max(1, len(train_cameras))
    if opt.epochs > 0:
        opt.iterations = opt.epochs * epoch_size
        opt.position_lr_max_steps = opt.iterations
        print("Using {} epochs x {} train cameras = {} iterations".format(opt.epochs, epoch_size, opt.iterations))
    gaussians.training_setup(opt, normal_lr=normal_lr, finetune=finetune)
    print("Using optimizer: {}".format(opt.optimizer))
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        first_iter = 0 #reset to 0
        gaussians.restore(model_params, opt, finetune)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    jsonl_path, csv_file, csv_writer = open_metric_writers(scene.model_path)
    epoch_metrics = {
        "loss": [],
        "l1_loss": [],
        "train_psnr": [],
        "step_time_ms": [],
        "forward_time_ms": [],
        "backward_time_ms": [],
        "optimizer_time_ms": [],
    }
    epoch_start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    first_iter += 1
    loss_fn_alex.to(gaussians.get_features.device)
    try:
        for iteration in range(first_iter, opt.iterations + 1):


            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = train_cameras.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            forward_start.record()
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            forward_end.record()
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            gt_image = viewpoint_cam.original_image.cuda()


            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            backward_start.record()
            loss.backward()
            backward_end.record()

            iter_end.record()
            torch.cuda.synchronize()

            with torch.no_grad():
                epoch_metrics["loss"].append(loss.item())
                epoch_metrics["l1_loss"].append(Ll1.item())
                epoch_metrics["train_psnr"].append(psnr(image, gt_image).mean().item())
                epoch_metrics["step_time_ms"].append(iter_start.elapsed_time(iter_end))
                epoch_metrics["forward_time_ms"].append(forward_start.elapsed_time(forward_end))
                epoch_metrics["backward_time_ms"].append(backward_start.elapsed_time(backward_end))

                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    current_epoch = (iteration - 1) // epoch_size + 1
                    progress_bar.set_postfix({"Epoch": current_epoch, "Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
            
                    ###get the gradient for xyz for each gaussian 02/21/2025
                    import numpy as np
                    grads = gaussians.xyz_gradient_accum / gaussians.denom
                    grads[grads.isnan()] = 0.0
                    np.save(os.path.join(scene.model_path, "tensor_data.npy"), grads.cpu().numpy())
                    ####
                
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration % 5000==0:
                    if finetune:
                        gaussians.optimizer.param_groups[1]['lr'] /= 1.5 #4 10/08 change, previous is 1.4
                    else:
                        gaussians.optimizer.param_groups[4]['lr'] /= 1.4

                if iteration < opt.densify_until_iter and not finetune:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, size_threshold, iteration)
                                                                           #change 0.005 to 0.01, 10/22/2024
                
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()


                # Optimizer step
                if iteration < opt.iterations:
                    optimizer_start.record()
                    gaussians.optimizer.step()
                    optimizer_end.record()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    torch.cuda.synchronize()
                    epoch_metrics["optimizer_time_ms"].append(optimizer_start.elapsed_time(optimizer_end))
                else:
                    epoch_metrics["optimizer_time_ms"].append(0.0)

                is_epoch_end = (iteration % epoch_size == 0) or (iteration == opt.iterations)
                if is_epoch_end:
                    epoch = (iteration - 1) // epoch_size + 1
                    epoch_time_sec = time.time() - epoch_start_time
                    metrics = {
                        "epoch": epoch,
                        "iteration": iteration,
                        "optimizer": opt.optimizer,
                        "num_points": int(gaussians.get_xyz.shape[0]),
                        "mean_loss": mean_or_zero(epoch_metrics["loss"]),
                        "mean_l1_loss": mean_or_zero(epoch_metrics["l1_loss"]),
                        "mean_train_psnr": mean_or_zero(epoch_metrics["train_psnr"]),
                        "mean_step_time_ms": mean_or_zero(epoch_metrics["step_time_ms"]),
                        "mean_forward_time_ms": mean_or_zero(epoch_metrics["forward_time_ms"]),
                        "mean_backward_time_ms": mean_or_zero(epoch_metrics["backward_time_ms"]),
                        "mean_optimizer_time_ms": mean_or_zero(epoch_metrics["optimizer_time_ms"]),
                        "epoch_time_sec": epoch_time_sec,
                        "examples_per_sec": len(epoch_metrics["loss"]) / epoch_time_sec if epoch_time_sec > 0 else 0.0,
                        "optimizer_state_bytes": optimizer_state_nbytes(gaussians.optimizer),
                        "parameter_bytes": parameter_nbytes(gaussians),
                        "cuda_allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                        "cuda_reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                    }
                    write_epoch_metrics(jsonl_path, csv_file, csv_writer, metrics)
                    print(
                        "\n[EPOCH {}] loss {:.6f} train_psnr {:.3f} opt_time_ms {:.3f} opt_state_mb {:.2f}".format(
                            epoch,
                            metrics["mean_loss"],
                            metrics["mean_train_psnr"],
                            metrics["mean_optimizer_time_ms"],
                            metrics["optimizer_state_bytes"] / (1024 * 1024),
                        )
                    )
                    for values in epoch_metrics.values():
                        values.clear()
                    epoch_start_time = time.time()
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    finally:
        csv_file.close()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += loss_fn_alex.forward(image, gt_image).squeeze()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test,ssim_test,lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        print(f"Number of point: {len(scene.gaussians._xyz)}")
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 1000,3_000, 5_000, 7_000, 12_000, 15_000, 18_000,21_000, 25_000, 28_000, 30_000]) #[7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7_000,18_000, 30_000])

    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000]) #7_000, 18_000, 22_000, 28_000, #[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--normal_lr", type=float, default = 0.003)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--load_iteration',type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    load_iteration = args.load_iteration
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.normal_lr, args.finetune, load_iteration)

    # All done
    print("\nTraining complete.")
