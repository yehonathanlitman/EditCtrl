import torch, os, argparse, accelerate, warnings
torch.backends.cuda.enable_cudnn_sdp(False)
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        enable_inpaint_local: bool = False,
        enable_inpaint_global: bool = False,
        global_freeze: bool = False,
        resume_ckpt: str = None,
        extra_preset_lora_paths: str = None,
        extra_preset_lora_models: str = None,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            enable_inpaint_global=enable_inpaint_global,
            redirect_common_files=False,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        if extra_preset_lora_paths is not None and extra_preset_lora_models is not None:
            _paths = [p.strip() for p in extra_preset_lora_paths.split(",") if p.strip()]
            _models = [m.strip() for m in extra_preset_lora_models.split(",") if m.strip()]
            if len(_paths) != len(_models):
                raise ValueError(
                    f"--extra_preset_lora_paths has {len(_paths)} entries but "
                    f"--extra_preset_lora_models has {len(_models)} entries; they must match."
                )
            for _p, _m in zip(_paths, _models):
                _target = getattr(self.pipe, _m, None)
                if _target is None:
                    print(f"[extra_preset_lora] WARNING: pipe.{_m} is None; skipping {_p}")
                    continue
                self.pipe.load_lora(_target, _p)
                print(f"[extra_preset_lora] loaded {_p} into pipe.{_m}")
        
        if global_freeze:
            for name, p in self.pipe.named_parameters():
                if "cross_attn" in name:
                    p.requires_grad = False
            for name, p in self.pipe.named_parameters():
                if "editctrl_" in name:
                    p.requires_grad = True
            trainable = [n for n, p in self.pipe.named_parameters() if p.requires_grad]
            print(f"[global_freeze] {len(trainable)} trainable params; sample: {trainable[:5]}")

        if resume_ckpt is not None:
            from diffsynth.core.loader.file import load_state_dict as _load_sd
            sd = _load_sd(resume_ckpt)
            missing, unexpected = self.pipe.dit.load_state_dict(sd, strict=False)
            print(f"[resume] loaded {len(sd)} keys from {resume_ckpt}; "
                  f"{len(unexpected)} unexpected key(s)")
            if len(unexpected) > 0:
                print(f"[resume] WARNING unexpected keys (first 5): {unexpected[:5]}")

        self.enable_inpaint_local = enable_inpaint_local
        self.enable_inpaint_global = enable_inpaint_global

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:inpaint": lambda pipe, inputs_shared, inputs_posi, inputs_nega:
                WanVideoInpaintMaskedLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        if self.enable_inpaint_local or self.enable_inpaint_global:
            inputs_shared["inpaint_local_enabled"] = self.enable_inpaint_local
            inputs_shared["inpaint_global_enabled"] = self.enable_inpaint_global
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Stop training once model_logger.num_steps reaches this many dataloader iterations.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    parser.add_argument("--enable_inpaint_local", action="store_true",
                        help="Enable token-level mask slicing in VACE (local path).")
    parser.add_argument("--enable_inpaint_global", action="store_true",
                        help="Enable downsampled-input-latent tokens in cross-attn context (global path).")
    parser.add_argument("--global_freeze", action="store_true",
                        help="Freeze all cross-attn then unfreeze editctrl_* params (global path).")
    parser.add_argument("--dataset_fps", type=int, default=8,
                        help="Target FPS for VideoInpaintingDataset subsampling (sft:inpaint only).")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Path to a saved DiT-delta checkpoint to resume from (loaded into pipe.dit with strict=False).")
    parser.add_argument("--extra_preset_lora_paths", type=str, default=None,
                        help="Comma-separated additional preset LoRA paths (loaded frozen, one per --extra_preset_lora_models entry).")
    parser.add_argument("--extra_preset_lora_models", type=str, default=None,
                        help="Comma-separated module names matching --extra_preset_lora_paths (e.g. 'vace2').")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    if args.task == "sft:inpaint":
        from diffsynth.core.data.video_inpainting_dataset import VideoInpaintingDataset
        dataset = VideoInpaintingDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=getattr(args, "dataset_fps", 8),
            is_train=True,
            repeat=args.dataset_repeat,
        )
    else:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4 if not args.framewise_decoding else 1,
                time_division_remainder=1 if not args.framewise_decoding else 0,
            ),
            special_operator_map={
                "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
                "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
                "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
            }
        )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        enable_inpaint_local=args.enable_inpaint_local,
        enable_inpaint_global=args.enable_inpaint_global,
        global_freeze=args.global_freeze,
        resume_ckpt=args.resume_ckpt,
        extra_preset_lora_paths=args.extra_preset_lora_paths,
        extra_preset_lora_models=args.extra_preset_lora_models,
    )
    resume_start_step = 0
    if args.resume_ckpt is not None:
        import re
        m = re.search(r"step-(\d+)", os.path.basename(args.resume_ckpt))
        if m:
            resume_start_step = int(m.group(1))
            print(f"[resume] continuing global step counter from {resume_start_step}")
        else:
            print(f"[resume] could not parse step number from {args.resume_ckpt!r}; starting step counter at 0")
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        start_step=resume_start_step,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "sft:inpaint": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
