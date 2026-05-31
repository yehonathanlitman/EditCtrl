export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MALLOC_TRIM_THRESHOLD_=0
accelerate launch --mixed_precision bf16 --num_processes 2 examples/wanvideo/model_training/train.py \
  --task "sft:inpaint" \
  --dataset_base_path VPData/videovo_raw_videos \
  --dataset_metadata_path VPData/pexels_videovo_train_dataset.csv \
  --data_file_keys "video,vace_video,vace_video_mask" \
  --height 480 --width 720 --num_frames 49 \
  --model_id_with_origin_paths "PAI/Wan2.2-VACE-Fun-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,PAI/Wan2.2-VACE-Fun-A14B:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-VACE-Fun-A14B:Wan2.1_VAE.pth" \
  --learning_rate 5e-6 \
  --num_epochs 1 \
  --save_steps 500 \
  --max_train_steps 10000 \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 128 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.2-VACE-Fun-A14B_editctrl_local_low_noise" \
  --extra_inputs "vace_video,vace_video_mask" \
  --enable_inpaint_local \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358 \
  --dataset_num_workers 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MALLOC_TRIM_THRESHOLD_=0
accelerate launch --mixed_precision bf16 --num_processes 2 examples/wanvideo/model_training/train.py \
  --task "sft:inpaint" \
  --dataset_base_path VPData/videovo_raw_videos \
  --dataset_metadata_path VPData/pexels_videovo_train_dataset.csv \
  --data_file_keys "video,vace_video,vace_video_mask" \
  --height 480 --width 720 --num_frames 49 \
  --model_id_with_origin_paths "PAI/Wan2.2-VACE-Fun-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,PAI/Wan2.2-VACE-Fun-A14B:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-VACE-Fun-A14B:Wan2.1_VAE.pth" \
  --learning_rate 5e-6 \
  --num_epochs 1 \
  --save_steps 500 \
  --max_train_steps 10000 \
  --preset_lora_path "./models/train/Wan2.2-VACE-Fun-A14B_editctrl_local_low_noise/step-10000.safetensors" \
  --preset_lora_model "vace" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-VACE-Fun-A14B_editctrl_global_low_noise" \
  --extra_inputs "vace_video,vace_video_mask" \
  --enable_inpaint_local \
  --enable_inpaint_global \
  --global_freeze \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358 \
  --dataset_num_workers 1
