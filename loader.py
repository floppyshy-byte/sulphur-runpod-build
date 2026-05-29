"""
Sulphur-2 custom loader for RunPod.
Loads the single .safetensors checkpoint and constructs a diffusers LTX2Pipeline.
Based on SulphurAI's pipeline.py, adapted for RunPod's HF cache layout.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Configs (inferred from Sulphur-2 / LTX-2.3 architecture)
# ---------------------------------------------------------------------------

_VAE_CFG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 128,
    "block_out_channels": [256, 512, 1024, 1024],
    "down_block_types": [
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
    ],
    "layers_per_block": [4, 6, 4, 2, 2],
    "spatio_temporal_scaling": [True, True, True, True],
    "downsample_type": ["spatial", "temporal", "spatiotemporal", "spatiotemporal"],
    "encoder_causal": True,
    "encoder_spatial_padding_mode": "zeros",
    "decoder_block_out_channels": [256, 512, 512, 1024],
    "decoder_layers_per_block": [4, 6, 4, 2, 2],
    "decoder_spatio_temporal_scaling": [True, True, True, True],
    "decoder_inject_noise": [False, False, False, False, False],
    "upsample_type": ["spatiotemporal", "spatiotemporal", "temporal", "spatial"],
    "upsample_residual": [True, True, True, True],
    "upsample_factor": [2, 2, 1, 2],
    "decoder_causal": False,
    "decoder_spatial_padding_mode": "reflect",
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-6,
    "scaling_factor": 1.0,
    "timestep_conditioning": False,
    "spatial_compression_ratio": 32,
    "temporal_compression_ratio": 8,
}

_AUDIO_VAE_CFG = {
    "attn_resolutions": None,
    "base_channels": 128,
    "causality_axis": "height",
    "ch_mult": [1, 2, 4],
    "double_z": True,
    "dropout": 0.0,
    "in_channels": 2,
    "is_causal": True,
    "latent_channels": 8,
    "mel_bins": 64,
    "mel_hop_length": 160,
    "mid_block_add_attention": False,
    "norm_type": "pixel",
    "num_res_blocks": 2,
    "output_channels": 2,
    "resolution": 256,
    "sample_rate": 16000,
}

# ---------------------------------------------------------------------------
# Key remapping helpers
# ---------------------------------------------------------------------------

def _extract(sd: dict, prefix: str) -> dict:
    n = len(prefix)
    return {k[n:]: v for k, v in sd.items() if k.startswith(prefix)}


_TRANSFORMER_KEY_MAP = [
    ("diffusion_model.patchify_proj.", "proj_in."),
    ("diffusion_model.audio_patchify_proj.", "audio_proj_in."),
    ("diffusion_model.adaln_single.", "time_embed."),
    ("diffusion_model.audio_adaln_single.", "audio_time_embed."),
    ("diffusion_model.av_ca_a2v_gate_adaln_single.", "av_cross_attn_video_a2v_gate."),
    ("diffusion_model.av_ca_v2a_gate_adaln_single.", "av_cross_attn_audio_v2a_gate."),
    ("diffusion_model.av_ca_video_scale_shift_adaln_single.", "av_cross_attn_video_scale_shift."),
    ("diffusion_model.av_ca_audio_scale_shift_adaln_single.", "av_cross_attn_audio_scale_shift."),
    ("diffusion_model.prompt_adaln_single.", "prompt_adaln."),
    ("diffusion_model.audio_prompt_adaln_single.", "audio_prompt_adaln."),
    ("diffusion_model.transformer_blocks.", "transformer_blocks."),
    ("diffusion_model.proj_out.", "proj_out."),
    ("diffusion_model.audio_proj_out.", "audio_proj_out."),
    ("diffusion_model.scale_shift_table", "scale_shift_table"),
    ("diffusion_model.audio_scale_shift_table", "audio_scale_shift_table"),
]


def _fix_transformer_key(k: str) -> str:
    k = k.replace(".k_norm.", ".norm_k.")
    k = k.replace(".q_norm.", ".norm_q.")
    k = k.replace(".scale_shift_table_a2v_ca_audio", ".audio_a2v_cross_attn_scale_shift_table")
    k = k.replace(".scale_shift_table_a2v_ca_video", ".video_a2v_cross_attn_scale_shift_table")
    return k


def _remap_transformer(model_sd: dict) -> dict:
    out = {}
    for ck, v in model_sd.items():
        mapped = False
        for src, dst in _TRANSFORMER_KEY_MAP:
            if ck.startswith(src):
                new_key = _fix_transformer_key(dst + ck[len(src):])
                out[new_key] = v
                mapped = True
                break
            if ck == src:
                out[_fix_transformer_key(dst)] = v
                mapped = True
                break
        if not mapped:
            pass
    return out


def _fix_connector_keys(rest: str) -> str:
    rest = rest.replace("transformer_1d_blocks.", "transformer_blocks.")
    rest = rest.replace(".k_norm.", ".norm_k.")
    rest = rest.replace(".q_norm.", ".norm_q.")
    return rest


def _remap_connectors(full_sd: dict) -> dict:
    out = {}
    for k, v in full_sd.items():
        if k.startswith("model.diffusion_model.video_embeddings_connector."):
            rest = k[len("model.diffusion_model.video_embeddings_connector."):]
            out[f"video_connector.{_fix_connector_keys(rest)}"] = v
        elif k.startswith("model.diffusion_model.audio_embeddings_connector."):
            rest = k[len("model.diffusion_model.audio_embeddings_connector."):]
            out[f"audio_connector.{_fix_connector_keys(rest)}"] = v
        elif k.startswith("text_embedding_projection.video_aggregate_embed."):
            rest = k[len("text_embedding_projection.video_aggregate_embed."):]
            out[f"video_text_proj_in.{rest}"] = v
        elif k.startswith("text_embedding_projection.audio_aggregate_embed."):
            rest = k[len("text_embedding_projection.audio_aggregate_embed."):]
            out[f"audio_text_proj_in.{rest}"] = v
    return out


def _remap_vae(full_sd: dict) -> dict:
    out = {}
    for ck, v in full_sd.items():
        if not ck.startswith("vae."):
            continue
        k = ck[4:]
        if k == "per_channel_statistics.mean-of-means":
            out["latents_mean"] = v
            continue
        if k == "per_channel_statistics.std-of-means":
            out["latents_std"] = v
            continue
        if k.startswith("encoder.conv_in.") or k.startswith("encoder.conv_out.") or k.startswith("encoder.norm_out."):
            out[k] = v
            continue
        if k.startswith("decoder.conv_in.") or k.startswith("decoder.conv_out.") or k.startswith("decoder.norm_out."):
            out[k] = v
            continue
        if k.startswith("encoder.down_blocks."):
            after = k[len("encoder.down_blocks."):]
            idx = int(after.split(".")[0])
            rest = after[len(str(idx)) + 1:]
            if idx == 8:
                rest = rest.replace("res_blocks.", "resnets.", 1)
                out[f"encoder.mid_block.{rest}"] = v
            elif idx % 2 == 0:
                diffusers_idx = idx // 2
                rest = rest.replace("res_blocks.", "resnets.", 1)
                out[f"encoder.down_blocks.{diffusers_idx}.{rest}"] = v
            else:
                diffusers_idx = idx // 2
                out[f"encoder.down_blocks.{diffusers_idx}.downsamplers.0.{rest}"] = v
            continue
        if k.startswith("decoder.up_blocks."):
            after = k[len("decoder.up_blocks."):]
            idx = int(after.split(".")[0])
            rest = after[len(str(idx)) + 1:]
            if idx == 0:
                rest = rest.replace("res_blocks.", "resnets.", 1)
                out[f"decoder.mid_block.{rest}"] = v
            elif idx % 2 == 1:
                diffusers_idx = (idx - 1) // 2
                out[f"decoder.up_blocks.{diffusers_idx}.upsamplers.0.{rest}"] = v
            else:
                diffusers_idx = (idx - 2) // 2
                rest = rest.replace("res_blocks.", "resnets.", 1)
                out[f"decoder.up_blocks.{diffusers_idx}.{rest}"] = v
            continue
    return out


def _remap_audio_vae(full_sd: dict) -> dict:
    out = {}
    for ck, v in full_sd.items():
        if not ck.startswith("audio_vae."):
            continue
        k = ck[len("audio_vae."):]
        if k == "per_channel_statistics.mean-of-means":
            out["latents_mean"] = v
        elif k == "per_channel_statistics.std-of-means":
            out["latents_std"] = v
        else:
            out[k] = v
    return out


def _remap_vocoder(full_sd: dict) -> dict:
    out = {}
    for ck, v in full_sd.items():
        if not ck.startswith("vocoder."):
            continue
        k = ck[len("vocoder."):]
        k = k.replace("conv_pre.", "conv_in.")
        k = k.replace("conv_post.", "conv_out.")
        k = k.replace("resblocks.", "resnets.")
        k = k.replace(".ups.", ".upsamplers.")
        k = k.replace("act_post.", "act_out.")
        k = k.replace(".downsample.lowpass.filter", ".downsample.filter")
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Checkpoint / cache discovery
# ---------------------------------------------------------------------------

def _find_safetensors(cache_dir: str) -> str | None:
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".safetensors") and "sulphur" in f.lower():
                return os.path.join(root, f)
    return None


def _find_text_encoder_path(cache_dir: str) -> str | None:
    # First look for text_encoder/ subdir in cached repos
    for root, dirs, _files in os.walk(cache_dir):
        if "text_encoder" in dirs:
            p = os.path.join(root, "text_encoder")
            for subroot, _subdirs, subfiles in os.walk(p):
                if "config.json" in subfiles:
                    return subroot
    # Fallback: look for gemma cached repo
    for root, dirs, _files in os.walk(cache_dir):
        for d in dirs:
            if "gemma" in d.lower():
                p = os.path.join(root, d)
                for subroot, _subdirs, subfiles in os.walk(p):
                    if "config.json" in subfiles:
                        return subroot
    return None


def _open_prefix(checkpoint: str, *prefixes: str) -> dict:
    from safetensors import safe_open
    with safe_open(checkpoint, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if any(k.startswith(p) for p in prefixes)]
        return {key: f.get_tensor(key) for key in keys}


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_pipeline(
    checkpoint: str | None = None,
    text_encoder_path: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from diffusers import (
        LTX2Pipeline,
        LTX2VideoTransformer3DModel,
        AutoencoderKLLTX2Video,
        AutoencoderKLLTX2Audio,
        LTXEulerAncestralRFScheduler,
    )
    from diffusers.pipelines.ltx2 import LTX2TextConnectors
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

    if checkpoint is None:
        checkpoint = _find_safetensors(cache_dir)
    if not checkpoint or not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Sulphur checkpoint not found in {cache_dir}")
    checkpoint = Path(checkpoint)

    if text_encoder_path is None:
        text_encoder_path = _find_text_encoder_path(cache_dir)
    if text_encoder_path and os.path.isdir(text_encoder_path):
        text_encoder_src = text_encoder_path
    else:
        text_encoder_src = "unsloth/gemma-3-12b-it"

    print(f"[Sulphur] Checkpoint: {checkpoint}", flush=True)
    print(f"[Sulphur] Text encoder: {text_encoder_src}", flush=True)

    # Video VAE
    print("[Sulphur] Building video VAE ...", flush=True)
    vae = AutoencoderKLLTX2Video.from_config(dict(_VAE_CFG)).to(dtype=torch_dtype)
    raw = _open_prefix(str(checkpoint), "vae.")
    vae_sd = _remap_vae(raw)
    del raw
    gc.collect()
    missing_v, unexpected_v = vae.load_state_dict(vae_sd, strict=False, assign=True)
    del vae_sd
    gc.collect()
    print(f"  vae: {len(missing_v)} missing, {len(unexpected_v)} unexpected", flush=True)
    vae = vae.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    # Audio VAE
    print("[Sulphur] Building audio VAE ...", flush=True)
    audio_vae = AutoencoderKLLTX2Audio.from_config(_AUDIO_VAE_CFG).to(dtype=torch_dtype)
    raw = _open_prefix(str(checkpoint), "audio_vae.")
    audio_vae_sd = _remap_audio_vae(raw)
    del raw
    gc.collect()
    missing_av, unexpected_av = audio_vae.load_state_dict(audio_vae_sd, strict=False, assign=True)
    del audio_vae_sd
    gc.collect()
    print(f"  audio_vae: {len(missing_av)} missing, {len(unexpected_av)} unexpected", flush=True)
    audio_vae = audio_vae.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    # Text connectors
    print("[Sulphur] Building text connectors ...")
    connectors = LTX2TextConnectors(
        caption_channels=5376,
        text_proj_in_factor=35,
        video_connector_num_attention_heads=32,
        video_connector_attention_head_dim=128,
        video_connector_num_layers=8,
        video_connector_num_learnable_registers=128,
        video_gated_attn=True,
        audio_connector_num_attention_heads=32,
        audio_connector_attention_head_dim=64,
        audio_connector_num_layers=8,
        audio_connector_num_learnable_registers=128,
        audio_gated_attn=True,
        connector_rope_base_seq_len=4096,
        rope_theta=10000.0,
        rope_double_precision=True,
        causal_temporal_positioning=False,
        rope_type="split",
        per_modality_projections=True,
        video_hidden_dim=4096,
        audio_hidden_dim=2048,
        proj_bias=True,
    ).to(dtype=torch_dtype)
    raw = _open_prefix(
        str(checkpoint),
        "model.diffusion_model.video_embeddings_connector.",
        "model.diffusion_model.audio_embeddings_connector.",
        "text_embedding_projection.",
    )
    conn_sd = _remap_connectors(raw)
    del raw
    gc.collect()
    missing_c, unexpected_c = connectors.load_state_dict(conn_sd, strict=False, assign=True)
    del conn_sd
    gc.collect()
    print(f"  connectors: {len(missing_c)} missing, {len(unexpected_c)} unexpected", flush=True)
    connectors = connectors.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    # Vocoder
    print("[Sulphur] Building vocoder ...", flush=True)
    vocoder = LTX2VocoderWithBWE().to(dtype=torch_dtype)
    raw = _open_prefix(str(checkpoint), "vocoder.")
    voc_sd = _remap_vocoder(raw)
    del raw
    gc.collect()
    missing_voc, unexpected_voc = vocoder.load_state_dict(voc_sd, strict=False, assign=True)
    del voc_sd
    gc.collect()
    print(f"  vocoder: {len(missing_voc)} missing, {len(unexpected_voc)} unexpected", flush=True)
    vocoder = vocoder.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    # Transformer
    print("[Sulphur] Building transformer ...", flush=True)
    transformer = LTX2VideoTransformer3DModel(
        in_channels=128,
        out_channels=128,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=32,
        attention_head_dim=128,
        cross_attention_dim=4096,
        vae_scale_factors=(8, 32, 32),
        pos_embed_max_pos=20,
        base_height=2048,
        base_width=2048,
        gated_attn=True,
        cross_attn_mod=True,
        audio_in_channels=128,
        audio_out_channels=128,
        audio_patch_size=1,
        audio_patch_size_t=1,
        audio_num_attention_heads=32,
        audio_attention_head_dim=64,
        audio_cross_attention_dim=2048,
        audio_scale_factor=4,
        audio_pos_embed_max_pos=20,
        audio_sampling_rate=16000,
        audio_hop_length=160,
        audio_gated_attn=True,
        audio_cross_attn_mod=True,
        num_layers=48,
        activation_fn="gelu-approximate",
        qk_norm="rms_norm_across_heads",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=5376,
        attention_bias=True,
        attention_out_bias=True,
        rope_theta=10000.0,
        rope_double_precision=True,
        causal_offset=1,
        timestep_scale_multiplier=1000,
        cross_attn_timestep_scale_multiplier=1000,
        rope_type="split",
        use_prompt_embeddings=False,
        perturbed_attn=False,
    ).to(dtype=torch_dtype)
    raw = _open_prefix(str(checkpoint), "model.")
    transformer_sd = _remap_transformer(_extract(raw, "model."))
    del raw
    gc.collect()
    missing, unexpected = transformer.load_state_dict(transformer_sd, strict=False, assign=True)
    del transformer_sd
    gc.collect()
    print(f"  transformer: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    transformer = transformer.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    # Text encoder — load directly to GPU dtype to avoid CPU RAM spike
    print("[Sulphur] Loading text encoder ...", flush=True)
    te_kwargs = {
        "torch_dtype": torch_dtype,
        "local_files_only": True,
    }
    tok_kwargs = {
        "local_files_only": True,
    }
    if device.startswith("cuda"):
        te_kwargs["device_map"] = {"": 0}
    text_encoder = AutoModelForCausalLM.from_pretrained(str(text_encoder_src), **te_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(str(text_encoder_src), **tok_kwargs)
    gc.collect()
    torch.cuda.empty_cache()

    # Scheduler
    scheduler = LTXEulerAncestralRFScheduler()

    # Assemble
    print("[Sulphur] Assembling pipeline ...")
    pipe = LTX2Pipeline(
        transformer=transformer,
        vae=vae,
        audio_vae=audio_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        connectors=connectors,
        vocoder=vocoder,
    )
    print("[Sulphur] Ready.")
    return pipe
