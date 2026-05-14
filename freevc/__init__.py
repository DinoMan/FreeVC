"""FreeVC: Text-free one-shot voice conversion."""

import os
import torch
import numpy as np
import librosa
from huggingface_hub import hf_hub_download
from pathlib import Path

from .models import SynthesizerTrn
from .wavlm import WavLM, WavLMConfig
from .speaker_encoder.voice_encoder import SpeakerEncoder
from . import utils


class FreeVC:
    """FreeVC inference wrapper."""

    def __init__(self, model_name="freevc", device="cuda"):
        """Load FreeVC model.

        Args:
            model_name: "freevc" or "freevc-s"
            device: "cuda" or "cpu"
        """
        self.device = device
        self.sr = 16000

        # Download checkpoints
        config_path = hf_hub_download("OlaWod/FreeVC", f"logs/{model_name}.json")
        ckpt_path = hf_hub_download("OlaWod/FreeVC", f"checkpoints/{model_name}.pth")
        wavlm_path = hf_hub_download("OlaWod/FreeVC", "wavlm/WavLM-Large.pt")
        spk_path = hf_hub_download("OlaWod/FreeVC", "speaker_encoder/ckpt/pretrained_bak_5805000.pt")

        # Load config
        self.hps = utils.get_hparams_from_file(config_path)

        # Load model
        self.net_g = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model,
        ).to(device)
        self.net_g.eval()
        utils.load_checkpoint(ckpt_path, self.net_g, None, True)

        # Load WavLM
        checkpoint = torch.load(wavlm_path, map_location=device)
        cfg = WavLMConfig(checkpoint['cfg'])
        self.cmodel = WavLM(cfg).to(device)
        self.cmodel.load_state_dict(checkpoint['model'])
        self.cmodel.eval()

        # Load speaker encoder
        self.smodel = SpeakerEncoder(spk_path, device=device)

    def convert(self, source_wav, target_wav):
        """Convert source speech to target speaker's voice.

        Args:
            source_wav: numpy array of source audio at 16kHz
            target_wav: numpy array of target audio at 16kHz

        Returns:
            (sample_rate, converted_audio_numpy)
        """
        with torch.no_grad():
            # Extract content from source
            wav_src = torch.from_numpy(source_wav).float().unsqueeze(0).to(self.device)
            c = self.cmodel.extract_features(wav_src)[0]
            c = c.transpose(1, 2)

            # Extract speaker embedding from target
            wav_tgt = target_wav.copy()
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            g_tgt = self.smodel.embed_utterance(wav_tgt)
            g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).unsqueeze(-1).to(self.device)

            # Generate
            audio = self.net_g.infer(c, g=g_tgt)
            audio = audio[0][0].data.cpu().float().numpy()

        return self.hps.data.sampling_rate, audio
