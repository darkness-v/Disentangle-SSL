import torch
import torch.nn as nn
import fairseq
from speechbrain.inference.speaker import EncoderClassifier
import torch.nn.functional as F
class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        ckpt_path = 'facebook/wav2vec2-xls-r-300m'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task_from_hf_hub([ckpt_path])
        self.model = model[0].to(device)
        self.device = device
        self.out_dim = 1024
    
    def extract_features(self, x):
        if next(self.model.parameters()).device != x.device or next(self.model.parameters()).dtype != x.dtype:
            self.model = self.model.to(x.device).to(x.dtype)
            self.model.train()
        
        if x.ndim == 3:
            x = x[:,:,0]
        emb = self.model(x, mask=False, features_only=True)['x']
        return emb

# Use ecapa-tdnn as speaker encoder
class SpeakerEncoder(nn.Module):
    def __init__(self, device):
        super(SpeakerEncoder, self).__init__()
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        self.device = device
        self.out_dim = self.classifier.mods.linear.out_features
    
    def forward(self, x):
        """
        x: [B, T, D]
        Returns:
            embedding: [B, out_dim]
        """
        encoder_output = self.classifier.mods.encoder(x)
        embedding = self.classifier.mods.attn(encoder_output)
        return embedding

class DisentangleSSLModel(nn.Module):
    def __init__(self, ssl_model, speaker_encoder, proj_dim):
        super(DisentangleSSLModel, self).__init__()
        self.ssl_model = ssl_model
        self.speaker_encoder = speaker_encoder
        self.proj_ssl = nn.Linear(self.ssl_model.out_dim, proj_dim)
        self.proj_spk = nn.Linear(self.speaker_encoder.out_dim, proj_dim)
    def forward(self, x, num_emotions = 5):
        ssl_feats = []
        for i in range(num_emotions):
            ssl_feat = self.ssl_model.extract_features(x[:, i].squeeze(-1)) # [B, T, ssl_dim]
            ssl_feats.append(ssl_feat)
        
        ssl_feat = torch.mean(torch.stack(ssl_feats, dim=0), dim=0)  # [B, T, ssl_dim]
        ssl_emb = self.proj_ssl(ssl_feat)  # [B, T, proj_dim]
        speaker_identity = self.speaker_encoder(ssl_emb)  # [B, spk_emb_dim]

        neutral_x = x[:, 0].squeeze(-1)  # [B, T]
        neutral_x_ssl = self.ssl_model.extract_features(neutral_x)  # [B, T, ssl_dim]
        spk_emb = self.speaker_encoder(neutral_x_ssl)  
        
        sim = F.cosine_similarity(speaker_identity, spk_emb, dim=-1)
        loss = 1.0 - sim
        batch_loss = loss.mean()
    
        
        return loss, batch_loss
