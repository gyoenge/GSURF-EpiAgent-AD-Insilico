import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from flash_attn.models.bert import BertEncoder

class EpiAgent(nn.Module):
    """
    A model that combines the features of Transformer-based architectures and domain-specific layers for cellular 
    regulatory network learning. It performs tasks like cell-cCRE alignment, replacing language modeling (RLM), 
    and signal reconstruction with a transformer-based backbone (BERT).
    
    Args:
        vocab_size (int): The size of the vocabulary (default is 1355449).
        num_layers (int): Number of layers in the transformer encoder (default is 18).
        embedding_dim (int): The dimensionality of the embeddings (default is 512).
        num_attention_heads (int): The number of attention heads in each transformer layer (default is 8).
        max_rank_embeddings (int): The maximum number of rank embeddings for positional encoding (default is 8192).
        MLP_hidden_for_RLM (int): The size of the hidden layer for the Replacing Language Modeling (RLM) task (default is 64).
        MLP_hidden_for_CCA (int): The size of the hidden layer for the Cell-cCRE Alignment (CCA) task (default is 128).
        pos_weight_for_RLM (bool or tensor): Positive weight for RLM loss, if specified (default is False).
        pos_weight_for_CCA (bool or tensor): Positive weight for CCA loss, if specified (default is False).
        pos_weight_signals (tensor): Positive weight for signal reconstruction loss (default is tensor(100)).
        use_flash_attn (bool): Whether to use FlashAttention for transformer encoder (default is True).
    """
    def __init__(self, 
                 vocab_size=1355449, 
                 num_layers=18, 
                 embedding_dim=512, 
                 num_attention_heads=8, 
                 max_rank_embeddings=8192,
                 MLP_hidden_for_RLM=64,
                 MLP_hidden_for_CCA=128,
                 pos_weight_for_RLM=False,
                 pos_weight_for_CCA=False,
                 pos_weight_signals=torch.tensor(100),
                 use_flash_attn=True):
        super(EpiAgent, self).__init__()

        # Model configuration
        self.vocab_size = vocab_size
        self.cCRE_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rank_embedding = nn.Embedding(max_rank_embeddings, embedding_dim)
        self.config = BertConfig(
            vocab_size=vocab_size,
            num_hidden_layers=num_layers,
            hidden_size=embedding_dim,
            num_attention_heads=num_attention_heads,
            intermediate_size=4 * embedding_dim,
            max_position_embeddings=max_rank_embeddings,
            use_flash_attn=use_flash_attn
        )
        self.EpiAgent_transformer = BertEncoder(self.config)

        # Replacing Language Modeling (RLM) components
        self.fc1_for_RLM = nn.Linear(embedding_dim, MLP_hidden_for_RLM)
        self.layer_norm_for_RLM = nn.LayerNorm(MLP_hidden_for_RLM)
        self.dropout_for_RLM = nn.Dropout(0.25)
        self.fc2_for_RLM = nn.Linear(MLP_hidden_for_RLM, 1)

        # Cell-cCRE Alignment (CCA) components
        self.fc1_for_CCA = nn.Linear(embedding_dim * 2, MLP_hidden_for_CCA)
        self.layer_norm_for_CCA = nn.LayerNorm(MLP_hidden_for_CCA)
        self.dropout_for_CCA = nn.Dropout(0.25)
        self.fc2_for_CCA = nn.Linear(MLP_hidden_for_CCA, 1)

        # Activation functions
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # Signal reconstruction components
        self.signal_decoder = nn.Linear(embedding_dim, vocab_size - 4)
        self.criterion_SR = nn.BCEWithLogitsLoss(pos_weight=pos_weight_signals)

        # RLM loss criterion
        self.criterion_RLM = nn.BCEWithLogitsLoss() if not pos_weight_for_RLM else nn.BCEWithLogitsLoss(pos_weight=pos_weight_for_RLM)

        # CCA loss criterion
        self.criterion_CCA = nn.BCEWithLogitsLoss() if not pos_weight_for_CCA else nn.BCEWithLogitsLoss(pos_weight=pos_weight_for_CCA)

    def forward(self, 
                input_ids=None,
                attention_mask=None,            # <- 추가
                inputs_embeds=None,             # <- 추가 (미사용)
                output_attentions=None,         # <- 추가 (미사용)
                output_hidden_states=None,      # <- 추가 (미사용)
                return_dict=None,               # <- 추가 (미사용)
                *,
                return_transformer_output=True, 
                return_SD_output=False, 
                calculate_rlm_loss=False, 
                calculate_cca_loss=False, 
                calculate_sr_loss=False, 
                ex_cell_ccre_ids=None,
                ex_cell_ccre_accessibility=None,
                signals=None,
                **kwargs):                      # <- 추가 (추후 들어올 잔여 키워드 흡수)
        outputs = {}

        # --- 임베딩 계산 ---
        ccre_embeddings = self.cCRE_embedding(input_ids)
        rank_indices = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        rank_embeddings = self.rank_embedding(rank_indices)

        # --- attention_mask 처리 ---
        # flash_attn BertEncoder는 key_padding_mask=True면 '패딩 위치'를 의미합니다.
        # 기존 코드는 (input_ids != 0)를 바로 넘겼으므로 그대로 유지하되,
        # attention_mask가 들어오면 동일 의미로 변환합니다.
        if attention_mask is not None:
            # attention_mask: 1은 keep, 0은 pad 라는 HF 관례가 많습니다.
            key_padding_mask = attention_mask.to(dtype=torch.bool)
        else:
            key_padding_mask = (input_ids != 0)

        # --- 트랜스포머 통과 ---
        transformer_outputs = self.EpiAgent_transformer(
            ccre_embeddings + rank_embeddings,
            key_padding_mask=key_padding_mask
        )
        outputs['transformer_outputs'] = transformer_outputs if return_transformer_output else None

        # --- 아래 나머지 로직은 기존 그대로 ---
        if calculate_rlm_loss:
            if ex_cell_ccre_accessibility is None:
                raise ValueError("`ex_cell_ccre_accessibility` is required for RLM loss computation.")
            rlm_loss, predicted_accessibility = self.RLM_loss(input_ids, transformer_outputs, ex_cell_ccre_accessibility)
            outputs['rlm_loss'] = rlm_loss
            outputs['predicted_accessibility'] = predicted_accessibility

        if calculate_cca_loss:
            if ex_cell_ccre_ids is None or ex_cell_ccre_accessibility is None:
                raise ValueError("Both `ex_cell_ccre_ids` and `ex_cell_ccre_accessibility` are required for CCA loss computation.")
            cca_loss, predicted_cca_accessibility = self.CCA_loss(
                transformer_outputs[:, 0, :], ex_cell_ccre_ids, ex_cell_ccre_accessibility
            )
            outputs['cca_loss'] = cca_loss
            outputs['predicted_cca_accessibility'] = predicted_cca_accessibility

        if calculate_sr_loss:
            if signals is None:
                raise ValueError("`signals` is required for SR loss computation.")
            predicted_signals = self.signal_decoder(transformer_outputs[:, 0, :])
            sr_loss = self.criterion_SR(predicted_signals, signals)
            outputs['sr_loss'] = sr_loss
            outputs['predicted_signals'] = predicted_signals
        elif return_SD_output:
            predicted_signals = self.signal_decoder(transformer_outputs[:, 0, :])
            outputs['predicted_signals'] = predicted_signals

        return outputs

    def CCA_loss(self, cell_embeddings, ex_cell_ccre_ids, ex_cell_ccre_accessibility):
        """
        Computes the Cell-cCRE Alignment (CCA) loss.

        Args:
            cell_embeddings (tensor): Embeddings for the input cells.
            ex_cell_ccre_ids (tensor): cCRE indices for external cells used in the cell-cCRE alignment task.
            ex_cell_ccre_accessibility (tensor): Binary tensor indicating the accessibility of cCREs in the cells.

        Returns:
            tuple: 
                - loss (tensor): The updated CCA loss.
                - predicted_cca_accessibility (list): Predicted accessibility scores for external cCREs.
        """
        repeat_counts = [len(ids) for ids in ex_cell_ccre_ids]
        expanded_cell_embeddings = torch.repeat_interleave(cell_embeddings, torch.tensor(repeat_counts, device=cell_embeddings.device), dim=0)
        concatenated_ccre_ids = torch.cat(ex_cell_ccre_ids, dim=0).to(cell_embeddings.device)
        flattened_accessibility = ex_cell_ccre_accessibility.view(-1, 1).to(cell_embeddings.device)

        concatenated_embeddings = torch.cat((expanded_cell_embeddings, self.cCRE_embedding(concatenated_ccre_ids)), dim=-1)
        predicted_scores = self.fc2_for_CCA(self.layer_norm_for_CCA(self.gelu(self.fc1_for_CCA(concatenated_embeddings))))
        loss = self.criterion_CCA(predicted_scores, flattened_accessibility)

        return loss, predicted_scores.view(-1).cpu().detach().numpy().tolist()

    def RLM_loss(self, input_ids, contextual_embeddings, ccre_accessibility):
        """
        Computes the Replacing Language Modeling (RLM) loss.

        Args:
            input_ids (tensor): cCRE IDs for the input cells.
            contextual_embeddings (tensor): Contextual embeddings for cCREs in the cells.
            ccre_accessibility (tensor): Binary tensor indicating whether the cCREs are accessible in the corresponding cells.

        Returns:
            tuple:
                - loss (tensor): The RLM loss.
                - predicted_accessibility (list): Predicted accessibility scores for cCREs.
        """
        valid_indices = input_ids > 3
        valid_embeddings = contextual_embeddings[valid_indices, :]
        accessibility_targets = ccre_accessibility.reshape(-1, 1)

        predicted_scores = self.fc2_for_RLM(self.layer_norm_for_RLM(self.gelu(self.fc1_for_RLM(valid_embeddings))))
        loss = self.criterion_RLM(predicted_scores, accessibility_targets.to(predicted_scores.device))

        return loss, predicted_scores.view(-1).cpu().detach().numpy().tolist()

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by focusing on hard-to-classify examples.

    Args:
        alpha (float): Weighting factor for the focal loss.
        gamma (float): Focusing parameter to reduce the impact of easy examples.
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Forward pass for focal loss computation.

        Args:
            inputs (tensor): Predicted logits.
            targets (tensor): Ground truth labels.

        Returns:
            tensor: Computed focal loss.
        """
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability is 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature
    def forward(self, feats, labels):
        # feats: [B, D] (정규화된 벡터)
        device = feats.device
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits = feats @ feats.T / self.t
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        mean_log_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        return -mean_log_pos.mean()