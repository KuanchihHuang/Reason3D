import logging
import os
import torch
import torch.nn as nn
import ipdb
import torch.nn.functional as F
import gorilla
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.common.dist_utils import download_cached_file
from lavis.models.reason3d_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.reason3d_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.models.reason3d_models.mask_decoder import MaskDecoder
from lavis.models.reason3d_models.point_extractor import PointExtractor
from lavis.models.reason3d_models.seg_loss import Criterion
from lavis.common.utils import is_url


@registry.register_model("reason3d_t5")
class Reason3DT5(BaseModel):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
    }

    def __init__(
        self,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        apply_lemmatizer=False,
        point_encoder_cfg=None,
        mask_decoder_cfg=None,
        seg_criterion_cfg=None,
        pred_confidence=0.5
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_seg() result with lemmas.
        """
        super().__init__()

        self.encoder = PointExtractor(**point_encoder_cfg)
        self.mask_decoder = MaskDecoder(**mask_decoder_cfg)
        self.region_decoder = MaskDecoder(**mask_decoder_cfg)
        gorilla.load_checkpoint(self.encoder, point_encoder_cfg["pretrained"], strict=False, map_location='cpu')
        
        self.pc_adapter = nn.Linear(point_encoder_cfg["media"], 1408)
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)

        num_added_tokens = self.t5_tokenizer.add_tokens("[SEG]")
        self.t5_tokenizer.add_tokens("[LOC]")
        self.seg_token_idx = self.t5_tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.loc_token_idx = self.t5_tokenizer("[LOC]", add_special_tokens=False).input_ids[0]

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)

        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data

        self.t5_model.get_output_embeddings().requires_grad_(True)
        self.t5_model.get_input_embeddings().requires_grad_(True)

        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)

        self.prompt = ""
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        in_dim = self.t5_model.config.hidden_size
        out_dim = mask_decoder_cfg["d_text"]
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        loc_text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]

        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.loc_text_hidden_fcs = nn.ModuleList([nn.Sequential(*loc_text_fc)])
        self.text_hidden_fcs.train()
        self.loc_text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        for param in self.loc_text_hidden_fcs.parameters():
            param.requires_grad = True
        self.criterion = Criterion(**seg_criterion_cfg)
        self.pred_confidence = pred_confidence

    def forward(self, samples):

        with self.maybe_autocast():
            answer = samples["answer"]
            text_input = samples["text_input"]
            n_answers = samples["n_answers"]
            sp_feats = self.encoder(samples)
            samples["sp_feats"] = sp_feats
            x_feat, batch_mask = self.mask_decoder.get_batches(sp_feats, samples["batch_offsets"])
            pc_embeds = x_feat
            pc_embeds = self.pc_adapter(pc_embeds)
            image_atts = (~batch_mask).long()
        
        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)  # 768 #2, 32, 768
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        if self.prompt:
            text_input = [self.prompt.format(question) for question in text_input]
        else:
            text_input = text_input

        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=400,
                return_tensors="pt",
            ).to(pc_embeds.device)
            output_tokens = self.t5_tokenizer(
                answer,
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(pc_embeds.device)
            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []

            for b, n in enumerate(n_answers):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]] * n
                batch_input_tokens_atts += [input_tokens.attention_mask[b]] * n
                batch_atts_t5 += [atts_t5[b]] * n
                batch_inputs_t5 += [inputs_t5[b]] * n

            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

            encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
                output_hidden_states=True
            )
            seq_out = outputs["decoder_hidden_states"][-1]
            seg_token_index = targets == self.seg_token_idx
            loc_token_index = targets == self.loc_token_idx
        
            seg_out = seq_out[seg_token_index]
            loc_out = seq_out[loc_token_index]

            seg_features = self.text_hidden_fcs[0](seg_out).unsqueeze(1)
            loc_features = self.loc_text_hidden_fcs[0](loc_out).unsqueeze(1)
            samples["text_features"] = seg_features
            samples["loc_text_features"] = loc_features

            out_region = self.region_decoder(samples['sp_feats'], samples['batch_offsets'], samples['loc_text_features'], None)
            pred_region_mask = out_region['masks'].squeeze()[~out_region['batch_mask']].sigmoid().unsqueeze(1)
        
            out = self.mask_decoder(samples['sp_feats'], samples['batch_offsets'], samples['text_features'], pred_region_mask.detach())

            seg_loss_region, log_vars = self.criterion(out_region, samples["gt_pmasks_region"], samples["gt_spmasks_region"], None)
            seg_loss, log_vars = self.criterion(out, samples["gt_pmasks"], samples["gt_spmasks"], None)

            loss = outputs.loss
            loss = loss + seg_loss + seg_loss_region
            return {"loss": loss}

    def predict_seg(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=200,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        repetition_penalty=1.0,
        **kwargs,
    ):

        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            text_input = samples["text_input"]
            sp_feats = self.encoder(samples)
            samples["sp_feats"] = sp_feats
            pc_embeds, batch_mask = self.mask_decoder.get_batches(sp_feats, samples["batch_offsets"])
            pc_embeds = self.pc_adapter(pc_embeds)
            image_atts = (~batch_mask).long()

        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        if isinstance(text_input, str):
            text_input = [text_input]

        prompt = self.prompt
        
        if prompt:
            text_input = [prompt.format(question) for question in text_input]
        else:
            text_input = text_input

        input_tokens = self.t5_tokenizer(text_input, padding="longest", return_tensors="pt").to(pc_embeds.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        num_beams = 1
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                return_dict_in_generate=True,
                output_hidden_states=True
                # for description, also use repetition penalty = 1.5
            )

            seg_mask = outputs["sequences"][:,1:] == self.seg_token_idx
            loc_mask = outputs["sequences"][:,1:] == self.loc_token_idx
            seq_out = outputs['decoder_hidden_states'][-1][-1]
            seg_out = seq_out[seg_mask].mean(axis=0, keepdim=True)
            loc_out = seq_out[loc_mask].mean(axis=0, keepdim=True)

            if seg_out.shape[0] == 0:
                #only allow batch size = 1
                seg_out = torch.rand((1,self.t5_model.config.hidden_size)).cuda()
            if loc_out.shape[0] == 0:
                #only allow batch size = 1
                loc_out = torch.rand((1,self.t5_model.config.hidden_size)).cuda()

            text_features = self.text_hidden_fcs[0](seg_out).unsqueeze(1)
            loc_text_features = self.loc_text_hidden_fcs[0](loc_out).unsqueeze(1)

            samples["text_features"] = text_features
            samples["loc_text_features"] = loc_text_features

            out_region = self.region_decoder(samples['sp_feats'], samples['batch_offsets'], samples['loc_text_features'], None)


            pred_region_mask = out_region['masks'].squeeze(0)[~out_region['batch_mask']].sigmoid().unsqueeze(1)
        
            result = self.mask_decoder(samples['sp_feats'], samples['batch_offsets'], samples['text_features'], pred_region_mask.detach())

            #samples["text_features"] = text_features
            #result = self.mask_decoder(**samples)
            return result

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        point_encoder_cfg = cfg.get("point_encoder_cfg")
        mask_decoder_cfg = cfg.get("mask_decoder_cfg")
        seg_criterion_cfg = cfg.get("seg_criterion_cfg")
        pred_confidence = cfg.get("pred_confidence")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")
        prompt = cfg.get("prompt", "")
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            apply_lemmatizer=apply_lemmatizer,
            point_encoder_cfg=point_encoder_cfg,
            mask_decoder_cfg=mask_decoder_cfg,
            seg_criterion_cfg=seg_criterion_cfg,
            pred_confidence=pred_confidence
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg
