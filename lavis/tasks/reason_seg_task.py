"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.models.reason3d_models.seg_loss import get_iou
import numpy as np
import torch

@registry.register_task("3d_reason_seg")
class ThreeDReasonSegTask(BaseTask):
    

    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
        save_results=False
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt
        self.save_results = save_results
        self.save_dir = "reason_preds"

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")
        save_results = run_cfg.get("save_results", False)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
            save_results=save_results
        )

    def valid_step(self, model, samples):
        result = model.predict_seg(
            samples=samples,
            answer_list=None,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )

        #TODO: currently only support B = 1 when predict
        assert len(samples["gt_pmasks"]) == 1, 'current only support batch size = 1'
        gt_pmask = samples["gt_pmasks"][0]
        gt_spmask = samples["gt_spmasks"][0]
        pred_spmask = result['masks'][-1].squeeze()
        spiou = get_iou(pred_spmask, gt_spmask, pred_confidence = model.pred_confidence)
        pred_pmask = pred_spmask[samples["superpoints"]]
        piou = get_iou(pred_pmask, gt_pmask, pred_confidence = model.pred_confidence)

        result = dict(scan_id=samples["scan_ids"][0], object_id=samples["object_ids"][0], ann_id=samples["ann_ids"][0], piou=piou, spiou=spiou, gt_pmask=gt_pmask, pred_pmask=pred_pmask)

        if self.save_results:
            import pickle
            os.makedirs(self.save_dir, exist_ok=True)
            ann_id = result["ann_id"]
            scan_id = result["scan_id"]
            gt_pmask = result["gt_pmask"].cpu().numpy()
            pred_pmask = result["pred_pmask"].sigmoid().cpu().numpy()
            text_input = samples['text_input'][0]
            sp_filename = samples["sp_filenames"][0]

            with open(os.path.join(self.save_dir, str(ann_id) + ".pkl"), 'wb') as f:
                pickle.dump({"scan_id":scan_id, "gt_pmask": gt_pmask, "pred_pmask": pred_pmask, "text_input": text_input, "sp_filename": sp_filename}, f)

        return [{"result": result}]

    
    def after_evaluation(self, val_result, split_name, epoch):
        
        pious = []
        spious = []

        print("===================================")
        print(f"3D Reasoning segmentation (Search) on Matterport3D ({len(val_result)} samples):")
        for i, result in enumerate(val_result):
            piou = result['result']['piou']
            spiou = result['result']['spiou']
            pious.append(piou)
            spious.append(spiou)

        pious = torch.stack(pious, dim=0).cpu().numpy()
        precision_half = (pious > 0.5).sum().astype(float) / pious.size
        precision_quarter = (pious > 0.25).sum().astype(float) / pious.size
        miou = pious.mean()

        print("Val result: mIoU/Acc50/Acc25 {:.4f}/{:.4f}/{:.4f}".format(
            miou, precision_half, precision_quarter
        ))
