import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration, Adafactor
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from collections import defaultdict
from info_nce import InfoNCE


class KGT5_Model(pl.LightningModule):
    def __init__(self,
                 config,
                 data_module,
                 model_size='t5-small', 
                 use_ptlm=False,
                 ):
        super().__init__()
        self.config = config
        self.dataset = data_module.dataset
        self.num_predictions = self.config.eval.num_predictions
        self.max_length = self.config.eval.max_length
        self.tokenizer = data_module.tokenizer
        vocab_size = self.tokenizer.vocab_size
        self.special_token_dict = {
            'description': "<extra_id_96>",  # 32003
            'tail_pre': "<extra_id_55>",  # 32044
            'head_pre': "<extra_id_56>",  # 32043
            'subject': "<extra_id_60>",  # 32039
            'relation': "<extra_id_61>",  # 32038
            'object': "<extra_id_62>",  # 32037
        }
        self.gama = 15.0
        self.tau = 0.8

        if self.tokenizer.vocab_size == 32100:
            vocab_size = 32128 # TODO this is hack for default t5 tokenizer. don't know why this happens

        print('Vocab size: ', vocab_size)
        if not use_ptlm:
            t5_config = T5Config().from_pretrained(model_size)
            t5_config.vocab_size = vocab_size
            self.model = T5ForConditionalGeneration(t5_config)
            print('Model loaded from scratch!')
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_size)
            print('Initialized model from pretrained weights (LM)')

        self.contrast_mapping = nn.Sequential(
            nn.Linear(self.model.model_dim, self.model.model_dim, bias=True),
            nn.ELU(),
            nn.Linear(self.model.model_dim, self.model.model_dim, bias=True)
        )

    def training_step(self, batch, batch_idx):

        outputs = self.model(**batch, output_hidden_states=True)
        CE_loss = outputs.loss  # Tail Regression loss

        # # 找出batch中Soft Token所在indices，计算额外的Triple loss
        # reasoning_loss = self.reasoning_loss(batch, outputs)  # Triple loss
        # loss = CE_loss + reasoning_loss
        #
        # self.log_dict({"CE_loss": CE_loss.detach(), "Triple_loss": reasoning_loss.detach()}, prog_bar=True)

        # similarity_loss = self.similarity_loss(batch, outputs)  # Similarity loss
        infonce_loss = self.InfoNCE_loss(batch, outputs)  # InfoNCE loss
        loss = CE_loss + infonce_loss

        self.log_dict({"CE_loss": CE_loss.detach(), "InfoNCE_loss": infonce_loss.detach()}, prog_bar=True)

        # self.log("CE_loss", CE_loss.detach(), prog_bar=True)
        # self.log("Triple_loss", reasoning_loss.detach(), prog_bar=True)
        self.log("loss", loss.detach())
        return loss

    def configure_optimizers(self):
        print('Using default adafactor, lr=None')
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)  # TODO: Loss Nan ???
        # optimizer = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=4e-5)
        return optimizer

    def reasoning_loss(self, batch, outputs):
        """
        找出batch中Soft Token所在indices，计算额外的Triple loss
        :param batch:
        :param outputs:
        :return:
        """
        # description: <extra_id_96> 32003;  tail_pre: <extra_id_55> 32044;  head_pre: <extra_id_56> 32043
        # [Subject] token <extra_id_60> 32039; [Relation] token <extra_id_61> 32038; [Object] token <extra_id_62> 32037
        input_ids = batch['input_ids']  # [batch_size, seq_len]

        # find number in input_ids which equals to special token
        mask_sub = input_ids == self.tokenizer.convert_tokens_to_ids(self.special_token_dict['subject'])
        mask_rel = input_ids == self.tokenizer.convert_tokens_to_ids(self.special_token_dict['relation'])
        mask_obj = input_ids == self.tokenizer.convert_tokens_to_ids(self.special_token_dict['object'])
        # mask_special_tokens = mask_sub | mask_rel | mask_obj
        # mask_special_tokens[:, 0] = True  # [tail_pre]/[head_pre] token
        subject_idx = torch.where(mask_sub)
        relation_idx = torch.where(mask_rel)
        object_idx = torch.where(mask_obj)
        # 取出model中encoder最后一层的hidden state
        encoder_last_hidden_state = outputs.encoder_last_hidden_state  # [batch_size, seq_len, hidden_size]
        # 找到subject, relation, object对应的hidden state
        subject_hidden_state = encoder_last_hidden_state[subject_idx[0], subject_idx[1], :]  # [batch_size, hidden_size]
        relation_hidden_state = encoder_last_hidden_state[relation_idx[0], relation_idx[1],
                                :]  # [batch_size, hidden_size]
        object_hidden_state = encoder_last_hidden_state[object_idx[0], object_idx[1], :]  # [batch_size, hidden_size]

        triple_loss = self.TransE(subject_hidden_state, relation_hidden_state, object_hidden_state)  # Triple loss
        reasoning_loss = -F.logsigmoid(triple_loss).mean()
        reasoning_loss = reasoning_loss * 0.003
        return reasoning_loss

    def similarity_loss(self, batch, outputs):
        # Compute the similarity loss between the tail and the head-relation pair
        input_ids = batch['input_ids']  # [batch_size, seq_len]
        encoder_last_hidden_state = outputs.encoder_last_hidden_state  # [batch_size, seq_len, hidden_size]
        decoder_hidden_states = outputs.decoder_hidden_states  # [batch_size, seq_len, hidden_size]

        hr_embedding = encoder_last_hidden_state[:, 1:, :]  # [batch_size, seq_len-1, hidden_size]
        t_embedding = decoder_hidden_states[-1]  # [batch_size, output_seq_len, hidden_size]

        # TODO: 注意力池化？
        hr_pool = torch.mean(hr_embedding, dim=1)  # [batch_size, hidden_size]
        t_pool = torch.mean(t_embedding, dim=1)  # [batch_size, hidden_size]

        similarity = F.cosine_similarity(hr_pool, t_pool, dim=-1)  # Similarity loss
        loss = 1 - similarity
        loss = loss.sum()
        return loss * 0.05

    def InfoNCE_loss(self, batch, outputs):
        encoder_last_hidden_state = outputs.encoder_last_hidden_state  # [batch_size, seq_len, hidden_size]
        decoder_hidden_states = outputs.decoder_hidden_states  # [batch_size, seq_len, hidden_size]

        hr_embedding = encoder_last_hidden_state[:, 1:, :]  # [batch_size, seq_len-1, hidden_size]
        t_embedding = decoder_hidden_states[-1]  # [batch_size, output_seq_len, hidden_size]

        hr_pool = torch.mean(hr_embedding, dim=1)  # [batch_size, hidden_size]
        t_pool = torch.mean(t_embedding, dim=1)  # [batch_size, hidden_size]

        hr = self.contrast_mapping(hr_pool)
        t = self.contrast_mapping(t_pool)

        # Compute the InfoNCE loss between the tail and the head-relation pair
        loss = InfoNCE()
        output = loss(hr, t)
        return output

    def get_scores(self, ids, scores):
        pad_token_id = self.tokenizer.pad_token_id
        # ids is list of tokenized strings
        # scores is a list of tensors. each tensor contains score of each token in vocab
        # conditioned on ids till that point
        # stack scores
        scores = torch.stack(scores, dim=1)

        # after stacking, shape is (batch_size*num_return_sequences, num tokens in sequence, vocab size)
        # get probs
        log_probs = torch.log_softmax(scores, dim=2)
        # remove start token
        ids = ids[:, 1:]
        # gather needed probs
        x = ids.unsqueeze(-1).expand(log_probs.shape)
        needed_logits = torch.gather(log_probs, 2, x)
        final_logits = needed_logits[:, :, 0]
        padded_mask = (ids == pad_token_id)
        final_logits[padded_mask] = 0
        final_scores = final_logits.sum(dim=-1)

        return final_scores

    # common function for test and val evaluation
    def evaluate(self, batch, mode='val'):
        # Todo: this method assumes a batch size of 1 currently, fix if needed

        # parsing the input
        input_batch = {
            'input_ids': batch['input_ids'], 
            'attention_mask': batch['attention_mask'],
            'temperature': 1.0,  # TODO: make this argument?
            'do_sample': True,
            'num_return_sequences': self.num_predictions,
            'num_beams': 1,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'max_length': self.max_length,
            'output_scores': True,
            'return_dict_in_generate': True,
        }
        outputs = self.generate(**input_batch)#, max_new_tokens=128)
        sequences = outputs.sequences
        scores = outputs.scores
        scores = self.get_scores(sequences, scores)
        predictions = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        targets = batch["targets"]
        queries = batch["queries"]
        is_tail_pred = batch["is_tail_pred"][0]
        target = targets[0]
        query = queries[0]
        ranks_dict = defaultdict(list)
        preds = np.array(predictions)
        true_pos = (preds == target).nonzero()[0]
        if len(true_pos) == 0:
            ranks_dict["ranks"].append(self.dataset.num_entities)
            if is_tail_pred:
                ranks_dict["tail_ranks"].append(self.dataset.num_entities)
            else:
                ranks_dict["head_ranks"].append(self.dataset.num_entities)
            return ranks_dict

        true_pos = true_pos[0]
        true_score = scores[true_pos]
        true_answers = self.dataset.filter_dict[query]
        unique_preds, unique_indices = np.unique(preds, return_index=True)
        scores = scores.detach().cpu().numpy()
        relevant_scores = scores[unique_indices]
        rank = 0
        ties = 0
        for p, score in zip(unique_preds.tolist(), relevant_scores.tolist()):
            if p in true_answers:
                continue
            if score > true_score:
                rank += 1
            if score == true_score:
                ties += 1
        ranks_dict["ranks"].append(rank + ties // 2 + 1)
        if is_tail_pred:
            ranks_dict["tail_ranks"].append(rank + ties // 2 + 1)
        else:
            ranks_dict["head_ranks"].append(rank + ties // 2 + 1)
        return ranks_dict

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    # validation loop
    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, mode='val')

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, mode='test')

    def metric_aggregation(self, ranks_dicts):
        ranks = np.array([rd["ranks"] for rd in ranks_dicts]).squeeze()
        head_ranks = np.array([rd["head_ranks"] for rd in ranks_dicts if len(rd["head_ranks"]) > 0]).squeeze()
        tail_ranks = np.array([rd["tail_ranks"] for rd in ranks_dicts if len(rd["tail_ranks"]) > 0]).squeeze()
        for r, suffix in zip([ranks, head_ranks, tail_ranks], ["", "_head", "_tail"]):
            if len(r) != 0:
                mrr = np.mean(1/r).item()
                h1 = np.mean(r <= 1).item()
                h3 = np.mean(r <= 3).item()
                h10 = np.mean(r <= 10).item()
            else:
                mrr = 0.0
                h1 = 0.0
                h3 = 0.0
                h10 = 0.0
            self.log(f"mrr{suffix}", mrr, sync_dist=True)
            self.log(f"h1{suffix}", h1, sync_dist=True)
            self.log(f"h3{suffix}", h3, sync_dist=True)
            self.log(f"h10{suffix}", h10, sync_dist=True)
            print(f"\nmrr{suffix}", mrr)
            print(f"h1{suffix}", h1)
            print(f"h3{suffix}", h3)
            print(f"h10{suffix}", h10)

    def on_validation_epoch_start(self) -> None:
        # call filterdict to make sure it is created
        self.dataset.filter_dict

    def on_test_epoch_start(self) -> None:
        # call filterdict to make sure it is created
        self.dataset.filter_dict

    def validation_epoch_end(self, ranks):
        return self.metric_aggregation(ranks)

    def test_epoch_end(self, ranks):
        return self.metric_aggregation(ranks)

    def TransE(self, head, relation, tail):
        score = (head + relation) - tail
        score = self.gama - torch.norm(score, p=1, dim=-1)
        return score

    def RotatE(self, head, relation, tail):
        head = head.unsqueeze(1)
        relation = relation.unsqueeze(1)
        tail = tail.unsqueeze(1)
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        rel1, rel2 = torch.chunk(relation, 2, dim=2)
        relation = rel1 + rel2

        phase_relation = relation / (384 / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gama - score.sum(dim=-1)
        return score

    def ComplEx(self, head, relation, tail):
        head = head.unsqueeze(1)
        relation = relation.unsqueeze(1)
        tail = tail.unsqueeze(1)
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score


    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    # def InfoNCE_loss(self, input_emb, subject_idx, relation_idx, object_idx):
    #     # 抽出head和rel的embedding拼起来，然后计算与tail的对比损失
    #     # input_emb: batch_size x num_neg_samples+1 x emb_dim
    #     batch_size, token_nums, emb_dim = input_emb.shape
    #     target_nodes = input_emb[object_idx[0], object_idx[1], :]  # [batch_size, hidden_size]
    #     context_samples = []
    #     for i in range(batch_size):
    #         tmp_subject = input_emb[i, subject_idx[1][i]+1:relation_idx[i], :]  # [subject_tokens, emb_dim]
    #         tmp_relation = input_emb[i, relation_idx[1][i]+1:object_idx[i], :]  # [relation_tokens, emb_dim]
    #         tmp_context = torch.concat((tmp_subject, tmp_relation), dim=0)  # [subject_tokens + relation_tokens, emb_dim]
    #         context_samples.append(tmp_context)
    #     loss = torch.tensor([0])
    #
    #     for i in range(batch_size):
    #         tmp_target = target_nodes[i, :].unsqueeze(0)  # [1, emb_dim]
    #
    #         tmp_loss = self.loss_func(input_emb[i, 0, :], target_nodes[i, :], context_samples[i])


