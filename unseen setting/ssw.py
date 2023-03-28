#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
import pickle
from ssw_aligner import local_pairwise_align_ssw

def calculate_sw(seq1, seq2, gap_open_penalty, gap_extend_penalty, match_score, mismatch_score):
    alignment_12 = local_pairwise_align_ssw(seq1, seq2, gap_open_penalty,
                                     gap_extend_penalty, match_score, mismatch_score)
    score_12 = alignment_12.optimal_alignment_score

    alignment_11 = local_pairwise_align_ssw(seq1, seq1, gap_open_penalty,
                                     gap_extend_penalty, match_score, mismatch_score)
    score_11 = alignment_11.optimal_alignment_score

    alignment_22 = local_pairwise_align_ssw(seq2, seq2, gap_open_penalty,
                                     gap_extend_penalty, match_score, mismatch_score)
    score_22 = alignment_22.optimal_alignment_score

    sw = score_12/np.sqrt(score_11*score_22)

    return sw





    




    
