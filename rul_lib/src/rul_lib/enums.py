from enum import Enum


class PipelineType(str, Enum):
  FE_REG = 'fe_reg'
  SEQ2VAL = 'seq2val'
