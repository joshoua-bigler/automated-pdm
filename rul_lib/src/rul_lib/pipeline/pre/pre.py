import pandas as pd
# local
from rul_lib.pipeline.pre.normalize import scale_target
from rul_lib.pipeline.pre.denoise import denoise
from rul_lib.pipeline.pre.cleanup import drop_constant_features, handle_missing_values, cleanup_features, resample
from rul_lib.pipeline.pre.fe_selection import select_by_correlation, select_topk_mi
from rul_lib.gls.gls import logger
from rul_lib.pipeline.pre.normalize import normalize


def preprocessing(data: dict[str, pd.DataFrame], config: dict) -> dict[str, pd.DataFrame]:
  ''' Full preprocessing pipeline according to config.

      Steps applied in order:
        - drop constant features
        - handle missing values
        - denoising
        - normalization
        - feature selection by correlation
        - feature selection by top-k mutual information
        - resampling
        - target scaling
        - cleanup (clipping, healthy data removal)

      Each step reads its config from config['preprocessing'][step_name], e.g.: config['preprocessing']['denoising']
  '''
  logger.info('Starting preprocessing pipeline')
  cfg_pre = config.get('preprocessing', {})
  data_s1 = drop_constant_features(data=data)
  data_s2 = handle_missing_values(data=data_s1, method=cfg_pre.get('imputation', {}).get('method'))
  denoised = denoise(data=data_s2, config=cfg_pre.get('denoising', {}))
  norm = normalize(data=denoised, config=cfg_pre.get('normalization', {}))
  cfg_corr = cfg_pre.get('select_by_correlation', {})
  logger.info(f'Select by correlation: apply={cfg_corr.get('apply', 'UNK')}; method={cfg_corr.get('method', 'UNK')}')
  norm_s1 = select_by_correlation(data=norm, config=cfg_pre.get('select_by_correlation', {}))
  cfg_topk = cfg_pre.get('select_topk_mi', {})
  logger.info(f'Select top k by mutual information: apply={cfg_topk.get('apply', 'UNK')}, top_k={cfg_topk.get('top_k', 'UNK')}')
  norm_s2 = select_topk_mi(data=norm_s1, config=cfg_pre.get('select_topk_mi', {}))
  logger.info(f'Selected top-k sensors by mutual information: {norm_s2.get('meta', {}).get('sensor_selection_mi', {}).get('kept', 'UNK')} kept')
  logger.info(f'Before resampling: {norm_s2['x_train'].shape[0]} training samples')
  norm_s3 = resample(data=norm_s2, config=cfg_pre.get('resampling', {}))
  logger.info(f'After resampling: {norm_s3['x_train'].shape[0]} training samples')
  norm_s4 = scale_target(data=norm_s3, config=cfg_pre.get('target_scaling', {}))
  norm_s5 = cleanup_features(data=norm_s4, config=cfg_pre.get('cleanup', {}))
  logger.info(f'Remaining sensors {norm_s5['x_train'].shape[1]}')
  return norm_s5