# FEMTO
## Modeling
- Combine CAE → CNN/TCN/LSTM for temporal-spatial features  
- Attention or Transformer modules mitigate mean-collapse  

## Domain Adaptation
- Use feature- or adversarial-based TL (MMD, WD-DANN, SDAN)  
- Fine-tune lower CNN/LSTM layers; freeze early features  
- GAN-based or diffusion augmentation helps with few runs  

## Key Challenges
- Sparse failures → overfitting  
- Cross-condition drift → domain adaptation needed  
- Models often regress to mean → add attention, hybrid features