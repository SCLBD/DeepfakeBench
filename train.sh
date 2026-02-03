python training/train.py \
--detector_path ./training/config/detector/xception.yaml  \
--train_dataset "FaceForensics++" \
--test_dataset "Celeb-DF-v1" "Celeb-DF-v2" \
--wandb \
--wandb_project DeepfakeBench \
--wandb_run_name xception_ffpp_to_celebdf \
--wandb_group xception_baselines \
--wandb_tags xception,ffpp,baseline \
--wandb_mode online