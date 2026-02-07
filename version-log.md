# v2.0 Add resume train function

      This update introduces a robust breakpoint resume mechanism to ensure training continuity 
    and prevent progress loss due to interruptions.

### Key Features:

#### Comprehensive State Saving:

    At the end of every epoch, a master checkpoint file unet_resume.pth is created or updated in the params/ directory.

#### Full Training Context:

    The checkpoint includes
    Weights(model_state_dict), 
    Optimizer state(optimizer_state_dict), 
    Learning rate scheduler state(scheduler_state_dict), 
    GradScaler for mixed precision(scaler_state_dict), 
    Current loss history(logger_train_losses).

#### Automatic Recovery:

    Upon starting a new epoch, the script automatically checks for the existence of unet_resume.pth. If found, it restores the entire training environment and resumes from the exact epoch where it left off.

#### Periodic Archiving:

    In addition to the master resume file, the system saves persistent snapshots every 5 epochs (e.g., unet_epoch_5.pth) for long-term versioning.