[v2.0](#v20-add-resume-train-function)

[v3.0](#v30-optimize-inference--multi-mode-result-saving)

# v2.0 Add resume train function

This update introduces a robust breakpoint resume mechanism to ensure training continuity and prevent progress loss due
to interruptions.

### Key Features:

#### Comprehensive State Saving:

At the end of every epoch, a master checkpoint file unet_resume.pth is created or updated in the params/ directory.

#### Full Training Context:

##### The checkpoint includes:

    Weights(model_state_dict), 
    Optimizer state(optimizer_state_dict), 
    Learning rate scheduler state(scheduler_state_dict), 
    GradScaler for mixed precision(scaler_state_dict), 
    Current loss history(logger_train_losses).

#### Automatic Recovery:

Upon starting a new epoch, the script automatically checks for the existence of unet_resume.pth. If found, it restores
the entire training environment and resumes from the exact epoch where it left off.

#### Periodic Archiving:

In addition to the master resume file, the system saves persistent snapshots every 5 epochs `(e.g., unet_epoch_5.pth)`
for long-term versioning.

# v3.0 Optimize Inference & Multi-mode Result Saving

This update focuses on enhancing the inference workflow, adding batch processing capabilities, and providing more
flexible options for saving prediction results.

### Key Features:

#### Dual Inference Scripts:

* **test_single.py**: Optimized for single-image quick testing with improved user interaction.
* **test_multiple.py**: A new script designed for batch processing, capable of iterating through entire folders and
  generating predictions for all images automatically.

#### Versatile Output Management:

You can now choose from three distinct output modes to suit different workflows:

* **Mode 0 (Default)**: Saves all results to a centralized `result/` directory.
* **Mode 1 (Specified)**: Allows users to input a custom directory path for organized storage.
* **Mode 2 (Source)**: Saves prediction results directly into the original image folder, keeping inputs and outputs
  together.

#### Enhanced Robustness:

* **Optimize auto-Directory Creation**: The system now automatically creates target directories (using `os.makedirs`) if
  they do
  not exist, preventing crashes during the saving process.
* **Input Optimization**: Implemented default value handling (`or "0"`) for faster user interaction—simply press Enter
  to use default settings.

#### Improved Visualization:

* The output pixel values are now automatically scaled during the saving process to ensure that class labels (0, 1, 2)
  are clearly visible as grayscale differences in the resulting PNG files. Print for verifying.