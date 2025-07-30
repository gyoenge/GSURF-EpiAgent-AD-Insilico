## Setup EpiAgent Finetune/Test Environment 

### Requirements 

- this env requires CUDA >= 12.4 
- using python venv 

### Steps 
- Set directory to `./` (project root)

- (Optional in tmux) Run `setup_tmux_scroll.sh`

    ```
    $ chmod +x ./setup/setup_tmux_scroll.sh
    $ ./setup/setup_tmux_scroll.sh
    ```

- Run `setup_venv.sh`

    ```
    $ chmod +x ./setup/setup_venv.sh
    $ ./setup/setup_venv.sh
    ```

- Run `install_torch.sh`

    ```
    $ chmod +x ./setup/install_torch.sh
    $ ./setup/install_torch.sh
    ```

- Run `install_aftertorch.sh`

    ```
    $ chmod +x ./setup/install_aftertorch.sh
    $ ./setup/install_aftertorch.sh
    ```

- Run `install_apttool.sh`

    ```
    $ chmod +x ./setup/install_apttool.sh
    $ ./setup/install_apttool.sh
    ```
