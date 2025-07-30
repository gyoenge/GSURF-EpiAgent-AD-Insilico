## Setup EpiAgent Finetune/Test Environment 

### Requirements 

- this env requires CUDA >= 12.4 
- using python venv 

### Steps 

- (Optional in tmux) Run `setup_tmux_scroll.sh`

    ```
    $ chmod +x setup_tmux_scroll.sh
    $ ./setup_tmux_scroll.sh
    ```

- Run `setup_venv.sh`

    ```
    $ chmod +x setup_venv.sh
    $ ./setup_venv.sh
    ```

- Run `install_torch.sh`

    ```
    $ chmod +x install_torch.sh
    $ ./install_torch.sh
    ```

- Run `install_aftertorch.sh`

    ```
    $ chmod +x install_aftertorch.sh
    $ ./ install_aftertorch.sh
    ```

- Run `install_apttool.sh`

    ```
    $ chmod +x install_apttool.sh
    $ ./install_apttool.sh
    ```
