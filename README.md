# Flask Template

This sample repo contains the recommended structure for a Python Flask project. In this sample, we use `flask` to build a web application and the `pytest` to run tests.

 For a more in-depth tutorial, see our [Flask tutorial](https://code.visualstudio.com/docs/python/tutorial-flask).

 The code in this repo aims to follow Python style guidelines as outlined in [PEP 8](https://peps.python.org/pep-0008/).

## Running the Sample

To successfully run this example, we recommend the following VS Code extensions:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) 

- Open the template folder in VS Code (**File** > **Open Folder...**)
- Create a Python virtual environment using the **Python: Create Environment** command found in the Command Palette (**View > Command Palette**). Ensure you install dependencies found in the `pyproject.toml` file
- Ensure your newly created environment is selected using the **Python: Select Interpreter** command found in the Command Palette
- Run the app using the Run and Debug view or by pressing `F5`
- To test your app, ensure you have the dependencies from `dev-requirements.txt` installed in your environment
- Navigate to the Test Panel to configure your Python test or by triggering the **Python: Configure Tests** command from the Command Palette
- Run tests in the Test Panel or by clicking the Play Button next to the individual tests in the `test_app.py` file

## CUDA 12.4 Installation on Windows

If you're installing CUDA 12.4 on Windows, you won’t use traditional terminal commands like Linux does. Instead, you’ll use the command prompt (cmd), PowerShell, or Windows Subsystem for Linux (WSL) if you prefer a Linux-like environment.

### Installing CUDA on Windows (CMD or PowerShell)

- **Download the CUDA Installer**  
  Go to the CUDA Downloads page, select Windows, then download the CUDA 12.4 .exe installer.
- **Run the Installer via Command Prompt**  
  Open cmd as Administrator and navigate to the folder containing the installer:

  ```powershell
  cd C:\Path\To\Your\CUDA_Installer
  ```

- **Execute the installer:**

  ```powershell
  start /wait cuda_12.4_windows.exe -silent -driver -toolkit
  ```

  (The `-silent` flag runs it without UI, and `-toolkit` ensures CUDA installs.)
- **Verify Installation**

  Open cmd and check:

  ```powershell
  nvcc --version
  ```

  If installed correctly, you’ll see the CUDA version output.

### Install PyTorch with CUDA 12.4

```sh
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Python Environment Creation (Recommended)

When creating your Python environment, use:

```sh
conda create -n myenv python=3.10.16
```
### REMINDER

Read weigts folder and inside backend folder txtfile

```sh