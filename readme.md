## App Installation

To install the app, we have to do the following:

1. Install Mini-Conda in your computer (if already not installed). The following link can be used for installation: [LINK](https://docs.anaconda.com/miniconda/install/#quick-command-line-install).

2. Clone the repo using git:
   ```
   https://github.com/Nafiz43/EvidenceBot
   ```

3. Create and activate a new virtual environment:
   ```bash
   conda create -n localGPT python=3.10.0
   conda activate localGPT
   ```

4. Install all the requirements:
   ```bash
   pip install -r requirements.txt
   ```

5. Install Ollama from the following [LINK](https://ollama.com/download).

6. Install Models using Ollama:
   ```bash
   ollama pull MODEL_NAME
   ```
   Replace MODEL_NAME with your desired model name. List of all the models is available at this [link](https://ollama.com/download).

7. Open the source directory, where the source code exists. Then, keep the documents that you want to analyze in the DATA_DOCUMENTS folder.

8. Open CLI (Command Line Interface) in the source directory and hit the following command:
   ```bash
   ollama pull MODEL_NAME
   ```

9. To run the app, navigate to the project directory and execute the following command in the command line:
   ```bash
   sh command.sh
   ```
