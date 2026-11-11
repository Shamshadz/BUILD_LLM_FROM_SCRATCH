# Local setup

This project uses a separate Python environment in `.venv` so book packages do
not affect other Python projects. The dependency list supports the full book,
Chapters 2 through 7.

## First-time installation (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name build-llm --display-name "Python (build-llm)"
```

## Each coding session

In VS Code, choose the `Python (build-llm)` kernel for notebooks. To open
Jupyter Lab from this folder:

```powershell
.\.venv\Scripts\jupyter lab
```

Install the local chapter modules in editable mode so they can be imported from
notebooks regardless of the notebook working directory:

```powershell
.\.venv\Scripts\python -m pip install -e .
```

For example, reusable Chapter 2 utilities can be imported with:

```python
from ch02 import SimpleTokenizerV1, GPTDatasetV1, create_dataloader_v1
```

To verify the environment:

```powershell
.\.venv\Scripts\python -c "import torch, tiktoken; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('tiktoken: OK')"
```

## Folder convention

Keep one folder per book chapter (for example, `ch02/`, `ch03/`). Put the
chapter notebook and its source text/data in that chapter's folder. Do not
commit `.venv`, notebook checkpoints, or cache files.

## Packages by chapter

| Chapters | Main additions |
| --- | --- |
| 2–4 | PyTorch, tiktoken, NumPy |
| 5 | TensorFlow (to load GPT-2 weights), tqdm |
| 6 | pandas, Matplotlib, TensorFlow |
| 7 | tqdm, psutil, Matplotlib, TensorFlow |

On Windows, TensorFlow runs on the CPU through the standard package. PyTorch
will report whether CUDA is available with the verification command above.
