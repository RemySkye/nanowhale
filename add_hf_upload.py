#!/usr/bin/env python3
"""
Add HuggingFace upload cell to the notebook.
"""

import json
from datetime import datetime

def add_hf_upload(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find the Google Drive upload cell
    drive_cell_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'drive.mount' in ''.join(cell['source']):
            drive_cell_idx = i
            break
    
    if drive_cell_idx is None:
        print("ERROR: Could not find Google Drive upload cell")
        return False
    
    # Create HuggingFace upload cell
    hf_upload_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7.1 Upload to HuggingFace Hub\n",
            "\n",
            "Automatically creates a new repo and uploads the trained model."
        ]
    }
    
    hf_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Upload to HuggingFace Hub\n",
            "try:\n",
            "    from huggingface_hub import HfApi, login\n",
            "    from pathlib import Path\n",
            "    \n",
            "    # Login to HuggingFace\n",
            "    from google.colab import userdata\n",
            "    hf_token = userdata.get(\"HF_TOKEN\")\n",
            "    if hf_token:\n",
            "        login(token=hf_token)\n",
            "        print(\"✅ Logged into HuggingFace Hub\")\n",
            "        \n",
            "        # Create API instance\n",
            "        api = HfApi()\n",
            "        \n",
            "        # Get username for repo name\n",
            "        try:\n",
            "            user_info = api.whoami(token=hf_token)\n",
            "            username = user_info[\"name\"]\n",
            "        except:\n",
            "            username = \"user\"\n",
            "        \n",
            "        # Create repo name with timestamp\n",
            "        timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
            "        repo_name = f\"nanowhale-1b-{timestamp}\"\n",
            "        repo_id = f\"{username}/{repo_name}\"\n",
            "        \n",
            "        # Create repository\n",
            "        print(f\"Creating HuggingFace repo: {repo_id}\")\n",
            "        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)\n",
            "        \n",
            "        # Upload model files\n",
            "        print(\"Uploading model to HuggingFace...\")\n",
            "        api.upload_folder(\n",
            "            folder_path=src,\n",
            "            repo_id=repo_id,\n",
            "            commit_message=f\"Upload nanowhale-1b model - Step {step+1}\"\n",
            "        )\n",
            "        print(f\"✅ Model uploaded to https://huggingface.co/{repo_id}\")\n",
            "    else:\n",
            "        print(\"⚠️  HF_TOKEN not found in Colab secrets. Skipping HuggingFace upload.\")\n",
            "        print(\"To enable: Click the key icon in Colab, add a secret named 'HF_TOKEN' with your HuggingFace API token.\")\n",
            "except Exception as e:\n",
            "    print(f\"⚠️  HuggingFace upload failed: {e}\")"
        ]
    }
    
    # Insert the new cells after the Drive upload cell
    nb['cells'].insert(drive_cell_idx + 1, hf_upload_cell)
    nb['cells'].insert(drive_cell_idx + 2, hf_code_cell)
    
    # Save the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Added HuggingFace upload cell after Google Drive upload (index {drive_cell_idx})")
    return True

if __name__ == '__main__':
    path = 'colab/nanowhale_1b_moe_colab.ipynb'
    success = add_hf_upload(path)
    if success:
        print("✅ HuggingFace upload functionality added to notebook")
    else:
        print("❌ Failed to add HuggingFace upload")
