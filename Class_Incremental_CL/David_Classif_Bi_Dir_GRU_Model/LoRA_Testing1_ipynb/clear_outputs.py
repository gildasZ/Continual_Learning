import os
import nbformat

# To clear the .ipynb list of output (excluding v1, v2)
exclude = {"CL_Classif_Bi_Dir_GRU_Model_v1.ipynb", 
           "CIL_Classif_Bi_Dir_GRU_Model_DynEx_CLoRA_lora_v2.ipynb"}

for filename in os.listdir("."):
    if filename.endswith(".ipynb") and filename not in exclude:
        print(f"Cleaning output of: {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # clear each cell output and execution_count
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None

        with open(filename, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
